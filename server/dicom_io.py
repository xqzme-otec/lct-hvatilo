"""Reading DXA DICOM files into a plain, safe-to-use structure.

Every file becomes a `DxaImage`. Reading never raises: a broken file yields a
`DxaImage` with `error` set, so batch processing can report it as Failure and go on.
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pydicom
from PIL import Image

# GE Lunar Prodigy exports in this dataset have no PixelSpacing; ExposedArea (mm)
# divided by the matrix size gives ~0.6 mm/px on most files. Some files carry
# ExposedArea = (520, 478) (a different quantity — a 52 cm wide hip frame is impossible),
# so derived values outside the plausible range fall back to the default.
DEFAULT_MM_PER_PX = 0.6
PLAUSIBLE_MM_PER_PX = (0.4, 0.9)

# pydicom complains about every UID in these files (non-conformant characters);
# the warnings are harmless and would flood the log on a 500-file batch.
warnings.filterwarnings("ignore", message="Invalid value for VR UI")
logging.getLogger("pydicom").setLevel(logging.ERROR)


@dataclass
class DxaImage:
    path: str
    study_uid: str = ""
    series_uid: str = ""
    image_uid: str = ""
    instance_number: Optional[int] = None
    pixels: Optional[np.ndarray] = None  # uint8 HxW
    mm_per_px: float = DEFAULT_MM_PER_PX
    mm_per_px_source: str = "default"  # "ExposedArea" | "PixelSpacing" | "default"
    manufacturer: str = ""
    model: str = ""
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.pixels is not None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.pixels.shape[:2]) if self.pixels is not None else (0, 0)

    @property
    def pixel_hash(self) -> str:
        """md5 of raw pixels — used to drop byte-identical duplicates."""
        return hashlib.md5(self.pixels.tobytes()).hexdigest() if self.pixels is not None else ""


def _to_uint8(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Normalise any bit depth / photometric interpretation to uint8 MONOCHROME2."""
    arr = np.asarray(arr)
    if arr.ndim == 3:  # RGB or multi-frame: take first channel/frame
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(arr, dtype=np.uint8)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = 255 - arr
    return np.ascontiguousarray(arr)


def _mm_per_px(ds: pydicom.Dataset) -> tuple[float, str]:
    spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
    if spacing:
        try:
            return float(spacing[0]), "PixelSpacing"
        except (TypeError, ValueError):
            pass
    area = ds.get("ExposedArea")
    try:
        if area and len(area) == 2 and int(area[0]) > 0 and int(ds.Columns) > 0:
            v = float(area[0]) / float(ds.Columns)
            if PLAUSIBLE_MM_PER_PX[0] <= v <= PLAUSIBLE_MM_PER_PX[1]:
                return v, "ExposedArea"
    except (TypeError, ValueError):
        pass
    return DEFAULT_MM_PER_PX, "default"


def load_dicom(path: str | Path) -> DxaImage:
    """Read one DICOM file. Never raises; failures are recorded in `.error`."""
    img = DxaImage(path=str(path))
    try:
        ds = pydicom.dcmread(str(path), force=True)
        img.study_uid = str(ds.get("StudyInstanceUID", ""))
        img.series_uid = str(ds.get("SeriesInstanceUID", ""))
        img.image_uid = str(ds.get("SOPInstanceUID", ""))
        img.manufacturer = str(ds.get("Manufacturer", ""))
        img.model = str(ds.get("ManufacturerModelName", ""))
        try:
            img.instance_number = int(ds.get("InstanceNumber"))
        except (TypeError, ValueError):
            pass
        img.meta = {
            "Modality": str(ds.get("Modality", "")),
            "SeriesDescription": str(ds.get("SeriesDescription", "")),
            "PatientOrientation": "/".join(ds.get("PatientOrientation", []) or []),
            "ExposedArea": list(ds.get("ExposedArea", []) or []),
            "Rows": int(ds.get("Rows", 0) or 0),
            "Columns": int(ds.get("Columns", 0) or 0),
        }
        if "PixelData" not in ds:
            raise ValueError("no PixelData")
        img.pixels = _to_uint8(ds.pixel_array, ds)
        img.mm_per_px, img.mm_per_px_source = _mm_per_px(ds)
    except Exception as e:  # noqa: BLE001 — any failure must be reported, not raised
        img.error = f"{type(e).__name__}: {e}"
    return img


def find_dicom_files(root: str | Path) -> list[Path]:
    """All candidate DICOM files under root (by extension, or extension-less files with DICM magic)."""
    root = Path(root)
    if root.is_file():
        return [root]
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in (".dcm", ".dicom"):
            out.append(p)
        elif p.suffix == "":
            try:
                with open(p, "rb") as f:
                    f.seek(128)
                    if f.read(4) == b"DICM":
                        out.append(p)
            except OSError:
                pass
    return out


def iter_dicoms(root: str | Path) -> Iterator[DxaImage]:
    for p in find_dicom_files(root):
        yield load_dicom(p)


def _dicom_to_png(dicom_path: Path, output_path: Path) -> None:
    """Конвертирует один DICOM-файл в PNG с применением rescale и windowing."""
    ds = pydicom.dcmread(str(dicom_path), force=True)

    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Cannot read pixel data from {dicom_path.name}: {exc}") from exc

    # Rescale (HU для CT и т.п.)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    # Windowing
    def _first(v):
        # DICOM может вернуть MultiValue / список
        if isinstance(v, (list, tuple)):
            return float(v[0])
        try:
            return float(v[0]) if not isinstance(v, (int, float)) else float(v)
        except Exception:
            return float(v)

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        wc, ww = _first(wc), _first(ww)
        low = wc - ww / 2.0
        high = wc + ww / 2.0
    else:
        low = float(np.min(arr))
        high = float(np.max(arr))

    arr = np.clip(arr, low, high)
    if high > low:
        arr = (arr - low) / (high - low) * 255.0
    else:
        arr = np.zeros_like(arr)

    # MONOCHROME1 — инвертированная яркость
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255.0 - arr

    arr = arr.astype(np.uint8)

    # Если вдруг цветной (RGB/YBR) — PIL сам разберётся, если shape (H, W, 3)
    Image.fromarray(arr).save(output_path, format="PNG")