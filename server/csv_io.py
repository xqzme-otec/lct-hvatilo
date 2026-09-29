import json
import csv
import io
import re
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
]

SPINE_REGION = "Поясничный отдел позвоночника"
HIP_REGION = "Проксимальный отдел бедра"

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(path: str):
    """
    Ключ для натуральной сортировки по имени файла:
    CR000002.png < CR000010.png < CR000100.png
    """
    name = Path(path).name
    return [int(p) if p.isdigit() else p.lower() for p in _NUM_RE.split(name)]


def _get_study_uid(study: dict) -> str:
    filename = study.get("filename", "")
    return Path(filename).stem if filename else ""


def _get_image_uid(file_path: str) -> str:
    return Path(file_path).stem if file_path else ""


def _compute_quality_prob(item: dict, violations: list[str]) -> float:
    if not item.get("ok", False):
        return 1.0
    if not violations:
        return 0.0

    probs: list[float] = []

    artifact = item.get("artifact") or {}
    if artifact.get("has_artifact"):
        conf = artifact.get("artifact_confidence") or {}
        vals = [v for v in conf.values() if isinstance(v, (int, float))]
        probs.append(float(max(vals)) if vals else 0.5)

    ferguson = (item.get("vertebrae") or {}).get("ferguson") or {}
    if ferguson.get("scoliosis"):
        probs.append(min(abs(ferguson.get("abs_deg", 0.0)) / 10.0, 1.0))

    vwr = item.get("vertebrae_w_rib") or {}
    if vwr.get("verdict") and vwr["verdict"] != "accept":
        probs.append(1.0)

    return round(max(probs) if probs else 0.5, 3)


def _map_region(region: str) -> str:
    key = (region or "").lower()
    if key == "spine":
        return SPINE_REGION
    if key == "hip" or key.startswith("hip_"):
        return HIP_REGION
    return region or ""

def _get_quality_and_violation(item: dict) -> tuple[int, str]:
    ok = item.get("ok", False)
    artifact = item.get("artifact") or {}
    has_artifact = bool(artifact.get("has_artifact", False))

    if not ok:
        return 1, item.get("error", "unknown_error")
    if has_artifact:
        return 1, "artifact"
    return 0, ""


def _get_time_of_processing_sec(item: dict) -> float:
    timings = item.get("timings") or {}
    total_ms = sum(v for v in timings.values() if isinstance(v, (int, float)))
    return round(total_ms / 1000.0, 3)


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _spine_violations(item: dict) -> list[str]:
    v = []

    # Некорректная укладка: пайплайн позвонков не принял результат
    vwr = item.get("vertebrae_w_rib") or {}
    verdict = vwr.get("verdict")
    if verdict and verdict != "accept":
        v.append("Некорректная укладка")

    # Ось позвоночника не выровнена (сколиоз по Ferguson)
    ferguson = (item.get("vertebrae") or {}).get("ferguson") or {}
    if ferguson.get("scoliosis"):
        v.append("Не выровнена ось позвоночника")

    # Посторонние предметы (артефакты)
    artifact = item.get("artifact") or {}
    if artifact.get("has_artifact"):
        v.append("Присутствуют посторонние предметы")

    # Если стадия упала и ошибка указывает на укладку
    if not item.get("ok", False):
        err = (item.get("error") or "").lower()
        if "position" in err or "layout" in err:
            v.append("Некорректная укладка")

    return _dedup(v)


def _hip_violations(item: dict) -> list[str]:
    v = []
    err = (item.get("error") or "").lower()

    # Область не поддерживается / не та, что ожидается
    if "unsupported" in err or "body region" in err:
        v.append("Некорректная область интереса")

    if "position" in err or "layout" in err:
        v.append("Некорректная укладка")

    return _dedup(v)


def json_to_rows(data: Any, sort: bool = True) -> list[dict]:
    """Превращает JSON-ответ эндпоинта в список строк для CSV."""
    studies = data if isinstance(data, list) else [data]
    rows = []

    for study in studies:
        study_uid = _get_study_uid(study)
        for item in study.get("results", []):
            file_path = item.get("file", "")
            raw_region = item.get("region", "") or ""
            key = raw_region.lower()

            if key == "spine":
                violations = _spine_violations(item)
            elif key == "hip" or key.startswith("hip_"):
                violations = _hip_violations(item)
            else:
                violations = []

            rows.append({
                "path_to_study": file_path,
                "study_uid": study_uid,
                "image_uid": _get_image_uid(file_path),
                "anatomical_region": _map_region(raw_region),
                "quality_class": 1 if violations else 0,
                "quality_prob": _compute_quality_prob(item, violations),
                "violation_type": ";".join(violations),
                "processing_status": "Success" if item.get("ok", False) else "Failure",
                "time_of_processing": _get_time_of_processing_sec(item),
            })

    if sort:
        rows.sort(key=lambda r: _natural_key(r["path_to_study"]))

    return rows


def rows_to_csv_string(rows: list[dict]) -> str:
    """Собирает CSV в строку (без записи на диск)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()

def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    return ("\ufeff" + rows_to_csv_string(rows)).encode("utf-8")

def enrich_json_with_csv(data: Any) -> dict:
    """
    Возвращает JSON-ответ, в котором кроме исходных данных
    есть готовый CSV-текст и структурированные строки.
    """
    rows = json_to_rows(data)
    csv_text = rows_to_csv_string(rows)

    return {
        "csv": csv_text,          # CSV целиком в виде строки
        "csv_rows": rows,         # те же данные, но как JSON-массив
        "csv_columns": FIELDNAMES,  # порядок колонок
        "source": data,           # исходный JSON-ответ эндпоинта
    }


def process_json_file(json_path: str, enriched_json_path: str | None = None,
                      csv_path: str | None = None) -> dict:
    """
    Читает JSON-файл, опционально сохраняет CSV и обогащённый JSON,
    возвращает обогащённый словарь.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched = enrich_json_with_csv(data)

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(enriched["csv"])

    if enriched_json_path:
        with open(enriched_json_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

    return enriched
