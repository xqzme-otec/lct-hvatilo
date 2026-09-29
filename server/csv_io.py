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


def json_to_rows(data: Any, sort: bool = True) -> list[dict]:
    """Превращает JSON-ответ эндпоинта в список строк для CSV."""
    studies = data if isinstance(data, list) else [data]
    rows = []

    for study in studies:
        study_uid = _get_study_uid(study)
        for item in study.get("results", []):
            file_path = item.get("file", "")
            quality_class, violation_type = _get_quality_and_violation(item)

            rows.append({
                "path_to_study": file_path,
                "study_uid": study_uid,
                "image_uid": _get_image_uid(file_path),
                "anatomical_region": item.get("region", ""),
                "quality_class": quality_class,
                "violation_type": violation_type,
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
