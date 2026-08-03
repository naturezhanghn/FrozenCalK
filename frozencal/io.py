from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_RECORD_FIELDS = ["group_id", "source", "instruction", "edited"]


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read candidate records from JSONL, JSON, or CSV.

    Required fields are group_id, source, instruction, and edited. Optional
    fields such as candidate_id and model are preserved in outputs.
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    elif path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    else:
        raise ValueError(f"Unsupported input format: {path}")
    for i, row in enumerate(rows):
        missing = [key for key in REQUIRED_RECORD_FIELDS if key not in row]
        if missing:
            raise ValueError(f"Record {i} is missing required fields: {missing}")
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json" or path.suffix.lower() == ".jsonl":
        if path.suffix.lower() == ".jsonl":
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        else:
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def unique_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    groups = []
    for row in records:
        gid = row["group_id"]
        if gid in seen:
            continue
        seen.add(gid)
        groups.append(row)
    return groups
