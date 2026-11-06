from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_catalog(data_dir: Path | None = None) -> tuple[dict, dict, dict]:
    base = data_dir or ROOT / "data"
    materials = {x["id"]: x for x in json.loads((base / "materials.json").read_text(encoding="utf-8"))}
    services = {x["id"]: x for x in json.loads((base / "services.json").read_text(encoding="utf-8"))}
    rules = json.loads((base / "rules.json").read_text(encoding="utf-8"))
    return materials, services, rules


def parse_deadline(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline 必须带时区，例如 +00:00 或 Z")
    return parsed


def estimate(material_ids: list[str], service_ids: list[str], budget: float, deadline: str, now: datetime, data_dir: Path | None = None) -> dict:
    materials, services, rules = load_catalog(data_dir)
    errors: list[str] = []
    if len(material_ids) != len(set(material_ids)) or len(service_ids) != len(set(service_ids)):
        errors.append("素材或服务不能重复")
    unknown = [x for x in material_ids if x not in materials] + [x for x in service_ids if x not in services]
    if unknown:
        errors.append(f"无效 ID: {','.join(unknown)}")
    visual = [x for x in material_ids if x in rules["primary_visuals"]]
    music = [x for x in material_ids if x in rules["music_ids"]]
    production = [x for x in service_ids if x in rules["production_ids"]]
    if len(visual) != 1:
        errors.append("必须恰好选择一种主视觉素材")
    if len(music) > rules["max_music"]:
        errors.append("背景音乐最多一种")
    if len(production) != 1:
        errors.append("必须恰好选择一种制作方式")
    total_cost = sum(materials[x]["cost"] for x in material_ids if x in materials) + sum(services[x]["cost"] for x in service_ids if x in services)
    total_hours = sum(materials[x]["hours"] for x in material_ids if x in materials) + sum(services[x]["hours"] for x in service_ids if x in services)
    finish = now + timedelta(hours=total_hours)
    valid = not errors
    return {"ok": valid, "material_ids": material_ids, "service_ids": service_ids,
            "estimated_cost": total_cost, "estimated_hours": total_hours,
            "estimated_finish_time": finish.isoformat(), "within_budget": valid and total_cost <= budget,
            "within_deadline": valid and finish <= parse_deadline(deadline), "errors": errors}
