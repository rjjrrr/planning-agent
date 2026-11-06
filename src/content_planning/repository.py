from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .data import estimate


class PlanRepository:
    def __init__(self, path: Path, data_dir: Path | None = None):
        self.path, self.data_dir = path, data_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS plans (plan_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, plan_hash TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(task_id, plan_hash))")

    def save(self, task_id: str, plan: dict, budget: float, deadline: str, now) -> dict:
        required = {"script", "material_ids", "service_ids", "estimate", "plan_version"}
        missing = sorted(required - set(plan))
        if missing:
            return {"ok": False, "error": f"方案缺少字段: {','.join(missing)}"}
        script = plan["script"]
        script_required = {"title", "creative_brief", "shots", "voiceover", "material_ids", "production_method"}
        if not isinstance(script, dict) or script_required - set(script):
            return {"ok": False, "error": "脚本不完整"}
        if sorted(script["material_ids"]) != sorted(plan["material_ids"]):
            return {"ok": False, "error": "脚本引用素材与方案不一致"}
        actual = estimate(plan["material_ids"], plan["service_ids"], budget, deadline, now, self.data_dir)
        supplied = plan["estimate"]
        if not actual["ok"] or not actual["within_budget"] or not actual["within_deadline"]:
            return {"ok": False, "error": "预算、时间或组合校验未通过", "estimate": actual}
        if supplied.get("estimated_cost") != actual["estimated_cost"] or supplied.get("estimated_hours") != actual["estimated_hours"]:
            return {"ok": False, "error": "估算结果与程序计算不一致", "estimate": actual}
        canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        plan_id = f"plan_{hashlib.sha256((task_id + digest).encode()).hexdigest()[:12]}"
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT plan_id FROM plans WHERE task_id=? AND plan_hash=?", (task_id, digest)).fetchone()
            if row:
                return {"ok": True, "plan_id": row[0], "idempotent": True, "estimate": actual}
            con.execute("INSERT INTO plans VALUES (?,?,?,?)", (plan_id, task_id, digest, canonical))
        return {"ok": True, "plan_id": plan_id, "idempotent": False, "estimate": actual}
