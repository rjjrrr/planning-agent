from datetime import datetime, timezone

from content_planning.data import estimate
from content_planning.repository import PlanRepository


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
DEADLINE = "2030-01-02T00:00:00+00:00"


def test_deterministic_estimate_and_combination_rules():
    result = estimate(["V001", "M001"], ["S001", "S002"], 12, DEADLINE, NOW)
    assert (result["estimated_cost"], result["estimated_hours"], result["within_budget"]) == (12, 8, True)
    invalid = estimate(["V001", "V002"], ["S002"], 99, DEADLINE, NOW)
    assert not invalid["ok"] and "主视觉" in invalid["errors"][0]


def test_deadline_requires_timezone():
    try:
        estimate(["V002", "M001"], ["S001", "S002"], 10, "2030-01-02T00:00:00", NOW)
    except ValueError as exc:
        assert "时区" in str(exc)
    else:
        raise AssertionError("naive deadline should be rejected")


def test_repository_rejects_tampered_estimate(tmp_path):
    repo = PlanRepository(tmp_path / "plans.sqlite")
    script = {"title": "x", "creative_brief": "x", "shots": [{}], "voiceover": "x", "material_ids": ["V002", "M001"], "production_method": "S001+S002"}
    result = repo.save("t", {"script": script, "material_ids": ["V002", "M001"], "service_ids": ["S001", "S002"], "estimate": {"estimated_cost": 0, "estimated_hours": 0}, "plan_version": 1}, 10, DEADLINE, NOW)
    assert not result["ok"] and "不一致" in result["error"]
