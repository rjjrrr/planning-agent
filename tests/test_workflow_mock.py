import sqlite3
from datetime import datetime, timezone

from content_planning.graph import ContentPlanningAgent


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
DEADLINE = "2030-01-02T00:00:00+00:00"


def run(tmp_path, **payload):
    agent = ContentPlanningAgent(tmp_path, "mock", now=NOW)
    try:
        return agent.run({"task_id": "case", "topic": "夏季新品", "budget": 20, "deadline": DEADLINE, **payload})
    finally:
        agent.close()


def test_t1_normal_budget_and_deadline_saves_plan(tmp_path):
    result = run(tmp_path)
    assert result["status"] == "completed"
    assert result["saved_plan_id"] and result["actual_cost"] == 12


def test_t2_missing_required_field_is_explicit(tmp_path):
    result = run(tmp_path, budget=None)
    assert result["status"] == "incomplete" and "budget" in result["error"]


def test_t3_injected_failure_is_observed_and_retried(tmp_path):
    agent = ContentPlanningAgent(tmp_path, "mock", now=NOW)
    try:
        result = agent.run({"task_id": "failure", "topic": "夏季新品", "budget": 20, "deadline": DEADLINE}, fault_injection={"search_materials": 1})
    finally:
        agent.close()
    assert result["status"] == "completed"
    assert result["tool_results"][0]["ok"] is False
    assert result["steps"]["search_materials"] == "success"


def test_t4_over_budget_is_adjusted_then_reestimated(tmp_path):
    result = run(tmp_path, budget=10)
    estimates = [x["estimated_cost"] for x in result["tool_results"] if x["tool"] == "estimate_cost"]
    assert result["status"] == "completed" and estimates == [12, 8]
    assert result["estimated_cost"] == 8 and result["plan"]["script"]["material_ids"][0] == "V002"


def test_t5_same_task_has_one_persisted_plan(tmp_path):
    first = run(tmp_path)
    second = run(tmp_path)
    with sqlite3.connect(tmp_path / "plans.sqlite") as con:
        count = con.execute("SELECT count(*) FROM plans WHERE task_id='case'").fetchone()[0]
    assert first["saved_plan_id"] == second["saved_plan_id"] and count == 1


def test_t6_checkpoint_resume_preserves_search(tmp_path):
    agent = ContentPlanningAgent(tmp_path, "mock", now=NOW)
    payload = {"task_id": "resume", "topic": "夏季新品", "budget": 20, "deadline": DEADLINE, "max_turns": 1}
    try:
        failed = agent.run(payload)
        resumed = agent.run({**payload, "max_turns": 12}, resume=True)
    finally:
        agent.close()
    assert failed["status"] == "incomplete"
    assert resumed["status"] == "completed"
    assert resumed["tool_results"][0]["tool"] == "search_materials"


def test_t7_no_feasible_candidate_is_incomplete(tmp_path):
    result = run(tmp_path, budget=0)
    assert result["status"] == "incomplete" and result["saved_plan_id"] is None


def test_t8_turn_limit_stops_safely(tmp_path):
    result = run(tmp_path, max_turns=1)
    assert result["status"] == "incomplete" and "max_turns" in result["error"]
