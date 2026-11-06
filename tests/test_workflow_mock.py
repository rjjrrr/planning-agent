import sqlite3
from datetime import datetime, timezone
import json

from langchain_core.messages import AIMessage
from content_planning.graph import ContentPlanningAgent


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
DEADLINE = "2030-01-02T00:00:00+00:00"


def run(tmp_path, **payload):
    agent = ContentPlanningAgent(tmp_path, "mock", now=NOW)
    try:
        return agent.run({"task_id": "case", "topic": "夏季新品", "budget": 20, "deadline": DEADLINE, **payload})
    finally:
        agent.close()


def tool_call(name, args, call_id):
    return AIMessage(content=f"select {name}", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedPlanner:
    """Controlled planner used only to test invalid model decisions through the real graph."""
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.index = 0

    def bind_tools(self, tools):
        self.tools = tools

    def decide(self, messages):
        if self.index >= len(self.decisions):
            return AIMessage(content="stop")
        decision = self.decisions[self.index]
        self.index += 1
        return decision

    def generate_script(self, topic, material_ids, requirements):
        return {"title": topic, "creative_brief": "brief", "shots": [{"visual": material_ids[0], "duration_seconds": 8}],
                "voiceover": "voice", "material_ids": material_ids, "production_method": requirements}


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
        payload["max_turns"] = 2
        failed = agent.run(payload, fault_injection={"generate_script": 1})
        resumed = agent.run({**payload, "max_turns": 12}, resume=True)
    finally:
        agent.close()
    assert failed["status"] == "incomplete"
    assert failed["tool_results"][-1]["tool"] == "generate_script"
    assert failed["tool_results"][-1]["ok"] is False
    assert resumed["status"] == "completed"
    assert resumed["tool_results"][0]["tool"] == "search_materials"
    assert sum(item["tool"] == "search_materials" for item in resumed["tool_results"]) == 1


def test_t7_no_feasible_candidate_is_incomplete(tmp_path):
    result = run(tmp_path, budget=0)
    assert result["status"] == "incomplete" and result["saved_plan_id"] is None


def test_t8_turn_limit_stops_safely(tmp_path):
    result = run(tmp_path, max_turns=1)
    assert result["status"] == "incomplete" and "max_turns" in result["error"]


def test_t9_script_without_a_fresh_estimate_cannot_be_saved(tmp_path):
    planner = ScriptedPlanner([
        tool_call("search_materials", {"query": "new product"}, "t9-1"),
        tool_call("generate_script", {"topic": "new product", "material_ids": ["V002", "M001"], "requirements": "S001+S002"}, "t9-2"),
        tool_call("save_plan", {"plan": {"script": {"title": "new product", "creative_brief": "brief", "shots": [{"visual": "V002", "duration_seconds": 8}], "voiceover": "voice", "material_ids": ["V002", "M001"], "production_method": "S001+S002"}, "plan_version": 1}}, "t9-3"),
    ])
    agent = ContentPlanningAgent(tmp_path, "mock", now=NOW, planner=planner)
    try:
        result = agent.run({"task_id": "t9", "topic": "new product", "budget": 10, "deadline": DEADLINE})
    finally:
        agent.close()
    with sqlite3.connect(tmp_path / "plans.sqlite") as con:
        count = con.execute("SELECT count(*) FROM plans WHERE task_id='t9'").fetchone()[0]
    save = next(item for item in result["tool_results"] if item["tool"] == "save_plan")
    assert result["status"] == "incomplete" and not save["ok"] and count == 0
    assert result["steps"]["estimate_cost"] == "pending"


def test_t10_unknown_tool_result_does_not_pollute_step_state(tmp_path):
    planner = ScriptedPlanner([tool_call("not_a_tool", {}, "t10-1")])
    agent = ContentPlanningAgent(tmp_path, "mock", now=NOW, planner=planner)
    try:
        result = agent.run({"task_id": "t10", "topic": "new product", "budget": 10, "deadline": DEADLINE})
    finally:
        agent.close()
    assert result["status"] == "incomplete"
    assert set(result["steps"]) == {"search_materials", "generate_script", "estimate_cost", "save_plan"}
    assert result["tool_results"][0]["ok"] is False


def test_trace_records_actual_tool_arguments_and_thread(tmp_path):
    result = run(tmp_path)
    trace_path = tmp_path / "traces" / f"{result['task_id']}.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["thread_id"] == result["task_id"]
    assert rows[0]["selected_tool"] == "search_materials"
    assert rows[0]["arguments"] == {"query": "短视频营销素材"}
