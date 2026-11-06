from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from .data import ROOT, estimate, load_catalog
from .llm import MockPlanner, OpenAIPlanner
from .repository import PlanRepository
from .trace import TraceWriter


class PlanningState(TypedDict, total=False):
    task_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    budget: float
    deadline: str
    plan: dict
    plan_version: int
    steps: dict
    tool_results: list[dict]
    estimated_cost: float | None
    estimated_hours: float | None
    actual_cost: float | None
    turn_count: int
    tool_call_count: int
    max_turns: int
    max_tool_calls: int
    status: str
    error: str | None
    saved_plan_id: str | None
    searched_ids: list[str]


STEP_NAMES = ("search_materials", "generate_script", "estimate_cost", "save_plan")


class ContentPlanningAgent:
    def __init__(self, runtime_dir: str | Path = "runtime", mode: str = "mock", *, now: datetime | None = None,
                 model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.now = now or datetime.now(timezone.utc)
        self.mode = mode
        self.planner = MockPlanner() if mode == "mock" else OpenAIPlanner(model, base_url, api_key)
        self.repository = PlanRepository(self.runtime_dir / "plans.sqlite")
        self.conn = sqlite3.connect(self.runtime_dir / "checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)

    def close(self):
        self.conn.close()

    def _tools(self, task_id: str, budget: float, deadline: str, fault_injection: dict[str, int] | None):
        failures = dict(fault_injection or {})
        counts: dict[str, int] = {}

        def maybe_fail(name: str):
            counts[name] = counts.get(name, 0) + 1
            if counts[name] <= failures.get(name, 0):
                return {"ok": False, "tool": name, "error": "injected tool failure", "retryable": True}
            return None

        @tool
        def search_materials(query: str) -> str:
            """Search the local material catalog. Use this before selecting materials."""
            if failed := maybe_fail("search_materials"):
                return json.dumps(failed, ensure_ascii=False)
            materials, _, _ = load_catalog()
            return json.dumps({"ok": True, "tool": "search_materials", "query": query, "materials": list(materials.values())}, ensure_ascii=False)

        @tool
        def generate_script(topic: str, material_ids: list[str], requirements: str,
                            state: Annotated[dict, InjectedState]) -> str:
            """Generate a marketing script using only provided, previously searched material IDs."""
            if failed := maybe_fail("generate_script"):
                return json.dumps(failed, ensure_ascii=False)
            if not set(material_ids) <= set(state.get("searched_ids", [])):
                return json.dumps({"ok": False, "tool": "generate_script", "error": "只能使用已检索素材", "retryable": True}, ensure_ascii=False)
            try:
                script = self.planner.generate_script(topic, material_ids, requirements)
                return json.dumps({"ok": True, "tool": "generate_script", "script": script, "adjusted": "adjusted" in requirements}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "tool": "generate_script", "error": str(exc), "retryable": True}, ensure_ascii=False)

        @tool
        def estimate_cost(material_ids: list[str], service_ids: list[str], state: Annotated[dict, InjectedState]) -> str:
            """Deterministically calculate cost and serial preparation time. Constraints are injected by the program."""
            if failed := maybe_fail("estimate_cost"):
                return json.dumps(failed, ensure_ascii=False)
            script = state.get("plan", {}).get("script", {})
            if not script:
                return json.dumps({"ok": False, "tool": "estimate_cost", "error": "必须先成功生成脚本", "retryable": True}, ensure_ascii=False)
            if sorted(material_ids) != sorted(script.get("material_ids", [])):
                return json.dumps({"ok": False, "tool": "estimate_cost", "error": "估算素材必须与当前脚本一致", "retryable": True}, ensure_ascii=False)
            return json.dumps({"tool": "estimate_cost", **estimate(material_ids, service_ids, budget, deadline, self.now)}, ensure_ascii=False)

        @tool
        def save_plan(plan: dict, state: Annotated[dict, InjectedState]) -> str:
            """Persist a complete validated plan. Program validates budget, deadline, script, and estimates."""
            if failed := maybe_fail("save_plan"):
                return json.dumps(failed, ensure_ascii=False)
            if plan.get("script") != state.get("plan", {}).get("script"):
                return json.dumps({"ok": False, "tool": "save_plan", "error": "只能保存当前已验证脚本", "retryable": True}, ensure_ascii=False)
            return json.dumps({"tool": "save_plan", **self.repository.save(task_id, plan, budget, deadline, self.now)}, ensure_ascii=False)

        return [search_materials, generate_script, estimate_cost, save_plan]

    def _build(self, task_id: str, budget: float, deadline: str, trace: TraceWriter, fault_injection=None):
        tools = self._tools(task_id, budget, deadline, fault_injection)
        self.planner.bind_tools(tools)

        def init_node(state: PlanningState):
            # On resume, preserve completed observations and allow the agent to decide from them.
            if state.get("task_id") == task_id and state.get("tool_results") and state.get("saved_plan_id"):
                return {"status": "completed", "error": None}
            if state.get("task_id") == task_id and state.get("tool_results"):
                return {"status": "running", "error": None}
            missing = [name for name, value in (("topic", state.get("topic")), ("budget", state.get("budget")), ("deadline", state.get("deadline"))) if value is None or value == ""]
            if missing:
                trace.write(task_id=task_id, turn=0, selected_tool=None, arguments={}, result_summary={}, step_status="skipped",
                            estimated_cost=None, next_action="final", decision_summary="input validation failed", error=f"缺少必填字段: {', '.join(missing)}")
                return {"status": "incomplete", "error": f"缺少必填字段: {', '.join(missing)}"}
            try:
                if float(state["budget"]) < 0:
                    raise ValueError("budget 必须非负")
                datetime.fromisoformat(str(state["deadline"]).replace("Z", "+00:00"))
            except ValueError as exc:
                return {"status": "incomplete", "error": f"输入无效: {exc}"}
            steps = {name: "pending" for name in STEP_NAMES}
            prompt = (f"你是内容任务规划 Agent。主题={state['topic']}；预算={state['budget']}；截止={state['deadline']}。"
                      "每轮只调用一个工具。顺序限制：成功检索后必须成功生成脚本，才可估算；估算必须使用脚本里的素材和制作方式；"
                      "脚本只能引用检索素材。每次方案变化后重新估算；只有预算和截止都通过才保存。"
                      "工具失败可重试、调整或结束。不得篡改约束；无解时直接结束。")
            task_message = f"请为主题“{state['topic']}”创建内容活动方案；预算为 {state['budget']}，截止时间为 {state['deadline']}。"
            return {"messages": [SystemMessage(content=prompt), HumanMessage(content=task_message)], "status": "running", "plan": {}, "plan_version": 0,
                    "steps": steps, "tool_results": [], "estimated_cost": None, "estimated_hours": None, "actual_cost": None,
                    "turn_count": 0, "tool_call_count": 0, "max_turns": state.get("max_turns", 12),
                    "max_tool_calls": state.get("max_tool_calls", 15), "error": None, "saved_plan_id": None, "searched_ids": []}

        def agent_node(state: PlanningState):
            if state.get("turn_count", 0) >= state.get("max_turns", 12):
                return {"status": "incomplete", "error": "达到 max_turns 上限"}
            if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 15):
                return {"status": "incomplete", "error": "达到 max_tool_calls 上限"}
            try:
                message = self.planner.decide(state["messages"])
            except Exception as exc:
                return {"status": "incomplete", "error": f"LLM 决策失败: {exc}"}
            calls = getattr(message, "tool_calls", []) or []
            if len(calls) > 1:
                return {"messages": [AIMessage(content="一次只允许一个工具调用")], "status": "incomplete", "error": "LLM 一轮选择多个工具"}
            if not calls and state.get("status") == "running":
                return {"messages": [message], "status": "incomplete", "error": "Agent 在保存前停止"}
            return {"messages": [message], "turn_count": state.get("turn_count", 0) + 1}

        def observe_node(state: PlanningState):
            results = list(state.get("tool_results", []))
            tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
            latest = tool_messages[-1] if tool_messages else None
            if latest is None:
                return {}
            try:
                result = json.loads(str(latest.content))
            except json.JSONDecodeError:
                result = {"ok": False, "tool": latest.name, "error": "工具返回不是 JSON"}
            # A tool result is processed exactly once per graph pass.
            if results and results[-1].get("call_id") == latest.tool_call_id:
                return {}
            result["call_id"] = latest.tool_call_id
            name, ok = result.get("tool", latest.name), result.get("ok", False)
            steps = dict(state.get("steps", {})); steps[name] = "success" if ok else "failed"
            update: dict[str, Any] = {"tool_results": results + [result], "steps": steps,
                                      "tool_call_count": state.get("tool_call_count", 0) + 1}
            if name == "search_materials" and ok:
                update["searched_ids"] = [x["id"] for x in result["materials"]]
            if name == "generate_script" and ok:
                script = result["script"]
                if not set(script.get("material_ids", [])) <= set(state.get("searched_ids", [])):
                    result["ok"] = False; result["error"] = "脚本包含未检索素材"; steps[name] = "failed"
                else:
                    update.update({"plan": {"script": script}, "plan_version": state.get("plan_version", 0) + 1,
                                   "estimated_cost": None, "estimated_hours": None})
                    steps["estimate_cost"] = "pending"; steps["save_plan"] = "pending"
            if name == "estimate_cost" and ok:
                update.update({"estimated_cost": result["estimated_cost"], "estimated_hours": result["estimated_hours"]})
            if name == "save_plan" and ok:
                update.update({"status": "completed", "saved_plan_id": result["plan_id"], "actual_cost": result["estimate"]["estimated_cost"]})
            trace.write(task_id=task_id, turn=state.get("turn_count", 0), selected_tool=name, arguments={},
                        result_summary={k: result.get(k) for k in ("ok", "error", "estimated_cost", "estimated_hours", "plan_id")},
                        step_status=steps.get(name), estimated_cost=update.get("estimated_cost", state.get("estimated_cost")),
                        next_action="agent" if update.get("status") != "completed" else "final", decision_summary="observable tool outcome", error=result.get("error"))
            return update

        def after_init(state): return "final" if state.get("status") == "incomplete" else "agent"
        def after_agent(state):
            message = state["messages"][-1] if state.get("messages") else None
            return "tools" if isinstance(message, AIMessage) and message.tool_calls and state.get("status") == "running" else "final"
        def after_observe(state): return "final" if state.get("status") in ("completed", "incomplete") else "agent"
        def final_node(state):
            if state.get("status") == "running":
                return {"status": "incomplete", "error": state.get("error") or "流程未完成"}
            return {}

        workflow = StateGraph(PlanningState)
        workflow.add_node("init", init_node); workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools)); workflow.add_node("observe", observe_node); workflow.add_node("final", final_node)
        workflow.add_edge(START, "init"); workflow.add_conditional_edges("init", after_init, {"agent": "agent", "final": "final"})
        workflow.add_conditional_edges("agent", after_agent, {"tools": "tools", "final": "final"})
        workflow.add_edge("tools", "observe"); workflow.add_conditional_edges("observe", after_observe, {"agent": "agent", "final": "final"}); workflow.add_edge("final", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, payload: dict, *, resume: bool = False, fault_injection: dict[str, int] | None = None) -> dict:
        task_id = payload.get("task_id") or str(uuid.uuid4())
        budget, deadline = payload.get("budget"), payload.get("deadline")
        # input validation needs a graph too; values are only used by tools after init validates them.
        budget_for_tools = float(budget) if budget not in (None, "") else 0.0
        deadline_for_tools = deadline or "1970-01-01T00:00:00+00:00"
        trace = TraceWriter(self.runtime_dir / "traces" / f"{task_id}.jsonl")
        app = self._build(task_id, budget_for_tools, deadline_for_tools, trace, fault_injection)
        config = {"configurable": {"thread_id": task_id}}
        if resume:
            snapshot = app.get_state(config)
            if snapshot.values:
                initial = dict(snapshot.values); initial["status"] = "running"; initial["error"] = None
                for key in ("max_turns", "max_tool_calls"):
                    if key in payload:
                        initial[key] = payload[key]
            else:
                initial = {**payload, "task_id": task_id}
        else:
            initial = {**payload, "task_id": task_id}
        state = app.invoke(initial, config)
        return self.result(state)

    @staticmethod
    def result(state: PlanningState) -> dict:
        return {key: state.get(key) for key in ("task_id", "status", "error", "plan", "plan_version", "steps", "tool_results", "estimated_cost", "estimated_hours", "actual_cost", "turn_count", "tool_call_count", "saved_plan_id")}
