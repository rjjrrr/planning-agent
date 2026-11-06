from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from .data import ROOT, estimate, load_catalog, parse_deadline
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
    plan_hash: str | None
    searched_ids: list[str]


STEP_NAMES = ("search_materials", "generate_script", "estimate_cost", "save_plan")


class ContentPlanningAgent:
    def __init__(self, runtime_dir: str | Path = "runtime", mode: str = "mock", *, now: datetime | None = None,
                 model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 planner: Any | None = None):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.now = now or datetime.now(timezone.utc)
        self.mode = mode
        self.planner = planner or (MockPlanner() if mode == "mock" else OpenAIPlanner(model, base_url, api_key))
        self.repository = PlanRepository(self.runtime_dir / "plans.sqlite")
        self.conn = sqlite3.connect(self.runtime_dir / "checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)

    def close(self):
        self.conn.close()

    def _tools(self, task_id: str, budget: float, deadline: str, fault_injection: dict[str, int] | None):
        failures = dict(fault_injection or {})
        counts: dict[str, int] = {}
        _, services, rules = load_catalog()

        def validate_script(script: dict, allowed_material_ids: list[str]) -> None:
            required = {"title", "creative_brief", "shots", "voiceover", "material_ids", "production_method"}
            if not isinstance(script, dict) or required - set(script):
                raise ValueError("脚本字段不完整")
            if not isinstance(script["material_ids"], list) or not script["material_ids"]:
                raise ValueError("脚本必须包含素材 ID")
            if not set(script["material_ids"]) <= set(allowed_material_ids):
                raise ValueError("脚本引用了未授权素材")
            if not isinstance(script["shots"], list) or not script["shots"]:
                raise ValueError("脚本必须包含至少一个分镜")
            if any(not isinstance(shot, dict) or shot.get("visual") not in script["material_ids"] for shot in script["shots"]):
                raise ValueError("分镜必须引用脚本素材 ID")
            method = script["production_method"]
            if not isinstance(method, str) or not any(service_id in method for service_id in rules["production_ids"]):
                raise ValueError("制作方式必须包含一个制作服务 ID")

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
            return json.dumps({"ok": True, "tool": "search_materials", "query": query, "materials": list(materials.values()),
                               "selection_rules": {"exactly_one_primary": rules["primary_visuals"], "max_music": 1,
                                                   "exactly_one_production": rules["production_ids"], "optional_service": "S001"}}, ensure_ascii=False)

        @tool
        def generate_script(topic: str, material_ids: list[Literal["V001", "V002", "V003", "M001", "M002"]], requirements: str,
                            state: Annotated[dict, InjectedState]) -> str:
            """Generate a marketing script. material_ids accept only media IDs V001/V002/V003/M001/M002, never service IDs."""
            if failed := maybe_fail("generate_script"):
                return json.dumps(failed, ensure_ascii=False)
            if not set(material_ids) <= set(state.get("searched_ids", [])):
                return json.dumps({"ok": False, "tool": "generate_script", "error": "只能使用已检索素材", "retryable": True}, ensure_ascii=False)
            try:
                script = self.planner.generate_script(topic, material_ids, requirements)
                validate_script(script, material_ids)
                return json.dumps({"ok": True, "tool": "generate_script", "script": script,
                                   "plan_version": state.get("plan_version", 0) + 1,
                                   "adjusted": "adjusted" in requirements}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "tool": "generate_script", "error": str(exc), "retryable": True}, ensure_ascii=False)

        @tool
        def estimate_cost(material_ids: list[Literal["V001", "V002", "V003", "M001", "M002"]],
                          service_ids: list[Literal["S001", "S002", "S003", "S004"]],
                          state: Annotated[dict, InjectedState]) -> str:
            """Deterministically calculate cost and serial preparation time. Use only IDs from the current script."""
            if failed := maybe_fail("estimate_cost"):
                return json.dumps(failed, ensure_ascii=False)
            script = state.get("plan", {}).get("script", {})
            if not script:
                return json.dumps({"ok": False, "tool": "estimate_cost", "error": "必须先成功生成脚本", "retryable": True}, ensure_ascii=False)
            if sorted(material_ids) != sorted(script.get("material_ids", [])):
                return json.dumps({"ok": False, "tool": "estimate_cost", "error": "估算素材必须与当前脚本一致", "retryable": True}, ensure_ascii=False)
            script_service_ids = [service_id for service_id in services if service_id in script.get("production_method", "")]
            if sorted(service_ids) != sorted(script_service_ids):
                return json.dumps({"ok": False, "tool": "estimate_cost", "error": "估算服务必须与当前脚本制作方式一致", "retryable": True}, ensure_ascii=False)
            return json.dumps({"tool": "estimate_cost", "plan_version": state.get("plan_version"),
                               **estimate(material_ids, service_ids, budget, deadline, self.now)}, ensure_ascii=False)

        @tool
        def save_plan(state: Annotated[dict, InjectedState], plan: dict | None = None) -> str:
            """Persist the current validated plan. Call without plan to save program-held State; supplied plans must exactly match it."""
            if failed := maybe_fail("save_plan"):
                return json.dumps(failed, ensure_ascii=False)
            current = state.get("plan", {})
            expected = {key: current.get(key) for key in ("script", "material_ids", "service_ids", "estimate")}
            expected["plan_version"] = state.get("plan_version")
            if plan is None:
                plan = expected
            elif plan.get("script") != current.get("script"):
                return json.dumps({"ok": False, "tool": "save_plan", "error": "只能保存当前已验证脚本", "retryable": True}, ensure_ascii=False)
            if any(plan.get(key) != value for key, value in expected.items()):
                return json.dumps({"ok": False, "tool": "save_plan", "error": "方案不是当前已估算版本", "retryable": True}, ensure_ascii=False)
            estimate_result = current.get("estimate", {})
            if not estimate_result.get("within_budget") or not estimate_result.get("within_deadline"):
                return json.dumps({"ok": False, "tool": "save_plan", "error": "当前方案尚未通过预算和截止时间", "retryable": True}, ensure_ascii=False)
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
                summary = {"plan_version": state.get("plan_version"), "steps": state.get("steps"),
                           "estimated_cost": state.get("estimated_cost"), "estimated_hours": state.get("estimated_hours")}
                return {"messages": [HumanMessage(content=f"任务从 checkpoint 恢复。请基于当前状态继续，不要重复成功工具，也不要沿用之前的停止决定。状态：{json.dumps(summary, ensure_ascii=False)}")],
                        "status": "running", "error": None}
            missing = [name for name, value in (("topic", state.get("topic")), ("budget", state.get("budget")), ("deadline", state.get("deadline"))) if value is None or value == ""]
            if missing:
                trace.write(task_id=task_id, thread_id=task_id, event="input_validation", turn=0, selected_tool=None, arguments={}, result_summary={}, step_status="skipped",
                            estimated_cost=None, next_action="final", decision_summary="input validation failed", error=f"缺少必填字段: {', '.join(missing)}")
                return {"status": "incomplete", "error": f"缺少必填字段: {', '.join(missing)}"}
            try:
                if float(state["budget"]) < 0:
                    raise ValueError("budget 必须非负")
                parse_deadline(str(state["deadline"]))
            except ValueError as exc:
                return {"status": "incomplete", "error": f"输入无效: {exc}"}
            steps = {name: "pending" for name in STEP_NAMES}
            prompt = (f"你是内容任务规划 Agent。主题={state['topic']}；预算={state['budget']}；截止={state['deadline']}。"
                      "每轮只调用一个工具。顺序限制：成功检索后必须成功生成脚本，才可估算；估算必须使用脚本里的素材和制作方式；"
                      "脚本只能引用检索素材。每次方案变化后重新估算；只有预算和截止都通过才保存。"
                      "组合规则：恰好一个 V001/V002/V003，最多一个 M001/M002，恰好一个 S002/S003/S004，S001 可选。"
                      "估算因组合无效、超预算或超时失败时，不得直接停止；应调用 generate_script 创建不同的合法方案并再次估算。"
                      "准备保存时调用 save_plan 时不要传 plan 参数：运行时会从当前已验证 State 构造方案，禁止自行重建嵌套方案。"
                      "只有工具明确返回无可行候选时才可结束。工具失败可重试、调整或结束。不得篡改约束。"
                      "In English: use exactly one primary visual, at most one music item, and one production service. An invalid or over-budget estimate requires a new valid script and re-estimate; do not stop until a tool explicitly says no feasible candidate exists.")
            task_message = f"请为主题“{state['topic']}”创建内容活动方案；预算为 {state['budget']}，截止时间为 {state['deadline']}。"
            return {"messages": [SystemMessage(content=prompt), HumanMessage(content=task_message)], "status": "running", "plan": {}, "plan_version": 0,
                    "steps": steps, "tool_results": [], "estimated_cost": None, "estimated_hours": None, "actual_cost": None,
                    "turn_count": 0, "tool_call_count": 0, "max_turns": state.get("max_turns", 12),
                    "max_tool_calls": state.get("max_tool_calls", 15), "error": None, "saved_plan_id": None,
                    "plan_hash": None, "searched_ids": []}

        def agent_node(state: PlanningState):
            if state.get("status") != "running":
                return {}
            if state.get("turn_count", 0) >= state.get("max_turns", 12):
                return {"status": "incomplete", "error": "达到 max_turns 上限"}
            if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 15):
                return {"status": "incomplete", "error": "达到 max_tool_calls 上限"}
            try:
                message = self.planner.decide(state["messages"])
            except Exception as exc:
                trace.write(task_id=task_id, thread_id=task_id, event="decision", turn=state.get("turn_count", 0),
                            selected_tool=None, arguments={}, result_summary={}, step_status=None,
                            estimated_cost=state.get("estimated_cost"), next_action="final",
                            decision_summary="LLM invocation failed", error=f"LLM 决策失败: {exc}")
                return {"status": "incomplete", "error": f"LLM 决策失败: {exc}"}
            calls = getattr(message, "tool_calls", []) or []
            if len(calls) > 1:
                return {"messages": [AIMessage(content="一次只允许一个工具调用")], "status": "incomplete", "error": "LLM 一轮选择多个工具"}
            if not calls and state.get("status") == "running":
                trace.write(task_id=task_id, thread_id=task_id, event="decision", turn=state.get("turn_count", 0), selected_tool=None,
                            arguments={}, result_summary={}, step_status=None, estimated_cost=state.get("estimated_cost"),
                            next_action="final", decision_summary="LLM chose to stop", error="Agent 在保存前停止")
                return {"messages": [message], "status": "incomplete", "error": "Agent 在保存前停止"}
            selected = calls[0].get("name")
            steps = dict(state.get("steps", {}))
            if selected in STEP_NAMES:
                steps[selected] = "running"
            trace.write(task_id=task_id, thread_id=task_id, event="decision", turn=state.get("turn_count", 0) + 1,
                        selected_tool=selected, arguments=calls[0].get("args", {}), result_summary={}, step_status=steps.get(selected),
                        estimated_cost=state.get("estimated_cost"), next_action=selected,
                        decision_summary=f"LLM selected {selected}", error=None)
            return {"messages": [message], "steps": steps, "turn_count": state.get("turn_count", 0) + 1}

        def observe_node(state: PlanningState):
            results = list(state.get("tool_results", []))
            tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
            latest = tool_messages[-1] if tool_messages else None
            if latest is None:
                return {}
            try:
                result = json.loads(str(latest.content))
            except json.JSONDecodeError:
                result = {"ok": False, "tool": latest.name, "error": f"工具返回无效: {str(latest.content)[:500]}"}
            # A tool result is processed exactly once per graph pass.
            if results and results[-1].get("call_id") == latest.tool_call_id:
                return {}
            result["call_id"] = latest.tool_call_id
            name, ok = result.get("tool", latest.name), result.get("ok", False)
            step_name = name if name in STEP_NAMES else None
            steps = dict(state.get("steps", {}))
            if step_name:
                steps[step_name] = "success" if ok else "failed"
            if not ok and not result.get("error") and result.get("errors"):
                result["error"] = "; ".join(result["errors"])
            update: dict[str, Any] = {"tool_results": results + [result], "steps": steps,
                                      "tool_call_count": state.get("tool_call_count", 0) + 1}
            if name == "search_materials" and ok:
                update["searched_ids"] = [x["id"] for x in result["materials"]]
            if name == "generate_script" and ok:
                script = result["script"]
                if not set(script.get("material_ids", [])) <= set(state.get("searched_ids", [])):
                    result["ok"] = False; result["error"] = "脚本包含未检索素材"; steps[name] = "failed"
                else:
                    update.update({"plan": {"script": script}, "plan_version": result["plan_version"],
                                   "estimated_cost": None, "estimated_hours": None})
                    steps["estimate_cost"] = "pending"; steps["save_plan"] = "pending"
            if name == "estimate_cost" and ok:
                estimate_for_plan = {key: value for key, value in result.items() if key != "call_id"}
                plan = dict(state.get("plan", {}))
                plan.update({"material_ids": result["material_ids"], "service_ids": result["service_ids"], "estimate": estimate_for_plan})
                update.update({"plan": plan, "estimated_cost": result["estimated_cost"], "estimated_hours": result["estimated_hours"]})
            if name == "save_plan" and ok:
                update.update({"status": "completed", "saved_plan_id": result["plan_id"], "plan_hash": result["plan_hash"],
                               "actual_cost": result["estimate"]["estimated_cost"]})
            selected_call = next((call for message in reversed(state["messages"]) if isinstance(message, AIMessage)
                                  for call in (message.tool_calls or []) if call.get("id") == latest.tool_call_id), None)
            trace.write(task_id=task_id, thread_id=task_id, event="tool_result", turn=state.get("turn_count", 0), selected_tool=name,
                        arguments=(selected_call or {}).get("args", {}),
                        result_summary={k: result.get(k) for k in ("ok", "error", "estimated_cost", "estimated_hours", "plan_id")},
                        step_status=steps.get(name), estimated_cost=update.get("estimated_cost", state.get("estimated_cost")),
                        next_action="pending_llm_decision" if update.get("status") != "completed" else "final",
                        decision_summary=f"LLM selected {name}", error=result.get("error"))
            return update

        def after_init(state): return "agent" if state.get("status") == "running" else "final"
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
        return {key: state.get(key) for key in ("task_id", "status", "error", "plan", "plan_version", "steps", "tool_results", "estimated_cost", "estimated_hours", "actual_cost", "turn_count", "tool_call_count", "saved_plan_id", "plan_hash")}
