from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI


def _tool(name: str, args: dict, ident: str) -> AIMessage:
    return AIMessage(content=f"选择 {name}", tool_calls=[{"name": name, "args": args, "id": ident, "type": "tool_call"}])


def _results(messages) -> list[dict]:
    out = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                out.append(json.loads(str(msg.content)))
            except json.JSONDecodeError:
                pass
    return out


class OpenAIPlanner:
    """Real, tool-calling planner. It never receives authority to alter constraints."""
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.client = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                 base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                                 api_key=api_key or os.getenv("OPENAI_API_KEY"), temperature=0)
        self.bound = None

    def bind_tools(self, tools):
        self.bound = self.client.bind_tools(tools)

    def decide(self, messages):
        if self.bound is None:
            raise RuntimeError("tools have not been bound")
        return self.bound.invoke(messages)

    def generate_script(self, topic: str, material_ids: list[str], requirements: str) -> dict:
        prompt = f'''为短视频营销生成严格 JSON，不能用未列出的素材。主题：{topic}\n允许素材 ID：{material_ids}\n要求：{requirements}\n返回字段 title, creative_brief, shots(数组，每项含visual和duration_seconds), voiceover, material_ids, production_method。'''
        response = self.client.invoke(prompt).content
        text = response if isinstance(response, str) else str(response)
        if "```" in text:
            text = text.split("```", 2)[1].removeprefix("json").strip()
        data = json.loads(text)
        if not set(data.get("material_ids", [])) <= set(material_ids):
            raise ValueError("LLM 脚本引用了未检索素材")
        return data


class MockPlanner:
    """Explicit offline fixture; it exercises the same LangGraph/ToolNode path, not a live model."""
    def __init__(self):
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools

    def generate_script(self, topic: str, material_ids: list[str], requirements: str) -> dict:
        return {"title": f"{topic}：一分钟亮点", "creative_brief": "先展示卖点，再给出行动号召。",
                "shots": [{"visual": material_ids[0], "duration_seconds": 8}, {"visual": material_ids[0], "duration_seconds": 12}],
                "voiceover": f"探索{topic}，现在就行动。", "material_ids": material_ids,
                "production_method": requirements}

    def decide(self, messages):
        results = _results(messages)
        last = results[-1] if results else {}
        calls = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
        ident = f"mock-{calls + 1}"
        if not results or (last.get("tool") == "search_materials" and not last.get("ok")):
            return _tool("search_materials", {"query": "短视频营销素材"}, ident)
        if last.get("tool") == "search_materials" and last.get("ok"):
            return _tool("generate_script", {"topic": _state_topic(messages), "material_ids": ["V001", "M001"], "requirements": "S001+S002"}, ident)
        if last.get("tool") == "generate_script" and last.get("ok"):
            services = _services_from_script(last["script"])
            return _tool("estimate_cost", {"material_ids": last["script"]["material_ids"], "service_ids": services}, ident)
        if last.get("tool") == "estimate_cost":
            if last.get("within_budget") and last.get("within_deadline"):
                script = _latest_script(results)
                return _tool("save_plan", {"plan": {"script": script, "material_ids": script["material_ids"], "service_ids": _services_from_script(script), "estimate": last, "plan_version": 1}}, ident)
            # A deliberately visible adjustment: low-cost material then regenerate and re-estimate.
            if last.get("estimated_cost", 0) > 0 and not any(r.get("adjusted") for r in results):
                return _tool("generate_script", {"topic": _state_topic(messages), "material_ids": ["V002", "M001"], "requirements": "S001+S002；adjusted"}, ident)
            return AIMessage(content="没有满足预算和时间的可行方案。")
        if last.get("tool") == "save_plan" and last.get("ok"):
            return AIMessage(content="方案已保存。")
        return AIMessage(content="工具返回无法继续处理。")


def _state_topic(messages) -> str:
    for m in messages:
        if getattr(m, "type", "") == "system" and "主题=" in str(m.content):
            return str(m.content).split("主题=", 1)[1].split("；", 1)[0]
    return "营销主题"


def _services_from_script(script: dict) -> list[str]:
    return [x for x in ["S001" if "S001" in script.get("production_method", "") else None,
                         "S002" if "S002" in script.get("production_method", "") else None,
                         "S003" if "S003" in script.get("production_method", "") else None,
                         "S004" if "S004" in script.get("production_method", "") else None] if x]


def _latest_script(results: list[dict]) -> dict:
    return next(r["script"] for r in reversed(results) if r.get("tool") == "generate_script" and r.get("ok"))
