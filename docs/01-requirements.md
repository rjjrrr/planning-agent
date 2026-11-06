# Requirements mapping

- LangGraph: `src/content_planning/graph.py` 使用 `StateGraph`、`ToolNode`、条件路由。
- LLM: `OpenAIPlanner` 绑定工具并做真实 tool calling；`generate_script` 单独调用真实 LLM 并校验 JSON。
- 规则: `data/` 为素材、服务、组合规则；`data.py` 作串行工时和成本计算。
- 可靠性: SQLite checkpoint、SQLite 幂等保存、JSONL trace、上限终止、失败观察与续跑。
- 验收: Mock 分层测试 T1–T8；live 测试需要调用方配置真实密钥。
