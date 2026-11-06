# Evaluation

本次审查后实际执行命令为 `PYTHONPATH=src pytest -q`，结果：**15 passed**（以当前机器实际输出为准）。执行时间为本次实现和验证的实际会话时间；未虚构 3 小时记录。

| 验收项 | 测试 | 实际结果 |
| --- | --- | --- |
| T1 正常保存 | `test_t1_normal_budget_and_deadline_saves_plan` | 完成，成本 12 |
| T2 缺字段 | `test_t2_missing_required_field_is_explicit` | incomplete |
| T3 故障 | `test_t3_injected_failure_is_observed_and_retried` | 首次失败后重试 |
| T4 预算调整 | `test_t4_over_budget_is_adjusted_then_reestimated` | 12 → 8 后保存 |
| T5 幂等 | `test_t5_same_task_has_one_persisted_plan` | 仅一条 SQLite 记录 |
| T6 续跑 | `test_t6_checkpoint_resume_preserves_search` | 工具失败后续跑，复用 search 结果 |
| T7 无解 | `test_t7_no_feasible_candidate_is_incomplete` | incomplete |
| T8 上限 | `test_t8_turn_limit_stops_safely` | 安全停止 |
| T9 脚本变更未估算 | `test_t9_script_without_a_fresh_estimate_cannot_be_saved` | 保存被拒绝，零持久化副作用 |
| T10 无效工具 | `test_t10_unknown_tool_result_does_not_pollute_step_state` | incomplete，步骤状态未污染 |
| Trace | `test_trace_records_actual_tool_arguments_and_thread` | 记录 thread、参数和决策事件 |
| LLM 异常 | `test_llm_exception_is_traced_without_state_corruption` | 记录异常且不污染步骤状态 |

实际 Mock 运行记录：[normal](../examples/traces/normal.jsonl)、[missing](../examples/traces/missing-budget.jsonl)、[failure](../examples/traces/tool-failure.jsonl)。这些记录明确是 Mock，不是 live 结果。

已用调用方提供的 GLM 密钥（仅进程环境变量，未写入文件）验证 `https://open.bigmodel.cn/api/paas/v4/` + `glm-4-flash`。首次返回 400，定位并修复为初始请求缺少 user message；后续 live 调用成功由模型选择并执行了 `search_materials`、`generate_script`、`estimate_cost`。该次模型生成的方案选了多个主视觉，估算工具按规则拒绝；命令运行时间到达执行环境的 30 秒限制时仍在等待模型下一次决策，故未产生完整 live 成功方案或伪造 live Trace。

审查阶段再次执行 live 验证。`glm-4-flash` 先提交了包含服务 ID 的非法素材参数，ToolNode 返回校验失败；模型下一轮修正为合法素材，成功生成脚本并估算成本，再因自行重建了不一致的嵌套保存参数而被保存工具拒绝。相应真实失败记录保留在 [live-glm4-flash-partial.jsonl](../examples/traces/live-glm4-flash-partial.jsonl)，不标记为通过。使用同一 `thread_id` 续跑时，已成功的检索、脚本和估算结果被复用；模型随后触发 `max_turns`，未产生重复保存。`glm-4.7` 和 `glm-5.3-flash` 的实际请求均返回 429（账户资源/额度不可用）。因此，真实 LLM 集成测试已执行但**未通过完整的重规划并保存验收**；Mock 测试不替代这一结论。

后续充值后重新尝试 `glm-5.3-flash` 与 `glm-5-turbo`：两者真实完成了检索并选择 `generate_script`，但前台 30 秒执行窗口在脚本生成请求时到期；将单次输出限制为 300 token 后，服务返回 429 并提示账户请求频率限制。代码现支持 `LLM_MAX_TOKENS`，并在 LLM 调用异常时写入 Trace。该轮仍未形成完整 live 成功路径，不能据此宣称真实模型验收通过。

限制：实时模型的选择有非确定性，生产环境应增加允许的模型列表、请求超时/退避、审计脱敏和人工审批。
