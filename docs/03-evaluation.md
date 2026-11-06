# Evaluation

本次实际执行命令为 `PYTHONPATH=src pytest -q`，结果：**10 passed**（7.29s）。执行时间为本次实现和验证的实际会话时间；未虚构 3 小时记录。

| 验收项 | 测试 | 实际结果 |
| --- | --- | --- |
| T1 正常保存 | `test_t1_normal_budget_and_deadline_saves_plan` | 完成，成本 12 |
| T2 缺字段 | `test_t2_missing_required_field_is_explicit` | incomplete |
| T3 故障 | `test_t3_injected_failure_is_observed_and_retried` | 首次失败后重试 |
| T4 预算调整 | `test_t4_over_budget_is_adjusted_then_reestimated` | 12 → 8 后保存 |
| T5 幂等 | `test_t5_same_task_has_one_persisted_plan` | 仅一条 SQLite 记录 |
| T6 续跑 | `test_t6_checkpoint_resume_preserves_search` | 保留 search 结果 |
| T7 无解 | `test_t7_no_feasible_candidate_is_incomplete` | incomplete |
| T8 上限 | `test_t8_turn_limit_stops_safely` | 安全停止 |

实际 Mock 运行记录：[normal](../examples/traces/normal.jsonl)、[missing](../examples/traces/missing-budget.jsonl)、[failure](../examples/traces/tool-failure.jsonl)。这些记录明确是 Mock，不是 live 结果。

已用调用方提供的 GLM 密钥（仅进程环境变量，未写入文件）验证 `https://open.bigmodel.cn/api/paas/v4/` + `glm-4-flash`。首次返回 400，定位并修复为初始请求缺少 user message；后续 live 调用成功由模型选择并执行了 `search_materials`、`generate_script`、`estimate_cost`。该次模型生成的方案选了多个主视觉，估算工具按规则拒绝；命令运行时间到达执行环境的 30 秒限制时仍在等待模型下一次决策，故未产生完整 live 成功方案或伪造 live Trace。

限制：实时模型的选择有非确定性，生产环境应增加允许的模型列表、请求超时/退避、审计脱敏和人工审批。
