# Solution

`init_node` 校验输入并初始化状态。`agent_node` 将当前消息（包含工具观察）交给绑定工具的 LLM；模型无工具调用时结束。`ToolNode` 实际执行四个 LangChain 工具。`observe_node` 解码结果、更新步骤/轮次/估算、使旧估算失效、写 Trace，随后路由回 Agent 或 Final。Trace 分为 `decision` 与 `tool_result` 事件，分别记录模型选择和实际结果；不记录思维链。

保存之前 `ToolNode` 注入 State，要求保存请求与当前脚本、素材、服务、估算和 `plan_version` 完全一致；`PlanRepository` 再计算成本和工时，并验证脚本字段、脚本素材一致性、组合规则、预算/截止、估算值以及哈希幂等键。因此 LLM 输出不是权限边界。

恢复时，graph 从同一 `task_id` 的 checkpoint 读取已观察结果；成功工具不会被重新执行。已保存任务直接返回既有完成状态。
