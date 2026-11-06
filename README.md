# Content Planning Agent

面向短视频运营的内容活动规划 Agent。它接收主题、预算、截止时间，检索素材、生成脚本、以程序计算成本/工时，并只在硬校验通过后保存方案。

## 架构

`init → agent(LLM) → ToolNode → observe → agent/final` 是 LangGraph `StateGraph` 循环。在线模式由 OpenAI-compatible LLM 的 tool calling 选择下一步；工具输出再作为观察反馈给模型。预算、截止时间、组合规则、估算和保存均由程序强制执行，模型无权改写。

状态通过 SQLite Checkpointer 按 `task_id`（thread_id）保存；方案表以 `task_id + SHA-256(plan)` 幂等。JSONL Trace 只保存可观察的工具/决策摘要，不保存思维链。

## 安装与运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m content_planning.cli examples/inputs/normal.json --mode mock
```

Mock 是离线、脚本化的测试替身，走同一 StateGraph 和 ToolNode，不能代表真实 LLM 能力。故障演示：

```powershell
python -m content_planning.cli examples/inputs/tool-failure.json --mode mock --inject-failure search_materials:1
```

真实 LLM：复制 `.env.example` 为 `.env`，填入 API 配置后运行：

```powershell
python -m content_planning.cli examples/inputs/normal.json --mode live --runtime-dir runtime
```

可选设置 `LLM_MAX_TOKENS=600` 限制单次模型输出，避免脚本生成过长拖慢循环。

续跑（保持相同 `task_id`，可提高限制）：

```powershell
python -m content_planning.cli examples/inputs/normal.json --mode mock --runtime-dir runtime --resume
```

## 测试与评审入口

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

测试覆盖 T1–T10，见 `tests/test_workflow_mock.py`；单元硬校验见 `tests/test_tools.py`。离线示例位于 `examples/outputs/` 与 `examples/traces/`。已进行一次 GLM live 连通和工具调用验证，实际限制与结果见 `docs/03-evaluation.md`；示例 Trace 仍明确标为 Mock，不冒充 live 记录。
