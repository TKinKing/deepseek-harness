"""运行真实的 Agent 循环，计算 sample/expenses.txt 中每行金额的总和。

安装仓库依赖后，在根目录运行：
    pnpm run build:native-system
    uv run --locked --project python/sdk python -B sample_test.py

按执行顺序读源码：
1. packages/sdk/server/src/server.ts 的 prompt() 把 SDK 输入交给 agent.followup()。
2. packages/core/agent-loop/src/agent.ts 的 send() 调用收件箱入队，kick() 启动 turn()。
3. turn() 调用 preStep() 领取输入并组装系统提示词和工具定义，再调用 step()。
   packages/core/system-prompt/src/index.ts 的 assemble() 负责提示词组装。
4. step() 调用模型适配器并读取响应流，随后通过 ctx.tools.execute() 分发 read。
   packages/fs/tool-fs/src/read.ts 的 applyReadTool() 注册实际的文件读取工具。
   packages/core/tools/src/batch.ts 把工具结果写入 Session。
5. packages/core/session/src/index.ts 的 deriveMessages() 把工具结果带进第二次请求。
   step() 收到文本回复后返回 completed，turn() 写入 turn/end。

sample/model.py 提供本地的固定模型策略。修改费用文件里的数字即可看到新结果。
"""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from deepseek_harness import DeepSeekHarness

from sample.model import local_model

ROOT = Path(__file__).resolve().parent
PROMPT = "Read expenses.txt and add up the amounts, one amount per line."


def main(workspace: Path = ROOT / "sample") -> None:
    """Submit one prompt and check the real Session events for the tool round trip."""
    with TemporaryDirectory(prefix="dsh-agent-sample-") as home, local_model() as base_url:
        # The SDK's test launcher runs the same source entrypoint as `pnpm dsh`.
        with closing(DeepSeekHarness(
            cwd=str(workspace),
            runtime_cwd=str(ROOT),
            _launch_args=("node", "--import", "tsx/esm", str(ROOT / "apps/cli/src/bin.ts"), "--profile", "sdk"),
            base_url=base_url,
            api_key="sample-local",
            env={"DSH_HOME": home, "DSH_PERMISSION_MODE": "read-only", "DSH_TELEMETRY_DISABLED": "1"},
            request_timeout_seconds=30,
        )) as harness:
            result = harness.run(PROMPT)

    expected = [
        "turn/start",
        "step/start", "assistant/message", "tool/call", "tool/result", "step/end",
        "step/start", "assistant/message", "step/end",
        "turn/end",
    ]
    stages = [event["type"] for event in result.events if event["type"] in expected]
    if stages != expected or result.finish_reason != "completed":
        raise RuntimeError(f"Unexpected Agent run ({result.finish_reason}): {stages}")

    print(PROMPT)
    for event in result.events:
        kind, data = event["type"], event["data"]
        if kind in ("step/start", "step/end"):
            print(f"  {kind} (step {data['step']})")
        elif kind == "tool/call":
            print(f"  {kind} {data['name']} {data['arguments']}")
        elif kind == "tool/result":
            lines = [line["text"] for line in data["meta"]["lines"]]
            print(f"  {kind} {', '.join(lines)}")
        elif kind == "user/message" and data["source"]["kind"] == "user":
            print(f"  {kind} (prompt)")
        elif kind in (*expected, "agent/inbox/spliced", "system/message"):
            print(f"  {kind}")
    print(result.final_response)


if __name__ == "__main__":
    main()
