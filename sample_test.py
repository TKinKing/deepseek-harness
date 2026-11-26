"""用真实的 Agent 循环进行多轮预算预演。

安装仓库依赖后，在根目录运行：
    pnpm run build:native-system
    uv run --locked --project python/sdk python -B sample_test.py

脚本在同一个 Session 中依次执行 DEMO_PROMPTS 中的全部六轮。
`Check the budget.` 重新读取两个文件，并保留本次会话中提出的调整。
expenses.txt 每行一个整数金额，budget.txt 提供一个整数预算。追加和扣除用于预演。

按执行顺序读源码：
1. packages/sdk/server/src/server.ts 的 prompt() 把 SDK 输入交给 agent.followup()。
2. packages/core/agent-loop/src/agent.ts 的 send() 调用收件箱入队，kick() 启动 turn()。
3. turn() 调用 preStep() 领取输入并组装系统提示词和工具定义，再调用 step()。
   packages/core/system-prompt/src/index.ts 的 assemble() 负责提示词组装。
4. step() 调用模型适配器并读取响应流，随后通过 ctx.tools.execute() 分发 read。
   packages/fs/tool-fs/src/read.ts 的 applyReadTool() 注册实际的文件读取工具。
   packages/core/tools/src/batch.ts 把工具结果写入 Session。
5. packages/core/session/src/index.ts 的 deriveMessages() 把工具结果带进下一次请求。
   只要模型继续请求工具，turn() 就会进入下一步。模型给出汇总后，turn() 写入 turn/end。
6. 每次输入调用同一个 Session 的 run()。下一轮请求仍包含之前的用户消息和工具结果。

sample/model.py 提供本地模型策略，根据每次收到的对话历史决定继续读取还是给出汇总。
"""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from python.sdk.src.deepseek_harness.api import DeepSeekHarness, Session
from sample.model import CHECK_PROMPT, SUMMARY_PROMPT, local_model

ROOT = Path(__file__).resolve().parent
DEMO_PROMPTS = (CHECK_PROMPT, "Add 75.", "Add 60.", "Add -50.", SUMMARY_PROMPT, CHECK_PROMPT)


def run_turn(session: Session, prompt: str) -> None:
    """Run one prompt and print the recorded lifecycle of that turn."""
    print(f"\n> {prompt}", flush=True)
    result = session.run(prompt)
    for event in result.events:
        kind, data = event["type"], event["data"]
        if kind == "turn/start":
            print(f"  {kind} (turn {data['turn']})")
        elif kind == "turn/end":
            print(f"  {kind} ({data['reason']['kind']})")
        elif kind in ("step/start", "step/end", "assistant/message"):
            print(f"  {kind} (step {data['step']})")
        elif kind == "tool/call":
            print(f"  {kind} {data['name']} {data['arguments']}")
        elif kind == "tool/result":
            lines = [line["text"] for line in data.get("meta", {}).get("lines", [])]
            detail = "read error" if data["message"].get("isError") else ", ".join(lines) or "empty file"
            print(f"  {kind} {detail}")
        elif kind == "user/message" and data["source"]["kind"] == "user":
            print(f"  {kind} (prompt)")
        elif kind in ("agent/inbox/spliced", "system/message"):
            print(f"  {kind}")
    if result.finish_reason != "completed":
        raise RuntimeError(f"Agent ended with {result.finish_reason}. {result.final_response}")
    print(result.final_response)


def main(workspace: Path = ROOT / "sample") -> None:
    """Run every demo prompt in one Session."""
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
            session = harness.start_session()
            print(f"Session {session.id}")
            for prompt in DEMO_PROMPTS:
                run_turn(session, prompt)


if __name__ == "__main__":
    main()
