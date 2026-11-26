"""Serve a scripted DeepSeek Messages endpoint for the expense calculation sample."""

import json
import re
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Iterator

READ_CALL_ID = "read-expenses"


class ExpenseModel(BaseHTTPRequestHandler):
    """Return a read request, then calculate the total from the Agent's tool result."""

    def do_POST(self) -> None:
        if self.path != "/v1/messages":
            self.send_error(404)
            return

        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tool_result = next(
            (
                block
                for message in request["messages"]
                for block in message["content"]
                if block["type"] == "tool_result" and block["tool_use_id"] == READ_CALL_ID
            ),
            None,
        )
        if tool_result is None:
            block = {"type": "tool_use", "id": READ_CALL_ID, "name": "read", "input": {}}
            delta = {"type": "input_json_delta", "partial_json": '{"file_path":"expenses.txt"}'}
            stop_reason = "tool_use"
        else:
            # The real read tool renders each source line as `line_number: text`.
            text = "\n".join(block["text"] for block in tool_result["content"] if block["type"] == "text")
            amounts = re.findall(r"^\d+:[ \t]*(-?\d+)[ \t]*$", text, re.MULTILINE)
            if tool_result.get("is_error") or not amounts:
                self.send_error(400, "Expected one integer per line in expenses.txt")
                return
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": f"Total: {sum(map(int, amounts))}"}
            stop_reason = "end_turn"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for kind, payload in [
            ("message_start", {"message": {"id": "expense-sample", "model": request["model"],
                                           "usage": {"input_tokens": 5, "output_tokens": 0}}}),
            ("content_block_start", {"index": 0, "content_block": block}),
            ("content_block_delta", {"index": 0, "delta": delta}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 3}}),
            ("message_stop", {}),
        ]:
            event = json.dumps({"type": kind, **payload})
            self.wfile.write(f"event: {kind}\ndata: {event}\n\n".encode())

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@contextmanager
def local_model() -> Iterator[str]:
    """Keep the model endpoint available on a free loopback port for one SDK run."""
    with ThreadingHTTPServer(("127.0.0.1", 0), ExpenseModel) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()
