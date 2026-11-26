"""Serve budget decisions from the Session history sent to a local model endpoint."""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Iterator

CHECK_PROMPT = "Check the budget."
SUMMARY_PROMPT = "Show the budget."
ADJUSTMENT = re.compile(r"Add ([+-]?[0-9]+)\.", re.IGNORECASE)
PAGE_SIZE = 2
READ_FOOTER = re.compile(r"^\((?:End of file - total \d+ lines|.*Use offset=(\d+) to continue\.)\)\r?$", re.MULTILINE)


@dataclass(frozen=True)
class ReadFile:
    """A single page of a file requested during a budget review."""

    name: str
    review: int
    offset: int = 1

    @property
    def call_id(self) -> str:
        return f"review-{self.review}-{self.name}-{self.offset}"


def read_page(call: ReadFile, result: dict[str, Any]) -> tuple[list[int], int | None]:
    """Parse the numbered integer rows and continuation from the native read tool."""
    text = "\n".join(block["text"] for block in result["content"] if block["type"] == "text")
    if result.get("is_error"):
        raise ValueError(f"{call.name}: {text}")
    footer = READ_FOOTER.search(text)
    if footer is None:
        raise ValueError(f"{call.name}: missing read continuation or end-of-file marker")
    next_offset = int(footer[1]) if footer[1] else None
    if next_offset is not None and next_offset <= call.offset:
        raise ValueError(f"{call.name}: read did not advance past line {call.offset}")
    rows = re.findall(r"^\d+: ([^\n]*)$", text, re.MULTILINE)
    try:
        amounts = [int(row) for row in rows]
    except ValueError as error:
        raise ValueError(f"{call.name}: expected one integer amount per line") from error
    return amounts, next_offset


def budget_response(messages: list[dict[str, Any]]) -> ReadFile | str:
    """Use the latest review's file results and all adjustments in the request history."""
    review = 0
    adjustments = 0
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message["role"] != "user":
            continue
        for block in message["content"]:
            if block["type"] == "tool_result":
                results[block["tool_use_id"]] = block
            elif block["type"] == "text" and block["text"] == CHECK_PROMPT:
                review += 1
                results.clear()
            elif block["type"] == "text" and (match := ADJUSTMENT.fullmatch(block["text"])):
                adjustments += int(match[1])

    recorded = 0
    offset = 1
    while offset is not None:
        call = ReadFile("expenses.txt", review, offset)
        if call.call_id not in results:
            return call
        amounts, offset = read_page(call, results[call.call_id])
        recorded += sum(amounts)

    call = ReadFile("budget.txt", review)
    if call.call_id not in results:
        return call
    budgets, continuation = read_page(call, results[call.call_id])
    if len(budgets) != 1 or continuation is not None:
        raise ValueError("budget.txt: expected a single integer budget")
    budget = budgets[0]
    forecast = recorded + adjustments
    remaining = budget - forecast
    status = f"Remaining: {remaining}" if remaining >= 0 else f"Over budget by: {-remaining}"
    return f"Recorded: {recorded} | Adjustments: {adjustments:+} | Forecast: {forecast} | Budget: {budget}\n{status}"


class ExpenseModel(BaseHTTPRequestHandler):
    """Stream a read call or a budget summary through the DeepSeek Messages protocol."""

    def do_POST(self) -> None:
        if self.path != "/v1/messages":
            self.send_error(404)
            return

        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            response = budget_response(request["messages"])
        except ValueError as error:
            response = f"Budget review could not finish. {error}"

        if isinstance(response, ReadFile):
            block = {"type": "tool_use", "id": response.call_id, "name": "read", "input": {}}
            arguments = {"file_path": response.name, "offset": response.offset, "limit": PAGE_SIZE}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(arguments)}
            stop_reason = "tool_use"
        else:
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": response}
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
    """Keep the model endpoint available on a free loopback port for the demo Session."""
    with ThreadingHTTPServer(("127.0.0.1", 0), ExpenseModel) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()
