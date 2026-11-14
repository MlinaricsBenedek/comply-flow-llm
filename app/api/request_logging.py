import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)


class RequestFileLoggingMiddleware:
    def __init__(self, app: Any, log_file: str | Path) -> None:
        self.app = app
        self.log_file = Path(log_file)
        self._write_lock = Lock()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body_chunks: list[bytes] = []
        response_status: int | None = None
        correlation_id: str | None = None
        cancelled = False

        async def receive_and_capture() -> dict[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                body_chunks.append(message.get("body", b""))
            return message

        async def send_and_capture(message: dict[str, Any]) -> None:
            nonlocal response_status, correlation_id
            if message["type"] == "http.response.start":
                response_status = message["status"]
                for name, value in message.get("headers", []):
                    if name.lower() == b"x-correlation-id":
                        correlation_id = value.decode("latin-1")
                        break
            await send(message)

        try:
            await self.app(scope, receive_and_capture, send_and_capture)
        except asyncio.CancelledError:
            cancelled = True
            request_id = correlation_id or scope.get("state", {}).get("correlation_id")
            logger.warning(
                "api_request_cancelled correlation_id=%s method=%s path=%s",
                request_id or "none",
                scope["method"],
                scope["path"],
                extra={
                    "correlation_id": request_id,
                    "operation": f'{scope["method"]} {scope["path"]}',
                },
            )
            raise
        except Exception as middleware_error:
            logger.error(f"Middleware error in app call: {middleware_error}", exc_info=True)
            raise
        finally:
            body = b"".join(body_chunks)
            try:
                body_value: Any = json.loads(body) if body else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                body_value = body.decode("utf-8", errors="replace")

            content_type = next(
                (
                    value.decode("latin-1")
                    for name, value in scope.get("headers", [])
                    if name.lower() == b"content-type"
                ),
                None,
            )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": scope["method"],
                "path": scope["path"],
                "query_string": scope.get("query_string", b"").decode(
                    "utf-8", errors="replace"
                ),
                "content_type": content_type,
                "request_body": body_value,
                "status_code": response_status,
                "correlation_id": (
                    correlation_id or scope.get("state", {}).get("correlation_id")
                ),
                "cancelled": cancelled,
            }
            try:
                self._append_record(record)
            except OSError:
                logger.exception(
                    "request_file_log_write_failed",
                    extra={"log_file": str(self.log_file)},
                )
            except Exception as logging_error:
                logger.error(f"Error in request logging: {logging_error}", exc_info=True)

    def _append_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
