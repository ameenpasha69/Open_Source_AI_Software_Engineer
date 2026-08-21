import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logging import request_id_var

logger = logging.getLogger("app.request")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a request_id (reusing an inbound X-Request-ID if the caller
    already has one, e.g. from a proxy) so every log line emitted while
    handling this request — including ones logged deep inside the agent
    loop for a synchronous call — can be correlated back to it. Also logs
    one line per request: method, path, status, duration.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        token = request_id_var.set(request_id)
        started = time.monotonic()
        try:
            try:
                response = await call_next(request)
            except Exception:
                duration = round(time.monotonic() - started, 3)
                logger.exception(
                    "request failed",
                    extra={"method": request.method, "path": request.url.path, "duration_seconds": duration},
                )
                raise

            duration = round(time.monotonic() - started, 3)
            logger.info(
                "request handled",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                },
            )
            response.headers["x-request-id"] = request_id
            return response
        finally:
            request_id_var.reset(token)
