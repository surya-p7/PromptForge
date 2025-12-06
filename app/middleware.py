from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from .logger import request_logger
import time

class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or None
        log = request_logger(rid, path=str(request.url))
        start = time.time()
        log(20, "request_start", method=request.method)
        response = await call_next(request)
        duration = (time.time()-start)*1000
        log(20, "request_end", status_code=response.status_code, duration_ms=duration)
        response.headers["x-request-duration-ms"] = str(int(duration))
        return response
