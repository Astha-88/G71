import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import access, auth, datasets, experiments, institutions, projects, hospitals, notifications
from db.database import init_db

logger = logging.getLogger(__name__)

# Wildcard origins ("*") cannot be combined with allow_credentials=True per the
# CORS spec — browsers reject the response outright. Use an explicit,
# env-driven allowlist instead. Set CORS_ORIGINS="https://app.example.com,https://admin.example.com"
# in production.
DEFAULT_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", ",".join(DEFAULT_DEV_ORIGINS)).split(",")
    if origin.strip()
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vienko Decentralized Healthcare FL Platform",
        version="1.0.0",
        description="Backend + blockchain orchestration for federated learning in healthcare",
    )

    @app.middleware("http")
    async def catch_unhandled(request: Request, call_next):
        """Turns any unhandled route exception into a normal 500 JSON response
        that then passes OUTWARD through CORSMiddleware and gets CORS headers.
        Registered BEFORE CORSMiddleware on purpose: the first middleware added
        is the innermost, so CORS wraps this one. The full exception is logged
        server-side (docker-compose logs backend); the client only ever sees a
        generic, safe message -- no traceback, SQL, hostname, or path.
        """
        try:
            return await call_next(request)
        except Exception:
            logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "An unexpected error occurred. Please try again."},
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Fallback only -- see the catch_unhandled middleware above for the real fix.

        Exceptions handled HERE are produced by Starlette's ServerErrorMiddleware,
        which sits OUTSIDE CORSMiddleware, so this response carries NO CORS headers:
        the browser hides it from JavaScript and fetch() rejects, which the UI
        reports as a "network error" even though the server returned a 500. That
        masking made several real backend failures look like connectivity problems.
        The catch_unhandled middleware (installed INSIDE CORS) catches route
        exceptions first so they return a normal, CORS-headed 500 with a safe message.
        """
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    # Routers
    app.include_router(institutions.router, prefix="/institutions", tags=["institutions"])
    app.include_router(auth.router, prefix="/users", tags=["users"])
    app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
    app.include_router(access.router, prefix="/dataset", tags=["dataset-access"])
    app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
    app.include_router(projects.router, prefix="/research-projects", tags=["research-projects"])
    app.include_router(hospitals.router, prefix="/hospitals", tags=["hospitals"])
    app.include_router(notifications.router, prefix="/notifications", tags=["notifications"])

    @app.on_event("startup")
    async def on_startup() -> None:
        init_db()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

