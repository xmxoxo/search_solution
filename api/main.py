import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.app_config import API_HOST, API_PORT
from api.utils.logger import g_logger
from api.routers import match, data, parse

@asynccontextmanager
async def lifespan(app: FastAPI):
    g_logger.info("Starting Intelligent Matching Engine API...")
    yield
    g_logger.info("Shutting down Intelligent Matching Engine API...")

app = FastAPI(
    title="Intelligent Matching Engine API",
    description="智能匹配引擎API - 支持语义匹配、多资源检索",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    g_logger.info(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    g_logger.error(f"[{request_id}] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": str(exc),
            "request_id": request_id
        }
    )

app.include_router(match.router, prefix="/api/v1", tags=["match"])
app.include_router(data.router, prefix="/api/v1", tags=["data"])
app.include_router(parse.router, prefix="/api/v1", tags=["parse"])

@app.get("/")
async def root():
    return {
        "name": "Intelligent Matching Engine API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
