"""
ProofPilot — FastAPI Application Entry Point
----------------------------------------------
Production microservice for AI Dispute Risk Management & Webhook Ingestion.
Built for Razorpay AI Buildathon 2026 (Track 02: AI Risk Manager).
Includes Request ID tracing, security headers, rate limiting, and ML drift endpoints.
"""

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.dependencies import get_active_predictor, get_all_reason_codes
from api.routes.disputes import router as disputes_router
from api.routes.drift import router as drift_router
from api.routes.health import router as health_router
from api.routes.webhook import router as webhook_router
from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize and warm up ML models and configs
    logger.info("Starting ProofPilot AI Risk Service warm-up...")
    get_all_reason_codes()
    predictor = get_active_predictor()
    logger.info(f"Model warm-up complete. Active model: {predictor.auc_score:.1%} ROC-AUC.")
    yield
    logger.info("Shutting down ProofPilot AI Risk Service...")


app = FastAPI(
    title="ProofPilot API — AI Dispute Risk Manager",
    description=(
        "Production-grade Pre-Submission Chargeback Readiness, Calibrated ML Win Prediction, "
        "Economic ROI Decisioning, and Razorpay HMAC-SHA256 Webhook Firewall."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


class SecurityAndTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate or pass-through Request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)

        # Inject Security and Tracing Headers
        response.headers["X-Request-ID"] = request_id
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


# Middleware registration
app.add_middleware(SecurityAndTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router)
app.include_router(webhook_router)
app.include_router(disputes_router)
app.include_router(drift_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
