from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uvicorn
import os

from recommendation_engine import RecommendationEngine
from config import settings


# ---------------------------------------------
# Server configuration
# ---------------------------------------------
port = int(os.environ.get("PORT", 8000))

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered assessment recommendation system for hiring",
    version="1.0.0"
)

# ---------------------------------------------
# CORS Configuration (Frontend Integration)
# ---------------------------------------------
# ✅ Allow your deployed Vercel frontend and local testing
origins = [
    "https://shl-frontend-mu.vercel.app",  # Your production frontend
    "http://localhost:3000",               # Local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------
# Initialize Recommendation Engine
# ---------------------------------------------
recommendation_engine = RecommendationEngine()


# ---------------------------------------------
# Request/Response Models
# ---------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="Job description or natural language query")
    max_results: Optional[int] = Field(
        settings.MAX_RECOMMENDATIONS,
        ge=settings.MIN_RECOMMENDATIONS,
        le=settings.MAX_RECOMMENDATIONS,
        description="Maximum number of recommendations to return"
    )


class AssessmentRecommendation(BaseModel):
    assessment_name: str
    assessment_url: str
    relevance_score: float
    description: Optional[str] = None
    test_type: Optional[str] = None


class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[AssessmentRecommendation]
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_loaded: bool


# ---------------------------------------------
# Endpoints
# ---------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Simple health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=app.version,
        model_loaded=recommendation_engine.is_loaded()
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: QueryRequest):
    """Return recommended SHL assessments for a given query"""
    start_time = datetime.now()

    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Please provide a meaningful query (min 5 characters).")

    try:
        recommendations = recommendation_engine.recommend(
            query=request.query.strip(),
            top_k=request.max_results
        )

        formatted_recommendations = [
            AssessmentRecommendation(
                assessment_name=rec["assessment_name"],
                assessment_url=rec["assessment_url"],
                relevance_score=rec.get("relevance_score", 0.0),
                description=rec.get("description"),
                test_type=rec.get("test_type")
            )
            for rec in recommendations
        ]

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return RecommendationResponse(
            query=request.query,
            recommendations=formatted_recommendations,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint showing available routes"""
    return {
        "message": "SHL Assessment Recommendation API is running",
        "version": app.version,
        "frontend": "https://shl-frontend-mu.vercel.app",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }


# ---------------------------------------------
# Application Entry Point
# ---------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.API_HOST, port=port)
