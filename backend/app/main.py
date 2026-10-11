from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.posts import router as posts_router
from app.db.supabase import get_supabase_client

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Blog Notebook with Supabase integration",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for Frontend (Vite localhost & GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(posts_router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def startup_event():
    client = get_supabase_client()
    if client:
        print("🚀 Supabase is connected.")
    else:
        print("💡 Running in Mock In-Memory Mode (Set SUPABASE_URL & SUPABASE_KEY in .env when ready)")


@app.get("/")
def root():
    return {
        "message": "Blog Notebook API is running",
        "docs": "/docs",
        "supabase_connected": get_supabase_client() is not None,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "supabase_connected": get_supabase_client() is not None,
    }

