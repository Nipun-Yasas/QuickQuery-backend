from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

# 1. Create the FastAPI instance (Uvicorn looks for this 'app' variable)
app = FastAPI(title="QuickQuery RAG Backend")

# 2. Add CORS Middleware (Crucial for when you connect your Next.js frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Include the routes we built in the previous step
# This prefix means all your routes will start with /api (e.g., /api/upload)
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "RAG Backend is running!"}