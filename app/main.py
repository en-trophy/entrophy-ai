from fastapi import FastAPI
from app.api.lesson_feedback import router as lessons_router

app = FastAPI()
app.include_router(lessons_router, prefix="/api/lessons")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
