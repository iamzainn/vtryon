from fastapi import FastAPI
from api import endpoints

app = FastAPI(title="Virtual Try-On API")

# Include API routes
app.include_router(endpoints.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)