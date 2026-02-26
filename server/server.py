from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from langserve import add_routes
from lang_helper import chain

app = FastAPI(description="RAG Hybrid chat Bot",  version="1.0")
add_routes(app, chain, path="/rag")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
