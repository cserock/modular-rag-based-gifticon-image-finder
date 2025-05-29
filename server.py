import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel, Field
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_teddynote import logging
from langchain.schema import Document
from langserve import add_routes
from dotenv import load_dotenv
from pprint import pprint
from packages.graphs.graph import app
from packages.graphs.loader import load_from_image
from common import CACHE_PATH, LLM_CACHE_PATH, PROJECT, UPLOAD_FILE_PATH

load_dotenv(verbose=True, override=True)

# 캐시 디렉토리를 생성합니다.
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# SQLiteCache를 사용합니다.
set_llm_cache(SQLiteCache(database_path=LLM_CACHE_PATH))

# langsmith 추적 시작
logging.langsmith(PROJECT)

server = FastAPI(
    title="ARPO-E API Server",
    version="1.0",
    description="ARPO-E 프로젝트의 API 서버 인터페이스입니다.",
)

# Set all CORS enabled origins
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# @server.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/chat/playground")


@server.post("/api/upload/", tags=["api/upload"])
async def upload_files(files: List[UploadFile] = File(...)):
    file_details = []
    coupon_infos = []

    for file in files:
        file_details.append({"filename": file.filename, "content_type": file.content_type})
        with open(f"{UPLOAD_FILE_PATH}/{file.filename}", "wb") as f:
            f.write(await file.read())
        coupon_info = load_from_image(f"{UPLOAD_FILE_PATH}/{file.filename}")
        coupon_infos.append(coupon_info)

    return JSONResponse({"coupon_infos": coupon_infos})

class InputSearch(BaseModel):
    """Input for the search endpoint."""
    question: str = Field(description="question")

class OutputSearch(BaseModel):
    """Input for the search endpoint."""
    question: str = Field(description="question")
    generation: dict = Field(description="generation")
    web_search: bool = Field(description="web_search")
    documents: List[Document] = Field(description="list of Document")
    
add_routes(
    server, 
    app.with_types(input_type=InputSearch, output_type=OutputSearch), 
    config_keys=["configurable"],
    path="/api/search",
    disabled_endpoints=["batch", "config_hashes"]
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(server, host="0.0.0.0", port=8000)