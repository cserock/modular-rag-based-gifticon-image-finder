from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from langchain_community.chat_models import ChatOllama
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import rootdir

load_dotenv(verbose=True, override=True)

# 프로젝트명 설정
PROJECT = 'arpo-app'

# 프르젝트 root path 설정
PROJECT_ROOT_PATH = rootdir.root_dir(__file__)

# cache 경로 설정
CACHE_PATH = f'{PROJECT_ROOT_PATH}/.cache'

# LLM CACHE 경로 설정
LLM_CACHE_PATH = f'{CACHE_PATH}/multimodal-rag_cache.db'

# VectorDB 저장 경로
VECTOR_DB_ROOT_PATH = f'{PROJECT_ROOT_PATH}/.vector_db'

# FAISS 저장 경로
FAISS_DB_PATH = f'{VECTOR_DB_ROOT_PATH}/faiss'

# FAISS index 이름
FAISS_INDEX_NAME = 'coupon'

# Chroma 저장 경로
CHROMA_DB_PATH = f'{VECTOR_DB_ROOT_PATH}/chroma'

# Chroma collection 이름
CHROMA_COLLECTION_NAME = "coupon"

# FILE UPLOAD 경로 설정
UPLOAD_FILE_PATH = './data/coupon_image_files'

# analytics 파일 경로 설정
ANALYTICS_FILE_PATH = './data/analytics.json'

# default LLM 설정
DEFAULT_LLM = 'OPENAI' # 'UPSTAGE', 'OLLAMA'

# default embedding model 설정
DEFAULT_EMBEDDING_MODEL = 'OPENAI' # 'UPSTAGE', 'OLLAMA'

# graph recursion limit 설정
GRAPH_RECURSION_LIMIT = 25

def get_llm(model=DEFAULT_LLM):
    if model == 'OPENAI':
        # model : gpt-4o-mini
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
            max_tokens=2048
        )
    elif model == 'UPSTAGE':
        # model : Upstage solar-pro
        llm = ChatUpstage(
            model="solar-pro",
            temperature=0.0,
            max_tokens=2048
        )
    elif model == 'OLLAMA':
        # model : EEVE-Korean-Instruct-10.8B:latest
        llm = ChatOllama(
            model="EEVE-Korean-Instruct-10.8B:latest",  
            temperature=0.0,
            max_tokens=2048
        )
    return llm

def get_embedding(embedding_model=DEFAULT_EMBEDDING_MODEL):
    if embedding_model == 'OPENAI':
        # OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    elif embedding_model == 'UPSTAGE':
        # UpstageEmbeddings
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    elif embedding_model == 'OLLAMA':
        # ollama pull nomic-embed-text 실행 필요
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings