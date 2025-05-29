import os, rootdir
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_teddynote import logging
from packages.graphs.graph import app
from pprint import pprint
from common import CACHE_PATH, LLM_CACHE_PATH, PROJECT

load_dotenv(verbose=True, override=True)

# 캐시 디렉토리를 생성합니다.
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# SQLiteCache를 사용합니다.
set_llm_cache(SQLiteCache(database_path=LLM_CACHE_PATH))

# langsmith 추적 시작
logging.langsmith(PROJECT)

if __name__ == "__main__":
    # print(app.invoke(input={"question": "굽네치킨 고추바사삭+콜라1.25L 쿠폰 가격을 알려줘"}))
    # result = app.invoke(input={"question": "치킨 쿠폰 정보를 알려줘"})
    # result = app.invoke(input={"question": "굽네치킨 고추바사삭+콜라1.25L 쿠폰 가격을 알려줘"})
    # result = app.invoke(input={"question": "스타벅스에서 사용가능한 기프티콘 또는 쿠폰을 알려줘"})
    # result = app.invoke(input={"question": "스타벅스 아이스 아메리카노 쿠폰 가격을 알려줘"})
    result = app.invoke(input={"questions": ["스타벅스 아이스 아메리카노 구매 사이트를 알려줘"]})
    # result = app.invoke(input={"question": "스타벅스 아이스아메리카노 쿠폰을 찾아줘"})
    # result = app.invoke(input={"question": "스타벅스에서 사용가능한 쿠폰을 찾아줘"})
    # result = app.invoke(input={"question": "스타벅스에서 사용가능한 쿠폰을 알려줘"})
    
    pprint(result)