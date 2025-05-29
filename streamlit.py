import streamlit as st
import os
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages.chat import ChatMessage
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid
from packages.graphs.graph import app
from pprint import pprint
from common import CACHE_PATH, LLM_CACHE_PATH, PROJECT, UPLOAD_FILE_PATH, PROJECT_ROOT_PATH, GRAPH_RECURSION_LIMIT, ANALYTICS_FILE_PATH
from packages.graphs.loader import load_from_image
import numpy as np
import pandas as pd
import random
import uuid
from annotated_text import util
import streamlit_analytics2 as streamlit_analytics
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

load_dotenv(verbose=True, override=True)

# 캐시 디렉토리를 생성합니다.
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# SQLiteCache를 사용합니다.
set_llm_cache(SQLiteCache(database_path=LLM_CACHE_PATH))

# langsmith 추적 시작
PROJECT = 'arpo-app-streamlit'
logging.langsmith(PROJECT)

BUTTON_PREFIX = '💡 '

# LangGraphconfig 설정
config = RunnableConfig(recursion_limit=GRAPH_RECURSION_LIMIT, configurable={"thread_id": random_uuid()})

# 인증 설정
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

def header_view():
    st.title("🎁 :blue[G]ifticon :blue[I]mage :blue[F]inder 🔎")
    st.subheader("Modular RAG 기반 기프티콘 이미지 검색기")

def main_view():
    intro = '''- 등록된 기프티콘 이미지를 자연어로 검색할 수 있습니다.  
        - 상품명, 발행처, 사용처, 유효기간 등 다양한 조건으로 디테일한 검색이 가능합니다.  
        - 기프티콘 검색 이외의 질문에도 실시간 정보 기반의 LLM과 웹검색을 통해 최신의 답변을 제시합니다.  
        - 정확도를 높히기 위해 답변, 질문, 컨텍스트의 연관성과 hallucination 여부에 대해 평가합니다.  
        - 모든 평가를 통과할 때까지 질문을 재작성하거나 답변을 재생성하는 과정을 반복합니다.  
        - 랜덤으로 제시되는 :blue[[**💡질문 예시**]] 를 사용하여 GIF모델이 제시하는 높은 정확도의 답변을 확인해 보세요.  
        - 좌측의 :blue[[**실행 결과**]] 에서 답변을 제공한 Tool과 답변의 출처를 확인할 수 있습니다.  
            ✔ Tools : 이미지 검색기, LLM, 웹검색
        '''
    st.markdown(intro)
    
    # 탭을 생성
    search_tab, upload_tab = st.tabs(["검색", "기프티콘 이미지 등록"])

    # 쿠폰 이미지 파일 업로드
    uploaded_file = upload_tab.file_uploader("검색을 위해 기프티콘 이미지 파일을 등록하세요.", type=["png", "jpg", "jpeg"])

    # 처음 1번만 실행하기 위한 코드
    if "messages" not in st.session_state:
        # 대화기록을 저장하기 위한 용도로 생성한다.
        st.session_state["messages"] = []

    # 사이드바 생성
    with st.sidebar:    
        st.header('[실행 결과]')

        tool_div = st.container()
        result_div = st.container()

    # sample question
    def get_question_sample():
        question_samples = st.session_state["question_samples"]
        rnd = random.randrange(0, len(question_samples))
        return question_samples[rnd]

    def on_button_click():
        invoke(st.session_state.question_sample)
        
        button_div.empty()
        question_sample = get_question_sample()
        button_div.button(f'{BUTTON_PREFIX} {question_sample}', key=uuid.uuid1(), use_container_width=True,
                        type='secondary', on_click=on_button_click)
        st.session_state["question_sample"] = question_sample

    if "question_samples" not in st.session_state:
        r_cols = ['question']
        question_samples = pd.read_csv(f'{PROJECT_ROOT_PATH}/data/question_sample.csv', names=r_cols,  sep=',', encoding='utf-8')
        question_samples = question_samples['question'].values.tolist()
        st.session_state["question_samples"] = question_samples

    if "question_sample" not in st.session_state:
        st.session_state["question_sample"] = get_question_sample()

    # 이전 대화를 출력
    def print_messages():
        for chat_message in st.session_state["messages"]:
            search_tab.chat_message(chat_message.role).write(chat_message.content)

    # 새로운 메시지를 추가
    def add_message(role, message):
        st.session_state["messages"].append(ChatMessage(role=role, content=message))

    # 등록한 이미지에서 추출한 text 정보를 embedding하여 vector DB에 저장
    @st.cache_resource(show_spinner="이미지를 분석하고 있습니다...")
    def embed_file(file):
        file_content = file.read()
        file_path = f"{UPLOAD_FILE_PATH}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        coupon_info = load_from_image(file_path)
        return coupon_info

    # 초기화 버튼 생성
    # clear_btn = search_tab.button("검색 내용 초기화")
    # if clear_btn:
    #     st.session_state["messages"] = []

    # 파일이 업로드 되었을 때
    if uploaded_file:
        coupon_info = embed_file(uploaded_file)
        upload_tab.caption("기프티콘 이미지 등록 결과")
        upload_tab.json(coupon_info, expanded=2)

        # 이미지가 등록되었다면 sidebar에 이미지 출력
        if "coupon_metadata" in coupon_info:    
            st.sidebar.image(coupon_info["coupon_metadata"][0]["image_path"])
        

    # 이전 대화 기록 출력
    print_messages()

    # 사용자의 입력
    user_input = st.chat_input("기프티콘을 찾아 보세요!")

    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    # 앱 실행
    def invoke(user_input):
        # 사용자의 입력
            search_tab.chat_message("user").write(user_input)

            # 앱 실행
            try:
                response = app.invoke(input={"questions": [user_input]}, config=config)
                # graph.invoke({"aggregate": []}, {"recursion_limit": 3})
            except GraphRecursionError:
                print("Recursion Error")
                search_tab.chat_message("assistant").write("일시적인 에러가 발생했습니다. 다시 시도해 주세요.")
                return

            pprint(response)

            answer = response['generation']['answer']
            # search_tab.markdown(answer)
            search_tab.chat_message("assistant").markdown(answer)

            # 사용한 tool 보여줌
            tool_div.markdown(
                util.get_annotated_html((response['used_tool'], "Tool")),
                unsafe_allow_html=True,
            )

            for image_path in response['generation']['image_path']:
                result_div.image(image_path)

            for source_url in response['generation']['source_url']:
                result_div.link_button(source_url, source_url)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", answer)

    # 사용자 입력
    if user_input:
        if user_input == "clear" or user_input == "초기화" or user_input == "삭제":
            search_tab.empty()
            st.session_state["messages"] = []
        else:
            invoke(user_input)

    with search_tab:
        # hint buttion
        button_div = st.empty()
        button_div.button(f'{BUTTON_PREFIX} {st.session_state.question_sample}', key=uuid.uuid1(), use_container_width=True,type='secondary', on_click=on_button_click)

# 트래킹 시작
streamlit_analytics.start_tracking(load_from_json=ANALYTICS_FILE_PATH)

print(st.session_state['authentication_status'])

# 인증 여부에 따라 view를 설정합니다.
if st.session_state['authentication_status']:
    header_view()
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
    authenticator.logout(location='sidebar')
    main_view()
elif st.session_state['authentication_status'] is False:
    header_view()
    st.error("Username/password is incorrect.  \n If you don't see the login input form, clear your browser cookies and try again.")
    authenticator.login(location='main')
elif st.session_state['authentication_status'] is None:
    header_view()
    st.warning("Please enter your username and password.  \n If you don't see the login input form, clear your browser cookies and try again.")
    authenticator.login(location='main')

# 트래킹 종료
streamlit_analytics.stop_tracking(save_to_json=ANALYTICS_FILE_PATH)