import cv2
import os
from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text
import warnings
from pprint import pprint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from common import FAISS_DB_PATH, FAISS_INDEX_NAME, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, get_llm, get_embedding
from packages.graphs.chains.image_evaluator import image_evaluator
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal

warnings.filterwarnings('ignore')

ERR_MSG_FAILED_IMAGE_UPLOAD = "기프티콘 또는 쿠폰 이미지 파일이 아닙니다."

class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)
        self.img_path = None
        self.ocr_result = {}

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)

        if self.ocr_result['description']:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = "No text detected."

        if debug:
            self.show_img_with_ocr()

        return ocr_text

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

    def show_img(self):
        plt_imshow(img=self.img_path)

    def show_img_with_ocr(self):
        img = cv2.imread(self.img_path)
        roi_img = img.copy()

        for text_result in self.ocr_result['bounding_poly']:
            text = text_result['description']
            tlX = text_result['vertices'][0]['x']
            tlY = text_result['vertices'][0]['y']
            trX = text_result['vertices'][1]['x']
            trY = text_result['vertices'][1]['y']
            brX = text_result['vertices'][2]['x']
            brY = text_result['vertices'][2]['y']
            blX = text_result['vertices'][3]['x']
            blY = text_result['vertices'][3]['y']

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

            topLeft = pts[0]
            topRight = pts[1]
            bottomRight = pts[2]
            bottomLeft = pts[3]

            cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)
            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 20, font_size=15)

            # print(text)

        plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))

class CouponSummary(BaseModel):
    title: str = Field(description="쿠폰이름")
    coupon_code: str = Field(description="쿠폰코드")
    publisher: str = Field(description="교환처")
    valid_date: str = Field(description="유효기간")
    order_number: str = Field(description="주문번호")
    summary: str = Field(description="쿠폰 정보 요약")
    valid_year: int = Field(description="유효기간의 연도")
    valid_month: int = Field(description="유효기간의 월")
    valid_day: int = Field(description="유효기간의 일")
    image_path: Optional[str] = None

def generate_coupon_summary(coupon_summary: CouponSummary):
    prompt = PromptTemplate.from_template(
        """
            당신은 주어진 쿠폰 정보를 자연스러운 한국어 문장으로 설명하는 agent입니다.
            아래의 주어진 정보 이외에는 절대 사용하면 안됩니다. 쿠폰코드는 필수로 포함되어야 하며 쿠폰에 대해 자연스러운 한국어 문장으로 설명해 주세요.
            - 쿠폰이름 : {title}
            - 쿠폰코드 : {coupon_code}
            - 발행처 : {publisher}
            - 유효기간 : {valid_date}
            - 주문번호 : {order_number}
            - 쿠폰요약 : {summary}
        """
    )

    input = {
        "title": coupon_summary.title, 
        "coupon_code": coupon_summary.coupon_code,
        "publisher": coupon_summary.publisher,
        "valid_date": coupon_summary.valid_date,
        "order_number": coupon_summary.order_number, 
        "summary": coupon_summary.summary
    }

    # llm 로딩
    llm = get_llm()
    
    # 체인 생성
    chain = prompt | llm

    output = chain.invoke(input)
    # print(output.content)
    return output.content

def inference_coupon_info_with_llm(sources: list):
    preprocess_prompt = PromptTemplate.from_template(
         """<WANT_TO_CACHE_HERE>
            당신은 쿠폰 이미지를 OCR 모델을 통해 추출한 키워드로 쿠폰 정보를 추론하는 agent입니다. 추론시 다음의 규칙을 지켜주세요.
            - 정보는 주어진 키워드에서만 사용하세요.
            - 쿠폰코드는 공백(space) 또는 대시(-)를 포함할 수 있으며 숫자로 구성되어 있습니다.
            - 유효기간은 종료일 기준으로 'YYYY년 mm월 dd일'의 형태로 작성해 주세요.
            - 유효기간의 연도는 종료일 기준으로 'YYYY'의 형태로 integer 타입으로 작성해 주세요.
            - 유효기간의 월은 종료일 기준으로 'm'의 형태로 integer 타입으로 작성해 주세요.
            - 유효기간의 일은 종료일 기준으로 'd'의 형태로 integer 타입으로 작성해 주세요.
            - 오타가 있을 수 있으니 맞춤법에 맞게 수정해 주세요.

            다음은 OCR 모델을 통해 추출한 쿠폰 키워드입니다. 이 정보를 바탕으로 쿠폰정보를 추론해 주세요.
            답변 출력시 추론에 대한 설명은 제외하고 아래 FORMAT을 참고해서 json형태로만 출력해 주세요.
            </WANT_TO_CACHE_HERE>
            
            KEYWORD:
            {coupon_info}

            FORMAT:
            {format}
        """
    )

    # PydanticOutputParser 생성
    preprocess_parser = PydanticOutputParser(pydantic_object=CouponSummary)

    # instruction 을 출력합니다.
    # print(preprocess_parser.get_format_instructions())
    preprocess_prompt = preprocess_prompt.partial(format=preprocess_parser.get_format_instructions())

    # llm 로딩
    preprocess_llm = get_llm()

    # 체인 생성
    preprocess_chain = preprocess_prompt | preprocess_llm

    # 결과 저장 리스트
    result_summary_text = []
    result_metadata = []
    result_ids = []

    # sources : coupon 정보 리스트
    for source in sources:
        input = {
            "coupon_info": source['coupon_info']
        }
        
        output = preprocess_chain.invoke(input)
        structured_output = preprocess_parser.parse(output.content)

        # 생성된 쿠폰 요약 추가
        structured_output.summary = generate_coupon_summary(structured_output)

        # 이미지 path 추가
        structured_output.image_path = source['image_path']
        
        result_ids.append(source['image_path'])
        result_summary_text.append(structured_output.summary)
        result_metadata.append(structured_output.__dict__)

    return {"id": result_ids, "coupon_summary_text": result_summary_text, "coupon_metadata": result_metadata}

def _save_to_faiss(coupon_infos: dict):
    # DB 로드
    coupon_db = FAISS.load_local(
        folder_path=FAISS_DB_PATH,
        index_name=FAISS_INDEX_NAME,
        embeddings=get_embedding(),
        allow_dangerous_deserialization=True,
    )

    # id가 DB에 있는지 확인하고 있으면 delete 실행
    is_existed_id = set(coupon_infos['id']).intersection(set(list(coupon_db.index_to_docstore_id.values())))
    if is_existed_id:
        coupon_db.delete(coupon_infos['id'])

    # DB에 추가
    result = coupon_db.add_texts(
        coupon_infos['coupon_summary_text'],
        metadatas=coupon_infos['coupon_metadata'],
        ids=coupon_infos['id']
    )

    # 로컬에 DB 저장
    coupon_db.save_local(folder_path=FAISS_DB_PATH, index_name=FAISS_INDEX_NAME)

    # 저장된 DB 내용 확인
    print('---SAVE TO FAISS COMPLETED---')
    pprint(result)

def _save_to_chroma(coupon_infos: dict):
    coupon_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=get_embedding(), collection_name=CHROMA_COLLECTION_NAME)

    # id가 DB에 있는지 확인하고 있으면 delete 실행
    is_existed_id = coupon_db.get(coupon_infos['id'])
    if is_existed_id:
        coupon_db.delete(coupon_infos['id'])
    
    # DB에 추가와 더불에 로컬에 DB 자동 저장
    result = coupon_db.add_texts(
        coupon_infos['coupon_summary_text'],
        metadatas=coupon_infos['coupon_metadata'],
        ids=coupon_infos['id']
    )

    # 저장된 DB 내용 확인
    print('---SAVE TO CHROMA COMPLETED---')
    pprint(result)

def add_coupon_to_vector_db(coupon_infos: dict):
    # FAISS에 저장
    _save_to_faiss(coupon_infos)

    # Chroma 저장
    _save_to_chroma(coupon_infos)

def load_from_image(image_path: str):    
    
    # 등록한 이미지가 쿠폰이나 기프티콘 이미지가 아닌 경우 error message 리턴    
    is_coupon_image = image_evaluator(image_path)
    print(is_coupon_image)

    # is_coupon_image : YES or NO 
    if 'NO' in is_coupon_image:
        return {"error" : True, "error_message" : ERR_MSG_FAILED_IMAGE_UPLOAD}
    
    # Pororo
    # ocr = PororoOcr()
    # ocr_result = ocr.run_ocr(image_path)
    # coupon_keyword = "','".join(ocr_result)
    # result = {
    #     "coupon_info": f"['{coupon_keyword}']",
    #     "image_path": image_path
    # }

    # gpt-4o
    llm = ChatOpenAI(
        temperature=0.0,
        max_tokens=2048,
        model_name="gpt-4o",
    )

    # OCR 모델 평가용 prompt
    system_prompt = """You are an Optical Character Recognition machine."""
    user_prompt = """You will extract all the characters from the image provided by the user, and you will only privide the extracted text in your response.
    As an OCR machine, You can only respond with the extracted text according to the following intruction.
    * Even if there are line breaks, if it is inferred that it is a product name, please represent it as a passage.
    * Answer with a string in the format of the example below.
    example : ['element_1','element_2',...,'elemnet_N']
    """

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )
    coupon_info_text = multimodal_llm.invoke(image_path, display_image=False)
    print(coupon_info_text)
    
    result = {
        "coupon_info": coupon_info_text,
        "image_path": image_path
    }

    coupon_infos = inference_coupon_info_with_llm([result])
    add_coupon_to_vector_db(coupon_infos)
    return coupon_infos


def _delete_faiss(ids):
    # DB 로드
    coupon_db = FAISS.load_local(
        folder_path=FAISS_DB_PATH,
        index_name=FAISS_INDEX_NAME,
        embeddings=get_embedding(),
        allow_dangerous_deserialization=True,
    )

    # document 삭제
    is_existed_id = set(ids).intersection(set(list(coupon_db.index_to_docstore_id.values())))
    if is_existed_id:
        coupon_db.delete(ids)
        # 로컬에 DB 저장
        coupon_db.save_local(folder_path=FAISS_DB_PATH, index_name=FAISS_INDEX_NAME)

        # 결과 출력
        print('---DELETE TO FAISS COMPLETED---')
        print(coupon_db.index_to_docstore_id)
    else:
        print('---DELETE TO FAISS FAILED---')
    

def _delete_chroma(ids):
    coupon_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=get_embedding(), collection_name=CHROMA_COLLECTION_NAME)
    coupon_db.delete(ids)
    # 결과 출력
    print('---DELETE TO CHROMA COMPLETED---')
    print(coupon_db.get())

# id로 document를 삭제 합니다.
def delete_by_id(ids):
    # delete FAISS
    _delete_faiss(ids)
    
    # Delete Chroma
    _delete_chroma(ids)

    # Delete file
    for id in ids:
        if os.path.isfile(id):
            os.remove(id)