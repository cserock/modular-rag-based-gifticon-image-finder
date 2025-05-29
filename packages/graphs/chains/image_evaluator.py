from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from utils.image_util import convert_to_base64
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal

def _prompt_func(data): 
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = [] 

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

def image_evaluator(image_path):
    # gpt-4o
    llm = ChatOpenAI(
        temperature=0.0,
        max_tokens=2048,
        model_name="gpt-4o",
    )

    # 기프티콘 이미지 평가용 prompt
    system_prompt = """You are an Coupon Image Recognition machine."""
    user_prompt = """"Is the image a coupon containing a barcode for ordering something? 
    Give a binary score 'YES' or 'NO'. 'YES' means that the image is coupon.
    """

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )
    output = multimodal_llm.invoke(image_path, display_image=False)
    
    # LLaVa 
    # mmlm = ChatOllama(model="llava:13b", temperature=0)
    # chain = _prompt_func | mmlm | StrOutputParser()

    # prompt_text = """Is the image a coupon containing a barcode for ordering something? Please answer Yes or No."""
    # image_b64 = convert_to_base64(image_path)
    # output = chain.invoke(  # 체인을 호출하여 쿼리를 실행합니다.
    #     {"text": prompt_text, "image": image_b64}
    # )
    return output