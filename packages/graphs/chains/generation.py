from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pprint import pprint
from common import get_llm

llm = get_llm()
# prompt = hub.pull("rlm/rag-prompt")
# """
#     You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#     Question: {question}
#     Context: {context}
#     Answer:
# """
# When answering questions about expiration dates, the reference date must be TODAY.
prompt = PromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question in Korean. 
        Multiple answers are possible, so try to find all the answers from the following pieces of retrieved context you can.
        It's important to note that <기프티콘> and <쿠폰> are synonymous.
        If you don't know the answer, just say that you don't know.

        # Question: {question}
        # Context: {context}
        
        Results should be in the JSON format:
        - answer: Use seven sentences maximum and keep the answer concise in Korean.
        - image_path: [If context metadata used in the answer has <image_path>, this is a list of image_paths. If it doesn't have a value, provide an empty list.]
        - source_url: [If the context metadata used in the answer has <source_url>, this is a list of source_urls. If it doesn't have a value, provide an empty list.]
        """
    )
# pprint(prompt)
generation_chain = prompt | llm | JsonOutputParser()