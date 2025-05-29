import rootdir
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from kiwipiepy import Kiwi
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from common import FAISS_DB_PATH, FAISS_INDEX_NAME, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, get_llm, get_embedding
from packages.graphs.consts import RETRIEVER_K

# 저장된 데이터를 로드
coupon_db = FAISS.load_local(
    folder_path=FAISS_DB_PATH,
    index_name=FAISS_INDEX_NAME,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True,
    )

# FAISS retriever
# search_type="similarity"
# faiss = coupon_db.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K}

# search_type="mmr"
faiss = coupon_db.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": RETRIEVER_K, "fetch_k": 10}
    )


# Chroma retriever
# Chroma 로딩
chroma_coupon_db = Chroma(
    persist_directory=CHROMA_DB_PATH, 
    embedding_function=get_embedding(), 
    collection_name=CHROMA_COLLECTION_NAME
    )
# 메타데이터 필드 정보 생성
metadata_field_info = [
    AttributeInfo(
        name="coupon_code",
        description="The code of <기프티콘> or <쿠폰>",
        type="string",
    ),
    AttributeInfo(
        name="order_number",
        description="The number the order",
        type="string",
    ),
    AttributeInfo(
        name="publisher",
        description="The publisher that <기프티콘> or <쿠폰> was published",
        type="string",
    ),
    AttributeInfo(
        name="valid_year",
        description="Modified year of the valid date of <기프티콘> or <쿠폰>",
        type="integer",
    ),
    AttributeInfo(
        name="valid_month",
        description="Modified month of the valid date of <기프티콘> or <쿠폰>",
        type="integer",
    ),
    AttributeInfo(
        name="valid_day",
        description="Modified day of the valid date of <기프티콘> or <쿠폰>",
        type="integer",
    ),
]

# SelfQueryRetriever 생성
chroma_retriever = SelfQueryRetriever.from_llm(
    llm=get_llm(),
    search_type="mmr",
    vectorstore=chroma_coupon_db,
    document_contents="Summary of <기프티콘> or <쿠폰>",
    metadata_field_info=metadata_field_info
    )

# Define your custom prompt template
# custom_template = '''
#     Your goal is to structure the user's query to match the request schema provided below.

#     << Structured Request Schema >>
#     When responding use a markdown code snippet with a JSON object formatted in the following schema:

#     ```json
#     {
#         "query": string \ text string to compare to document contents
#         "filter": string \ logical condition statement for filtering documents
#     }
#     ```

#     The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

#     A logical condition statement is composed of one or more comparison and logical operation statements.

#     A comparison statement takes the form: `comp(attr, val)`:
#     - `comp` (eq | ne | gt | gte | lt | lte): comparator
#     - `attr` (string):  name of attribute to apply the comparison to
#     - `val` (string): is the comparison value

#     A logical operation statement takes the form `op(statement1, statement2, ...)`:
#     - `op` (and | or): logical operator
#     - `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

#     Make sure that you only use the comparators and logical operators listed above and no others.
#     Make sure that filters only refer to attributes that exist in the data source.
#     Make sure that filters only use the attributed names with its function names if there are functions applied on them.
#     Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
#     Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
#     Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.

#     << Example 1. >>
#     Data Source:
#     ```json
#     {
#         "content": "Lyrics of a song",
#         "attributes": {
#             "artist": {
#                 "type": "string",
#                 "description": "Name of the song artist"
#             },
#             "length": {
#                 "type": "integer",
#                 "description": "Length of the song in seconds"
#             },
#             "genre": {
#                 "type": "string",
#                 "description": "The song genre, one of "pop", "rock" or "rap""
#             }
#         }
#     }
#     ```

#     User Query:
#     What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

#     Structured Request:
#     ```json
#     {
#         "query": "teenager love",
#         "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
#     }
#     ```


#     << Example 2. >>
#     Data Source:
#     ```json
#     {
#         "content": "Lyrics of a song",
#         "attributes": {
#             "artist": {
#                 "type": "string",
#                 "description": "Name of the song artist"
#             },
#             "length": {
#                 "type": "integer",
#                 "description": "Length of the song in seconds"
#             },
#             "genre": {
#                 "type": "string",
#                 "description": "The song genre, one of "pop", "rock" or "rap""
#             }
#         }
#     }
#     ```

#     User Query:
#     What are songs that were not published on Spotify

#     Structured Request:
#     ```json
#     {
#         "query": "",
#         "filter": "NO_FILTER"
#     }
#     ```


#     << Example 3. >>
#     Data Source:
#     ```json
#     {
#         "content": "Summary of <기프티콘> or <쿠폰>",
#         "attributes": {
#         "coupon_code": {
#             "description": "The code of <\uae30\ud504\ud2f0\ucf58> or <\ucfe0\ud3f0>",
#             "type": "string"
#         },
#         "order_number": {
#             "description": "The number the order",
#             "type": "string"
#         },
#         "publisher": {
#             "description": "The publisher that <\uae30\ud504\ud2f0\ucf58> or <\ucfe0\ud3f0> was published",
#             "type": "string"
#         },
#         "valid_year": {
#             "description": "Modified year of the valid date of <\uae30\ud504\ud2f0\ucf58> or <\ucfe0\ud3f0>",
#             "type": "integer"
#         },
#         "valid_month": {
#             "description": "Modified month of the valid date of <\uae30\ud504\ud2f0\ucf58> or <\ucfe0\ud3f0>",
#             "type": "integer"
#         },
#         "valid_day": {
#             "description": "Modified day of the valid date of <\uae30\ud504\ud2f0\ucf58> or <\ucfe0\ud3f0>",
#             "type": "integer"
#         }
#     }
#     }
#     ```

#     User Query:
#     {query}

#     Structured Request:
# '''

# # Create a PromptTemplate instance from your custom template
# few_shot_prompt = PromptTemplate.from_template(custom_template)

# # Pass the custom prompt to the create_sql_query_chain function
# chain = create_sql_query_chain(get_llm(), chroma_coupon_db, prompt=few_shot_prompt)

# chroma_retriever = SelfQueryRetriever(
#     vectorstore=chroma_coupon_db,
#     llm_chain=chain,
#     use_original_query=True
# )

# BM25Retriever 설정
# FAISS로부터 documents를 추출합니다.
docs_dict = coupon_db.docstore.__dict__['_dict']

# list로 저장
docs = list(docs_dict.values())

result_summary_text = []
result_metadata = []

for doc in docs:
    result_summary_text.append(doc.page_content)
    result_metadata.append(doc.metadata)

kiwi = Kiwi()
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

kiwi_bm25 = BM25Retriever.from_texts(result_summary_text, metadatas=result_metadata, preprocess_func=kiwi_tokenize,  k=RETRIEVER_K)

retriever = EnsembleRetriever(
    retrievers=[kiwi_bm25, faiss, chroma_retriever],  # 사용할 검색 모델의 리스트
    weights=[0.4, 0.3, 0.3],  # 각 검색 모델의 결과에 적용할 가중치
    search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
)

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Cohere Reranker 설정
compressor = CohereRerank(model="rerank-multilingual-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# retriever 설정
retriever = compression_retriever