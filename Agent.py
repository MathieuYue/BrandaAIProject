import os
from dotenv import load_dotenv
load_dotenv()

from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host="https://us.cloud.langfuse.com", # 🇺🇸 US region
)

from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore

url = "9cb98d40-7c3b-441b-9b6d-4f1ac13eb1fa.europe-west3-0.gcp.cloud.qdrant.io"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

prompt_template = PromptTemplate.from_template(
    """You are a knowledgeable assistant tasked with answering questions. Use only the provided context to generate the most accurate and relevant answer. If the answer cannot be answered by the provided context, say that you do not know.
        
        Context: {context}

        Question: {question}
    """
    )

llm = ChatOpenAI(model="gpt-3.5-turbo")

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"), model="rerank-english-v3.0"
)

qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=url,
        prefer_grpc=True,
        api_key=os.getenv("QDRANT_CLUSTER_KEY"),
        collection_name="brandeis.edu"
    )

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=qdrant.as_retriever(search_kwargs={"k": 20})
)
@chain
def retrieve_and_rerank(query):
    compressed_docs = compression_retriever.get_relevant_documents(query)
    for doc in compressed_docs:
        print(doc)
        print('\n')
    return compressed_docs

# @chain
# def retrieve_vectordb(query):
#     qdrant = QdrantVectorStore.from_existing_collection(
#         embedding=embeddings,
#         url=url,
#         prefer_grpc=True,
#         api_key=os.getenv("QDRANT_CLUSTER_KEY"),
#         collection_name="brandeis.edu"
#     )
#     results = qdrant.similarity_search(query, k=5)
#     return results

@chain
def construct_prompt(passthrough_object):
    context = passthrough_object.get("context")
    question = passthrough_object.get("question")
    prompt = prompt_template.format(context=context, question=question)
    return prompt

chain = {"context": retrieve_and_rerank, "question": RunnablePassthrough()} | construct_prompt | llm | StrOutputParser()

def run():
    while (True):
        query = input("Enter question here: ")
        if query == 'exit':
            break
        result = chain.invoke(input = query, config={"callbacks": [langfuse_handler]})
        print(result)

run()