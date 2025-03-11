import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

url = "9cb98d40-7c3b-441b-9b6d-4f1ac13eb1fa.europe-west3-0.gcp.cloud.qdrant.io"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

prompt_template = PromptTemplate.from_template(
    """You are a knowledgeable assistant tasked with answering questions. Use only the provided context to generate the most accurate and relevant answer. If the answer cannot be answered by the provided context, say that you do not know.
        
        Context: {context}

        Question: {question}
    """
    )

class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

llm = ChatOpenAI(model="gpt-3.5-turbo").with_structured_output(CitedAnswer)

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

def retrieve_and_rerank(query):
    compressed_docs = compression_retriever.invoke(query)

    source_map = {
        i: {
            "title": doc.metadata.get("title", "Unknown Title"),
            "url": doc.metadata.get("source"),
            "snippet": doc.page_content,
            "source_id": i
        }
        for i, doc in enumerate(compressed_docs)
    }

    formatted_context = "\n\n".join([
        f"Source ID: {i}\nArticle Snippet: {doc.page_content}" for i, doc in enumerate(compressed_docs)
    ])
    
    return {"formatted_context": formatted_context, "source_map": source_map}

@chain
def construct_prompt(passthrough_object):
    context = passthrough_object.get("context")
    question = passthrough_object.get("question")
    prompt = prompt_template.format(context=context, question=question)
    return prompt

chain = construct_prompt | llm

def cite_sources(citations, map):
    if not citations or not map:
        return ""
    output = "The above response was LLM generated based on information in the following sources, refer to these sources directly for most accurate information:\n\n"
    for c in citations:
        output += map[c]["title"] + "\n" + map[c]["url"] + "\n\n"
    return output

def run(query):
    retrieval_results = retrieve_and_rerank(query)
    source_map = retrieval_results["source_map"]
    results = chain.invoke({"context": retrieval_results["formatted_context"], "question": query})
    print(results.answer)
    cites = results.citations
    print("\n")
    print(cite_sources(cites, source_map))

run("what are the courses for brandeis cosi major")