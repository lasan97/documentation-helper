import os
from itertools import islice
from typing import Iterator, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.vectorstores import Pinecone as PineconLangChain
from pinecone import Pinecone

load_dotenv()

def ingest_docs(path:str, batch_size: int):
    print(f"dir:{path}")
    loader = ReadTheDocsLoader(path)

    raw_documents = loader.lazy_load()
    total = 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for docs in batched_docs_iterator(raw_documents, batch_size):
        print(f"loaded {len(docs)} documents")
        total += len(docs)

        documents = text_splitter.split_documents(documents=docs)
        print(f"Splitted into {len(documents)} chunks")

        change_document_source_path(documents)
        print(f"Going to insert {len(documents)} to pinecone")

        embeddings = OpenAIEmbeddings()
        PineconLangChain.from_documents(documents=documents, embedding=embeddings, index_name="langchain-doc")

        if total % 1000 == 0:
            print(f"Processed {total} documents")

    print(f"total loaded {total} documents")
    print("***** Added to Pinecone vectorstore vectors *****")


def batched_docs_iterator(docs_iterator: Iterator, batch_size: int) -> Iterator:
    """docs_iterator에서 batch_size만큼 문서를 가져옵니다."""
    while True:
        batch = list(islice(docs_iterator, batch_size))
        if not batch:
            break
        yield batch


def change_document_source_path(documents: List[Document]):
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})


if __name__ == '__main__':
    ingest_docs("langchain-docs/api.python.langchain.com/en/latest", 500)