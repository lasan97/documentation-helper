import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.vectorstores import Pinecone as PineconLangChain
from pinecone import Pinecone

def ingest_docs(path:str) -> None:
    print(f"dir:{path}")
    loader = ReadTheDocsLoader(path=path)
    raw_documents = loader.load()

    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to pinecone")

    embeddings = OpenAIEmbeddings()
    PineconLangChain.from_documents(documents=documents, embedding=embeddings, index_name="langchain-doc")

    print("***** Added to Pinecone vectorstore vectors *****")


if __name__ == '__main__':
    ingest_docs("langchain-docs/api.python.langchain.com/en/latest")