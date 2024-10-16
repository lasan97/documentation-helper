from typing import Any

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from consts import INDEX_NAME

load_dotenv()

def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docserach = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini")
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docserach.as_retriever(),
        return_source_documents=True,
    )

    return qa({"query": query})


def test(query: str):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    print(llm.invoke(query))


if __name__ == "__main__":
    # test("RetrievalQA 체인에 대해 설명해줘")
    print(run_llm(query="RetrievalQA 체인에 대해 설명해줘"))
