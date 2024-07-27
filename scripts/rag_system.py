import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from datasets import load_dataset

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None

    def load_and_process_data(self, num_articles=1000):
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")
        articles = [item['article'] for item in dataset]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents(articles)

        self.vector_store = Chroma.from_documents(docs, self.embeddings)

    def setup_retrieval_qa(self):
        retriever = self.vector_store.as_retriever()
        llm = OpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    def query(self, question: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_retrieval_qa() first.")
        return self.qa_chain.run(question)

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.load_and_process_data()
    rag_system.setup_retrieval_qa()

    # Example query
    question = "What is the capital of France?"
    answer = rag_system.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")