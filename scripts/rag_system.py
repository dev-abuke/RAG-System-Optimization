import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from datasets import load_dataset

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.retrieval_chain = None
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def load_and_process_data(self, num_articles=1000):
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]", cache_dir="./cache")
        self.articles = [item['article'] for item in dataset]

        print(len(self.articles), "articles loaded", "The Test article is:", self.articles[0])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents(self.articles)

        print(len(docs), "chunks created")

        self.vector_store = Chroma.from_documents(docs, self.embeddings)

    def create_context_prompt(self):

        # Prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt

    def setup_retrieval_qa(self):
        retriever = self.vector_store.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # Prompt
        prompt = self.create_context_prompt()
        # retrieval chain
        self.retrieval_chain = retriever | self.format_docs
        # qa chain
        self.qa_chain = (
            {"context": self.retrieval_chain, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_retrieval_qa() first.")
        return self.qa_chain.invoke(question)

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.load_and_process_data()
    rag_system.setup_retrieval_qa()

    # Example retrieval
    rag_system.retrieval_chain.invoke("Who decided to give one of her kidneys?")

    # Example query
    question = "Who decided to give one of her kidneys?"
    answer = rag_system.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")