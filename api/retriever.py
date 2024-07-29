# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import WeaviateHybridSearchRetriever
from qdrant_client import QdrantClient

from .factory import get_text_splitter
from .config import load_config

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

class Retriever:
    def __init__(self, documents: list):
        self.documents = documents

        self.persist_directory = 'db'

        self.retriever_type = config["retriever"]

        if config["retriever"] == "chroma_dense":
            self.retriever = Chroma(persist_directory=self.persist_directory, embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
        if config["retriever"] == "qdrant_dense":
            client = QdrantClient(
                path="db/qdrant",
            )
            self.retriever = Qdrant(
                client=client,
                embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
                collection_name="rizzbuzz"
            )
        elif config["retriever"] == "hybrid":
            print("We are Using Weaviate Hybrid Search Retriever")
            
            import weaviate

            auth_config = weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

            client = weaviate.Client(
                url="https://muadizqtuzo6witpfdq.c0.us-east1.gcp.weaviate.cloud",
                additional_headers={
                        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
                },
                auth_client_secret=auth_config
            )

            self.retriever = WeaviateHybridSearchRetriever(
                client=client,
                index_name="Metala",
                text_key="hybrid_search_text",
                attributes=[],
                create_schema_if_missing=True,
            )
            print("Loaded the Hybrid Search Retriever", self.retriever)
        else:
            raise ValueError("Invalid retriever type")
            
    def store_documents(self, documents: list[Document]):
        logger.info(f"Storing documents From function {len(documents)}")

        splits = get_text_splitter().split_documents(documents)

        logger.info(f"Splits {len(splits)}")

        print(
            "The Split Sample is :: ", splits[0]
        )
        
        docs = self.retriever.add_documents(splits)
        
        logger.info(f"Documents added {docs[0]}")

    def retrieve(self, query: str):
        results = self.retriever.similarity_search(query, k = 2)
        return results
    def get_retriever(self):
        if config["retriever"] == "dense" or config["retriever"] == "qdrant_dense":
            return self.retriever.as_retriever(search_kwargs={'k': 2})
        elif config["retriever"] == "hybrid":
            return self.retriever

# Singleton instance of the Retriever
retriever_instance = None

def get_retriever_instance(documents: list = None):
    global retriever_instance
    if retriever_instance is None:
        retriever_instance = Retriever(documents)
    return retriever_instance
