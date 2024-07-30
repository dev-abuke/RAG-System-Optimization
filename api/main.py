import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_core.documents import Document

from datasets import load_dataset

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, SessionLocal
from .models import Base
from .routers import qa, history
from .retriever import get_retriever_instance

import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("The Logger INFO CWD:: "+ os.getcwd())

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include routers
app.include_router(qa.router)
app.include_router(history.router)

# allow all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def load_and_process_data( num_articles=1000):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]", cache_dir="./cache")

    articles = [Document(page_content=item['article'], metadata={"idx": item['id']}) for item in dataset]

    print(len(articles), "articles loaded", "The Test article is:", articles[0])

    return articles

# Initialize the retriever with documents
# documents = get_document_from_docx([], "data/raw/docx")

documents = load_and_process_data()
leng = len(documents)
print("The Document Sample as a list length is :: ", leng)
print("The Document Sample is :: ", type(documents[0]))
logger.info(f"The Logger INFO :: {len}")
retriever = get_retriever_instance(documents).store_documents(documents)