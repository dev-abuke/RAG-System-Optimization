from langchain_core.documents import Document
from datasets import load_dataset
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd

def generate_testset(test_size=6):

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]", cache_dir="./cache")

    articles = [Document(page_content=item['article'], metadata={"id": item['id']}) for item in dataset]

    article_slice = articles[200:250]

    # generator with openai models
    generator_llm = critic_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    # critic_llm = ChatOpenAI(model="gpt-4") # uncomment to use gpt-4
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # generate testset
    testset = generator.generate_with_langchain_docs(article_slice, test_size=test_size, distributions={simple: 0.33, reasoning: 0.34, multi_context: 0.33})

    synthetic = testset.to_pandas()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")

    synthetic.to_csv(f"/teamspace/studios/this_studio/RAG-System-Optimization/data/{timestamp}testset.csv", index=False)

    return synthetic

if __name__ == "__main__":
    generate_testset()