from langchain_core.documents import Document
from datasets import load_dataset
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd

def generate_testset(test_size: int = 10) -> pd.DataFrame:
    """
    Generate a testset using the TestsetGenerator class from the ragas package.

    Args:
        test_size (int): The number of test samples to generate. Default is 6.

    Returns:
        pd.DataFrame: The generated testset as a pandas DataFrame.
    """
    dataset = load_dataset(
        "cnn_dailymail", "3.0.0", split="validation[:1000]", cache_dir="./cache"
    )

    articles = [
        Document(page_content=item["article"], metadata={"id": item["id"]})
        for item in dataset
    ]

    article_slice = articles[200:250]

    # Generator with OpenAI models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

    critic_llm = ChatOpenAI(model="gpt-4o-mini")

    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm, critic_llm, embeddings
    )

    # Generate testset
    testset = generator.generate_with_langchain_docs(
        article_slice,
        test_size=test_size,
        distributions={simple: 0.25, reasoning: 0.35, multi_context: 0.4},
    )

    synthetic = testset.to_pandas()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")

    synthetic.to_csv(
        f"/teamspace/studios/this_studio/RAG-System-Optimization/data/{timestamp}testset.csv",
        index=False,
    )

    return synthetic

if __name__ == "__main__":
    generate_testset()