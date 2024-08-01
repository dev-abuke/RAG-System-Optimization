import pandas as pd
import requests
from datasets import Dataset
from ragas import evaluate
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    faithfulness,
    context_recall,
    answer_correctness
)

class RagasEvaluation:

    def __init__(
            self, test_name: str, 
            number_of_questions: int = 3, 
            test_sets_csv_path: str = '../data/20240801-092254testset.csv'):
        
        # make sure the class is initialized with a test_name else raise an error
        self.test_name = test_name

        self.number_of_questions = number_of_questions

        if not self.test_name:
            raise ValueError("Please specify a test_name.") 

        self.test_sets_df = pd.read_csv(test_sets_csv_path)

        if self.test_sets_df.shape[0] < self.number_of_questions:
            raise ValueError("The number of questions is greater than the number of rows in the test_sets_df.")

        self.questions, self.ground_truths, self.evolution_type = self.get_questions()

        self.dataset = self.get_dataset()

        self.result_df: pd.DataFrame = self.run_evaluation()

    def get_questions(self):

        # get the question and ground truth from the test_sets_df dataframe and store in num_question and num_ground_truth
        num_question = self.test_sets_df['question'][:self.number_of_questions]
        
        num_ground_truth = self.test_sets_df['ground_truth'][:self.number_of_questions]

        evolution_type = self.test_sets_df['evolution_type'][:self.number_of_questions]

        questions = []

        ground_truths = []

        for question, ground_truth in zip(num_question, num_ground_truth):
            questions.append(question)
            ground_truths.append(ground_truth)

        return questions, ground_truths, evolution_type

    def get_answers_and_context(self):

        url = f"http://127.0.0.1:8000/qa/{self.test_name}"
        answers = []
        contexts = []
        for q in self.questions:
            data = {
                "query": q
            }
            response = requests.post(url, json=data)

            answers.append(response.json()['response'])
            contexts.append(response.json()['context'])

        return answers, contexts

    def get_dataset(self):
        answers, contexts = self.get_answers_and_context()

        array_contexts = [[con] for con in contexts]
        # To dict
        data = {
            "question": self.questions,
            "answer": answers,
            "contexts": array_contexts,
            "ground_truth": self.ground_truths
        }

        # Convert dict to dataset
        dataset = Dataset.from_dict(data)

        return dataset

    def run_evaluation(self):

        result = evaluate(
            dataset = self.dataset, 
            llm = ChatOpenAI(temperature=0), # use gpt-4o to increase context window, but has high cost
            metrics=[
                context_recall,
                faithfulness,
                answer_correctness,
            ],
        )

        result_df = result.to_pandas()

        types = []

        for evol_type in self.evolution_type:
            types.append(evol_type)

        result_df['evolution_type'] = types

        result_df.to_csv(f"/teamspace/studios/this_studio/RAG-System-Optimization/data/{self.test_name}.csv", 
                  index=False
                  )

        return result_df

    def plot_evaluation(self):
        # calculate the average score of context precision per total score
        rows = self.result_df.shape[0]

        context_recall_sum = self.result_df['context_recall'].sum()
        faithfulness_sum = self.result_df['faithfulness'].sum()
        answer_correctness_sum = self.result_df['answer_correctness'].sum()

        context_recall_score = context_recall_sum / rows * 100
        faithfulness_score = faithfulness_sum / rows * 100

        answer_correctness_score = answer_correctness_sum / rows * 100

        # Visualization
        labels = ['Context Recall', 'Faithfulness', 'Answer Correctness']
        scores = [context_recall_score, faithfulness_score, answer_correctness_score]

        plt.figure(figsize=(6, 3))
        plt.bar(labels, scores, color=['darkBlue', 'green', 'purple'])
        # tilt the x axis labels to 45 degrees
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Metrics')
        plt.ylabel('Scores (%)')
        plt.title(f'{self.test_name} Evaluation Scores')
        plt.ylim(0, 100)

        # Annotate bars with scores
        for i, score in enumerate(scores):
            plt.text(i, score - 10, f'{score:.2f}%', ha='center', va='top', color='white')

        # Add legend for evolution types
        evolution_types = self.result_df['evolution_type'].unique()
        evolution_colors = ['red', 'blue', 'yellow', 'cyan']  # Define colors for each evolution type
        color_map = {evolution: color for evolution, color in zip(evolution_types, evolution_colors)}

        for evolution, color in color_map.items():
            plt.scatter([], [], color=color, label=evolution)  # Adding dots for legend

        plt.legend(title="Evolution Types")

        plt.show()

if __name__ == "__main__":
    result = RagasEvaluation("test_name").plot_evaluation()

    print(result)