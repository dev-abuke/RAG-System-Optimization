import pandas as pd

def get_questions(number_of_questions = 3):
    merged_dataframe = pd.read_csv('/teamspace/studios/this_studio/RAG-System-Optimization/data/merged_testset.csv')

    # get the question and ground truth from the merged dataframe and store in three_question and three_ground_truth
    num_question = merged_dataframe['question'][5:5 + number_of_questions]
    num_ground_truth = merged_dataframe['ground_truth'][5:5 + number_of_questions]

    questions = []
    ground_truths = []

    for question, ground_truth in zip(num_question, num_ground_truth):
        questions.append(question)
        ground_truths.append(ground_truth)

    return questions, ground_truth