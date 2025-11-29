import pandas as pd

run1 = pd.read_csv("medpair_baseline/results/irr_removed_responses_run1.csv")
run2 = pd.read_csv("medpair_baseline/results/irr_removed_responses_run2.csv")
run3 = pd.read_csv("medpair_baseline/results/irr_removed_responses_run3.csv")

combined = pd.concat([run1, run2, run3])
ids = combined["ID_corr"].unique()
valid_labels = {"High Relevance", "Low Relevance", "Irrelevant"}

columns = ["ID_corr", 
           "centaur_question_corr", 
           "answer_corr", 
           "data_source_corr", 
           "majority_vote",
           "num_responses",
           "question_options"]
relevancy_complete = pd.DataFrame(columns=columns)

majority_acc = 0

reruns = []

data_source_cts = {"mmlu": 0, "jama": 0, "medxpert": 0, "medbullets": 0}
data_source_acc = {"mmlu": 0, "jama": 0, "medxpert": 0, "medbullets": 0}

for id in ids:
    question_df = combined[combined["ID_corr"] == id]
    new_question_df = pd.DataFrame(columns=columns)

    data_source = str(question_df.iloc[0]["data_source_corr"])
    num_correct = 0
    votes = {}

    for _, run in question_df.iterrows():
        correct = True
        
        valid_answers = {f"<answer>Option {ans}</answer>" for ans in "ABCDEFGHIJ"}

        if correct and str(run["GPT5_prediction"]) in valid_answers:
            new_question_df.loc[len(new_question_df)] = run
            if str(run["GPT5_prediction"]) not in votes:
                votes[str(run["GPT5_prediction"])] = 0
                votes[str(run["GPT5_prediction"])] += 1
            num_correct += 1
    
    if num_correct == 0:
        reruns.append(id)
        continue

    majority_vote = max(votes, key=votes.get)
    if majority_vote == f"<answer>Option {str(question_df.iloc[0]["answer_corr"])}</answer>":
        majority_acc += 1
        data_source_acc[data_source] += 1
    
    if num_correct == 1:
        data_source_cts[data_source] += 1
        new_row_contents = {"ID_corr": question_df.iloc[0]["ID_corr"],
                            "centaur_question_corr": question_df.iloc[0]["New_Sentences"],
                            "answer_corr": question_df.iloc[0]["answer_corr"],
                            "data_source_corr": question_df.iloc[0]["data_source_corr"],
                            "question_options": question_df.iloc[0]["question_options"],
                            "majority_vote": majority_vote,
                            "num_responses": 1}
        
        relevancy_complete.loc[len(relevancy_complete)] = new_row_contents
            

    elif num_correct == 2:
        data_source_cts[data_source] += 1
        new_row_contents = {"ID_corr": question_df.iloc[0]["ID_corr"],
                            "centaur_question_corr": question_df.iloc[0]["New_Sentences"],
                            "answer_corr": question_df.iloc[0]["answer_corr"],
                            "data_source_corr": question_df.iloc[0]["data_source_corr"],
                            "question_options": question_df.iloc[0]["question_options"],
                            "majority_vote": majority_vote,
                            "num_responses": 2}
        
        relevancy_complete.loc[len(relevancy_complete)] = new_row_contents
        

    elif num_correct == 3:
        data_source_cts[data_source] += 1
        new_row_contents = {"ID_corr": question_df.iloc[0]["ID_corr"],
                            "centaur_question_corr": question_df.iloc[0]["New_Sentences"],
                            "answer_corr": question_df.iloc[0]["answer_corr"],
                            "data_source_corr": question_df.iloc[0]["data_source_corr"],
                            "question_options": question_df.iloc[0]["question_options"],
                            "majority_vote": majority_vote,
                            "num_responses": 3}
        
        relevancy_complete.loc[len(relevancy_complete)] = new_row_contents

print(data_source_cts)
print(data_source_acc)
print(majority_acc)
print(len(relevancy_complete))
print(relevancy_complete["num_responses"].value_counts())
relevancy_complete.to_csv("medpair_baseline/results/gpt5-irr-removed-relevancy-combined.csv")