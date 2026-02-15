import pandas as pd


def merge_question_results(save_path,
                           gpt5,
                           gpt4o,
                           llama,
                           medgemma,
                           qwen):

    ids = pd.unique(gpt5["ID_corr"])
    answer_dict = {"A": 1,
                   "B": 2,
                   "C": 3, 
                   "D": 4,
                   "E": 5,
                   "F": 6,
                   "G": 7}

    rows = []

    for id in ids:
        gpt5_answer = gpt5.loc[gpt5["ID_corr"] == id, "GPT5_answer"].iloc[0]
        gpt4o_answer = gpt4o.loc[gpt4o["ID_corr"] == id, "LLM_answer"].iloc[0]
        llama_answer = llama.loc[llama["ID_corr"] == id, "LLM_answer"].iloc[0]
        medgemma_answer = medgemma.loc[medgemma["ID_corr"] == id, "LLM_answer"].iloc[0]
        qwen_answer = qwen.loc[qwen["ID_corr"] == id, "LLM_answer"].iloc[0]

        rows.append({
            # "ID_corr": id,
            "GPT5": answer_dict[gpt5_answer] if gpt5_answer in answer_dict else "NA",
            "GPT4o": answer_dict[gpt4o_answer] if gpt4o_answer in answer_dict else "NA",
            "Llama": answer_dict[llama_answer] if llama_answer in answer_dict else "NA",
            "MedGemma": answer_dict[medgemma_answer] if medgemma_answer in answer_dict else "NA",
            "Qwen": answer_dict[qwen_answer] if qwen_answer in answer_dict else "NA",
        })
    
    new_df = pd.DataFrame(rows)
    new_df.to_csv(save_path, index=False, header=False)

def main():
    dir = "Results/"
    gpt5_csv = f"{dir}gpt5/EquiPAIRQuestions/race_questions.csv"
    gpt4o_csv = f"{dir}gpt4o/equipair/race_questions.csv"
    llama_csv = f"{dir}llama/equipair/race_questions.csv"
    medgemma_csv = f"{dir}medgemma/equipair/race_questions.csv"
    qwen_csv = f"{dir}qwen/equipair/race_questions.csv"

    gpt5 = pd.read_csv(gpt5_csv)
    gpt4o = pd.read_csv(gpt4o_csv)
    llama = pd.read_csv(llama_csv)
    medgemma = pd.read_csv(medgemma_csv)
    qwen = pd.read_csv(qwen_csv)

    merge_question_results(f"{dir}RaceMerged.csv",
                           gpt5,
                           gpt4o,
                           llama,
                           medgemma,
                           qwen)

if __name__ == "__main__":
    main()