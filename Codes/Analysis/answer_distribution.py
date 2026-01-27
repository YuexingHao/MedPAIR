import argparse
import pandas as pd
import matplotlib.pyplot as plt

def calculate_relevance(df, experiment_name, save_file):
    # Store per-case relevance percentages
    answer_ct = {}

    for _, row in df.iterrows():
        answer = str(row["LLM_answer"]).strip()
        if answer:
            if answer not in answer_ct:
                answer_ct[answer] = 0
            answer_ct[answer] += 1

    try:
        with open(save_file, 'a') as file:
            file.write(f"{experiment_name}\n")
            file.write(f"Distribution of relevant sentences: {answer_ct}")
            file.write("\n")

    except Exception as e:
        print(f"Error: {e}")
    
    names = list(answer_ct.keys())
    values = list(answer_ct.values())
    plt.bar(names, values)
    plt.xlabel('Answer')
    plt.ylabel('Count')
    plt.title(f'{experiment_name} Answer Distribution')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Specify paths for input CSVs to be analyzed")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    args = parser.parse_args()

    input_csv = args.input_csv
    experiment_name = args.experiment
    save_file = "Results/Answer_Distribution.txt"

    df = pd.read_csv(input_csv)

    calculate_relevance(df, experiment_name, save_file)


if __name__ == "__main__":
    main()