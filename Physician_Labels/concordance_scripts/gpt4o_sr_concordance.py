#!/usr/bin/env python3
"""SR concordance matching (same logic as notebooks/sr_concordance/GPT4o_Concordance_Rerun + Stats Check.ipynb)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402


def main() -> None:
    d = paths.SR_CONCORDANCE_DATA
    df = pd.read_csv(d / "MAJORITY_Vote_GPT4o_Self_Reported__Labels.csv")
    df2 = pd.read_csv(d / "Centaur_1300_FirstRound.csv")
    physician_df = pd.read_csv(d / "Physician_Selected_Sentence_IDs.csv")

    print(f"Initial df length: {len(df)}")
    print(f"Initial df2 length: {len(df2)}")
    print(f"Physician df length: {len(physician_df)}")

    result = pd.merge(df2, df, left_on="Origin", right_on="ID", how="left")

    def collect_column_numbers(row, value_list):
        matching_columns = []
        skip_columns = [
            "ID",
            "Origin",
            "high",
            "low/irr",
            "high_number",
            "low_number",
            "sen_num_diff",
            "match_percentage",
            "human_Contain_Match",
        ]
        for col in row.index:
            if col not in skip_columns:
                if any(val in str(row[col]) for val in value_list):
                    match = re.search(r"label_(\d+)", col)
                    if match:
                        matching_columns.append(match.group(1))
        return ", ".join(matching_columns) if matching_columns else ""

    result["high"] = result.apply(lambda row: collect_column_numbers(row, ["High Relevance"]), axis=1)
    result["low/irr"] = result.apply(
        lambda row: collect_column_numbers(row, ["Low Relevance", "Irrelevant"]), axis=1
    )

    def count_items(value):
        if not value or pd.isna(value):
            return 0
        return len(value.split(", "))

    result["high_number"] = result["high"].apply(count_items)
    result["low_number"] = result["low/irr"].apply(count_items)

    print("\nFirst few rows of physician_df:")
    print(physician_df.head())

    join_column = None
    for col in ["ID", "Origin", "id", "origin"]:
        if col in physician_df.columns:
            join_column = col
            print(f"Found join column in physician_df: {join_column}")
            break

    if join_column:
        if "human_sentence_ids" in physician_df.columns:
            result = pd.merge(
                result,
                physician_df[[join_column, "keep_k", "human_sentence_ids"]],
                left_on="Origin",
                right_on=join_column,
                how="left",
            )
            if join_column != "Origin":
                result.drop(columns=[join_column], inplace=True)

            result["diff_numeric"] = result["high_number"] - result["keep_k"]
            result["sen_num_diff"] = result.apply(
                lambda row: f"+{row['diff_numeric']}"
                if row["diff_numeric"] > 0
                else f"{row['diff_numeric']}",
                axis=1,
            )

            def calculate_match_percentage(row):
                if pd.isna(row["high"]) or pd.isna(row["human_sentence_ids"]):
                    return 0.0
                high_set = set(row["high"].split(", ") if row["high"] else [])
                human_set = set(
                    str(row["human_sentence_ids"]).split(", ")
                    if not pd.isna(row["human_sentence_ids"])
                    else []
                )
                matches = high_set.intersection(human_set)
                denominator = max(row["high_number"], row["keep_k"])
                if denominator == 0:
                    return 0.0
                return (len(matches) / denominator) * 100

            def calculate_human_contain_match(row):
                if (
                    pd.isna(row["high"])
                    or pd.isna(row["human_sentence_ids"])
                    or row["keep_k"] == 0
                ):
                    return 0.0
                high_set = set(row["high"].split(", ") if row["high"] else [])
                human_set = set(
                    str(row["human_sentence_ids"]).split(", ")
                    if not pd.isna(row["human_sentence_ids"])
                    else []
                )
                matches = high_set.intersection(human_set)
                return (len(matches) / row["keep_k"]) * 100

            result["match_percentage"] = result.apply(calculate_match_percentage, axis=1)
            result["human_Contain_Match"] = result.apply(calculate_human_contain_match, axis=1)

            diff_mean = result["diff_numeric"].mean()
            diff_std = result["diff_numeric"].std()
            match_mean = result["match_percentage"].mean()
            match_std = result["match_percentage"].std()
            human_match_mean = result["human_Contain_Match"].mean()
            human_match_std = result["human_Contain_Match"].std()

            print("\nStatistics for sentence number differences:")
            print(f"Mean difference: {diff_mean:.2f}")
            print(f"Standard deviation: {diff_std:.2f}")
            print("\nStatistics for match percentage (matches / max(high_number, keep_k)):")
            print(f"Mean match percentage: {match_mean:.2f}%")
            print(f"Standard deviation: {match_std:.2f}%")
            print("\nStatistics for human_Contain_Match (matches / keep_k):")
            print(f"Mean percentage: {human_match_mean:.2f}%")
            print(f"Standard deviation: {human_match_std:.2f}%")

            print("\nMatch percentage statistics by data_source_df3 category:")
            print("\nFor match_percentage:")
            match_by_source = result.groupby("data_source_df3")["match_percentage"].agg(["mean", "std", "count"])
            match_by_source.columns = ["Mean Match %", "Std Dev", "Count"]
            match_by_source["Mean Match %"] = match_by_source["Mean Match %"].round(2)
            match_by_source["Std Dev"] = match_by_source["Std Dev"].round(2)
            print(match_by_source.sort_values("Mean Match %", ascending=False))

            print("\nFor human_Contain_Match:")
            human_match_by_source = result.groupby("data_source_df3")["human_Contain_Match"].agg(
                ["mean", "std", "count"]
            )
            human_match_by_source.columns = ["Mean Match %", "Std Dev", "Count"]
            human_match_by_source["Mean Match %"] = human_match_by_source["Mean Match %"].round(2)
            human_match_by_source["Std Dev"] = human_match_by_source["Std Dev"].round(2)
            print(human_match_by_source.sort_values("Mean Match %", ascending=False))
        else:
            print("Warning: 'human_sentence_ids' column not found in physician_df")
            print("Available columns in physician_df:", physician_df.columns.tolist())
    else:
        print("Warning: Could not find appropriate join column in physician_df")

    print(f"Final result length: {len(result)}")
    print(f"Length preserved: {len(result) == len(df2)}")
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)
    print("\nFirst 5 rows of the result dataframe:")
    print(result.head())

    out = d / "GPT4o_SR_Concordance_Result.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
