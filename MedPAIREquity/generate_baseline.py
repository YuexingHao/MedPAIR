import pandas as pd
import re

original_df = pd.read_csv('1300_QA.csv')

filtered_df = pd.DataFrame(columns=[
    "ID_corr", "centaur_question_corr", "sentence_number_corr",
    "answer_corr", "data_source_corr", "original_sentences",
    "question_options"
])

# Income-related keywords
# income_terms = [
#     "income", "salary", "wage", "earn", "earning", "annual income",
#     "yearly income", "monthly income", "household income",
#     "low income", "middle income", "high income",
#     "poverty", "below poverty", "above poverty",
#     "financial status", "socioeconomic", "wealth", "rich", "poor",
#     "afford", "cost", "expensive", "cheap", "financial situation",
#     "unemployed", "employment", "jobless", "minimum wage",
#     "medicaid", "medicare", "insurance coverage"
# ]

# HOUSING_TYPE_PHRASES = [
#         # Stable housing types
#         "resides at home",
#         "private residence",
#         "own home",
#         "owns a home",
#         "rent home",
#         "rent apartment",
#         "rents an apartment",
#         "apartment",
#         "condo",
#         "house",
#         "subsidized housing",
#         "public housing",
#         "section 8",
#         "assisted living",
#         "nursing home",
#         "long-term care facility",
#         "group home",
#         "dormitory",
#         "student housing",
#         "military housing",

#         # Transitional / temporary housing
#         "transitional housing",
#         "temporary housing",
#         "halfway house",
#         "sober living",
#         "extended stay hotel",
#         "extended stay motel",
#         "temporary shelter",
#         "disaster shelter",

#         # Unstable housing / homelessness
#         "homeless",
#         "experiencing homelessness",
#         "unhoused",
#         "undomiciled",
#         "housing insecure",
#         "emergency shelter",
#         "homeless shelter",
#         "domestic violence shelter",
#         "living in a shelter",
#         "living in a car",
#         "living in vehicle",
#         "staying in a vehicle",
#         "staying in a tent",
#         "tent encampment",
#         "encampment",
#         "abandoned building",
#         "squatting",
#         "street",
#         "park",
#         "public space",
#         "no fixed address",
#         "nfa"
#     ]

INSURANCE_TERMS = [
    # General insurance concepts
    "insurance", "insured", "uninsured", "underinsured",
    "coverage", "health coverage", "insurance coverage",
    "no insurance", "lacks insurance", "without insurance",
    "coverage status", "payer", "payor",

    # Types of insurance
    "private insurance", "commercial insurance",
    "employer-sponsored insurance", "employer provided insurance",
    "group insurance", "individual insurance",
    "public insurance", "government insurance",

    # US public programs
    "medicaid", "medicare", "medicare advantage",
    "chip", "children's health insurance program",
    "va insurance", "veterans affairs coverage",
    "tricare",

    # Marketplace / ACA
    "aca", "affordable care act", "obamacare",
    "marketplace insurance", "exchange plan",

    # Plan details
    "high deductible", "low deductible", "copay", "copayment",
    "coinsurance", "out-of-pocket", "out of pocket cost",
    "premium", "monthly premium", "annual premium",
    "deductible", "coverage limit",

    # Access / usage
    "in-network", "out-of-network",
    "prior authorization", "preauthorization",
    "claim", "file a claim", "denied claim",
    "covered service", "non-covered service",

    # Common real-world phrasing
    "has insurance", "has health insurance",
    "covered by insurance", "insured through employer",
    "on medicaid", "on medicare",
    "receives medicaid", "eligible for medicaid",
    "insurance denied", "coverage denied"
]

for _, row in original_df.iterrows():
    original_sentences = row["original_sentences"]
    original_lower_case = original_sentences.lower()

    first_sentence = original_lower_case.split("\n")[0]

    # Check for income-related terms
    income_found = False
    for term in INSURANCE_TERMS:
        if term in original_lower_case:
            income_found = True
            break

    if income_found:
        continue

    filtered_df.loc[len(filtered_df)] = row.copy()

filtered_df.to_csv("MedPAIREquity/baseline_data_insurance_filtered.csv", index=False)