import pandas as pd
import re

original_df = pd.read_csv('MedPAIREquity/1300_QA.csv')

columns = ["ID_corr", "race", "centaur_question_corr", "sentence_number_corr", "answer_corr", "data_source_corr", "race_injected_sentence", "question_options"]
equity_df = pd.DataFrame(columns=columns)

race_terms = ["white", "black", "asian", "hispanic", "native american", "pacific islander", "caucasian"]
ethnicities = [
    # East Asia
    "chinese", "korean", "japanese", "vietnamese", "thai", "laotian", "cambodian", "filipino", "indonesian", "malaysian",
    "burmese", "mongolian", "tibetan", "himalayan", "nepalese",

    # South Asia
    "indian", "pakistani", "bangladeshi", "sri lankan", "maldivian", "bhutanese", "kashmiri", "punjabi", "bengali", "tamil",

    # Central Asia & Caucasus
    "afghan", "iranian", "kurdish", "turkish", "azerbaijani", "armenian", "georgian", "kazakh", "uzbek", "tajik",
    "kyrgyz", "turkmen", "uyghur", "hazara", "pashtun",

    # Middle East / Arab
    "saudi", "yemeni", "omani", "emirati", "qatari", "bahraini", "lebanese", "syrian", "jordanian", "palestinian",
    "israeli", "iraqi", "egyptian",

    # North Africa
    "moroccan", "algerian", "tunisian", "libyan", "sudanese", "berber", "nubian", "coptic", "amazigh", "bedouin",

    # East Africa
    "ethiopian", "eritrean", "somali", "djiboutian", "kenyan", "tanzanian", "ugandan", "rwandan", "burundian", "maasai",

    # Central Africa
    "congolese", "angolan", "zambian", "zimbabwean", "malawian", "mozambican", "central african", "chadian", "gabonese", "equatorial guinean",

    # Southern Africa
    "namibian", "botswanan", "south african", "xhosa", "zulu", "sotho", "swazi", "lesotho", "madagascan", "mauritian",

    # West Africa
    "nigerian", "yoruba", "igbo", "hausa", "ghanaian", "ivorian", "senegalese", "malian", "guinean", "sierra leonean",
    "liberian", "burkinabe", "togolese", "beninese", "gambian",

    # Europe - Western
    "spanish", "portuguese", "french", "italian", "german", "english", "scottish", "welsh", "irish", "dutch",
    "belgian", "swiss", "austrian", "luxembourgish", "andorran",

    # Europe - Northern
    "swedish", "norwegian", "danish", "finnish", "icelandic", "estonian", "latvian", "lithuanian", "sami", "faroese",

    # Europe - Southern / Balkans
    "greek", "albanian", "croatian", "serbian", "bosnian", "montenegrin", "macedonian", "bulgarian", "romanian", "moldovan",

    # Europe - Eastern / Slavic
    "polish", "ukrainian", "belarusian", "russian", "czech", "slovak", "hungarian", "slovenian", "kosovar", "roma",

    # Caribbean
    "cuban", "puerto rican", "dominican", "haitian", "jamaican", "barbadian", "trinidadian", "bahamian", "grenadian", "saint lucian",

    # Central America
    "mexican", "guatemalan", "belizean", "honduran", "salvadoran", "nicaraguan", "costa rican", "panamanian", "mayan", "garifuna",

    # South America
    "brazilian", "argentinian", "chilean", "peruvian", "bolivian", "ecuadorian", "colombian", "venezuelan", "paraguayan", "uruguayan",
    "mapuche", "quechua", "aymara", "guarani", "ashaninka",

    # Indigenous North America
    "cherokee", "navajo", "cree", "ojibwe", "apache", "sioux", "blackfoot", "iroquois", "mohawk", "inuit",

    # Pacific Islands
    "hawaiian", "samoan", "tongan", "fijian", "papuan", "maori", "marshallese", "palauan", "chamorro", "micronesian",

    # Mixed / diaspora identifiers
    "afro-caribbean", "afro-latino", "mestizo", "mulatto", "creole", "ashkenazi jewish", "sephardic jewish", "mizrahic jewish", "romani", "basque"
]

race_terms = race_terms + ethnicities

demographics = [" man ", " woman ", " boy ", " girl ", " male ", " female "]

for _, row in original_df.iterrows():
    original_sentences = row["original_sentences"]
    original_lower_case = original_sentences.lower()
    first_sentence = original_lower_case.split("\n")[0]

    race_found = False
    for race in race_terms:
        if race in original_lower_case:
            race_found = True
            break

    if race_found:
        continue
    
    demographic_found = None
    for demographic in demographics:
        if demographic in first_sentence:
            demographic_found = demographic
            break
    
    if not demographic_found:
        continue


    # Add race term and make new row in dataframe
    black_patient = re.sub(
        rf"\b({re.escape(demographic_found)})\b", 
        r" black\1", 
        original_sentences, 
        count=1  # only replace the first occurrence
    )
    black_q = "### Step 1: Read excerpt\n" + black_patient + "\n### Step 2: Answer QA Details\n" + row["question_options"] + "\n### Step 3: Determine sentence relevance\n" + black_patient
    equity_df.loc[len(equity_df)] = [row["ID_corr"], "black", black_q, row["sentence_number_corr"], row["answer_corr"], row["data_source_corr"], black_patient, row["question_options"]]

    white_patient = re.sub(
        rf"\b({re.escape(demographic_found)})\b", 
        r" white\1", 
        original_sentences, 
        count=1  # only replace the first occurrence
    )
    white_q = "### Step 1: Read excerpt\n" + white_patient + "\n### Step 2: Answer QA Details\n" + row["question_options"] + "\n### Step 3: Determine sentence relevance\n" + white_patient
    equity_df.loc[len(equity_df)] = [row["ID_corr"], "white", white_q, row["sentence_number_corr"], row["answer_corr"], row["data_source_corr"], white_patient, row["question_options"]]


equity_df.to_csv("MedPAIREquity/equity_data.csv", index=False)