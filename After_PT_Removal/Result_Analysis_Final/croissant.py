import json
import os
from huggingface_hub import login, upload_file

# Define the croissant metadata using the full ML Croissant JSON-LD format
croissant_metadata = {
    "@context": {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "arrayShape": "cr:arrayShape",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "data": {
            "@id": "cr:data",
            "@type": "@json"
        },
        "dataBiases": "cr:dataBiases",
        "dataCollection": "cr:dataCollection",
        "dataType": {
            "@id": "cr:dataType",
            "@type": "@vocab"
        },
        "dct": "http://purl.org/dc/terms/",
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isArray": "cr:isArray",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "personalSensitiveInformation": "cr:personalSensitiveInformation",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform"
    },
    "@type": "sc:Dataset",
    "distribution": [
        {
            "@type": "cr:FileObject",
            "@id": "repo",
            "name": "repo",
            "description": "The Hugging Face git repository.",
            "contentUrl": "https://huggingface.co/datasets/YuexingHao/MedPAIR",
            "encodingFormat": "git+https",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # This is a placeholder; replace with actual value if known
        },
        {
            "@type": "cr:FileSet",
            "@id": "raw-physician-labels",
            "containedIn": {"@id": "repo"},
            "encodingFormat": "text/csv",
            "contentUrl": "https://huggingface.co/datasets/YuexingHao/MedPAIR/resolve/main/Raw_Physician_Labels.csv",
            "includes": "Raw_Physician_Labels.csv"  # Added the includes property
        },
        {
            "@type": "cr:FileSet",
            "@id": "sentence-label-original",
            "containedIn": {"@id": "repo"},
            "encodingFormat": "text/csv",
            "contentUrl": "https://huggingface.co/datasets/YuexingHao/MedPAIR/resolve/main/Sentence_Label_Original_2k.csv",
            "includes": "Sentence_Label_Original_2k.csv"  # Added the includes property
        }
    ],
    "recordSet": [
        {
            "@type": "cr:RecordSet",
            "@id": "raw_physicians",
            "name": "raw_physicians",
            "description": "Raw labels provided by physicians",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": "raw_physicians/origin",
                    "dataType": "sc:Text",
                    "source": {
                        "fileSet": {"@id": "raw-physician-labels"},
                        "extract": {"column": "Origin"}
                    }
                },
                {
                    "@type": "cr:Field",
                    "@id": "raw_physicians/ground_truth",
                    "dataType": "sc:Text",
                    "source": {
                        "fileSet": {"@id": "raw-physician-labels"},
                        "extract": {"column": "Ground_Truth"}
                    }
                }
                # You would add more fields here based on the actual columns in your CSV
            ]
        },
        {
            "@type": "cr:RecordSet",
            "@id": "sentence_labels",
            "name": "sentence_labels",
            "description": "2,000 labeled sentences from the original dataset",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": "sentence_labels/origin",
                    "dataType": "sc:Text",
                    "source": {
                        "fileSet": {"@id": "sentence-label-original"},
                        "extract": {"column": "Origin"}
                    }
                },
                {
                    "@type": "cr:Field",
                    "@id": "sentence_labels/ground_truth",
                    "dataType": "sc:Text",
                    "source": {
                        "fileSet": {"@id": "sentence-label-original"},
                        "extract": {"column": "Ground_Truth_Answer"}
                    }
                }
                # You would add more fields here based on the actual columns in your CSV
            ]
        }
    ],
    "conformsTo": "http://mlcommons.org/croissant/1.1",
    "name": "MedPAIR",
    "description": "Medical dataset for preference alignment and instruction ranking",
    "alternateName": ["YuexingHao/MedPAIR", "MedPAIR"],
    "creator": {
        "@type": "Person",
        "name": "Yuexing Hao",
        "url": "https://huggingface.co/YuexingHao"
    },
    "keywords": ["medical", "healthcare", "NLP", "preference alignment", "instruction ranking"],
    "license": "https://opensource.org/licenses/MIT",
    "url": "https://huggingface.co/datasets/YuexingHao/MedPAIR",
    # Adding recommended properties
    "citeAs": "Hao, Y. (2023). MedPAIR: A Medical Dataset for Preference Alignment and Instruction Ranking",
    "datePublished": "2023-08-01",  # Replace with actual publish date if known
    "version": "1.0.0"
}

# Save the metadata to a file
with open("croissant.json", "w") as f:
    json.dump(croissant_metadata, f, indent=2)

print("Created croissant.json metadata file with fixes for validation errors")
print("Note: This is a template that should be modified based on your actual dataset structure.")
print("The field definitions in recordSet should match your actual CSV columns.")

# Option to manually download and view the file
print("\nTo view the generated file, you can run:")
print("cat croissant.json")

# Log in to Hugging Face and upload the file
try:
    print("\nDo you want to upload this file to your Hugging Face repository? (yes/no)")
    response = input().strip().lower()
    if response == "yes":
        login()
        upload_file(
            path_or_fileobj="croissant.json",
            path_in_repo="croissant.json",
            repo_id="YuexingHao/MedPAIR",
            repo_type="dataset"
        )
        print("Successfully uploaded croissant.json to your dataset repository!")
    else:
        print("Upload skipped. You can manually upload the croissant.json file to your Hugging Face dataset repository.")
except Exception as e:
    print(f"Error: {e}")
    print("You can manually upload the croissant.json file to your Hugging Face dataset repository.")