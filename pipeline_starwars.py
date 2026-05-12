import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 1. LOAD DATASET (JSONL file)
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset.append(data)
    return dataset

# Load the file 
starwars_data = load_dataset("dataset_starwars.jsonl")
print(f"Successfully loaded {len(starwars_data)} stories!\n")

# 2. LOAD MODEL & TOKENIZER (Hugging Face)
# For this demo, we use a pre-trained NER (Named Entity Recognition) model.
# 'xlm-roberta-large-finetuned-conll03-english' is good at finding Persons and Locations.

model_name = "xlm-roberta-large-finetuned-conll03-english"

print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline to handle tokenization and inference automatically
nlp_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")



# 3. ANALYZE TEXT (Benchmarking Baseline)
print("\nStarting extraction...\n" + "-"*40)

# We test the model using the first sentence from our dataset
test_text = starwars_data[0]["text"]
expected_triples = starwars_data[0]["triples"]

print(f"TEXT: {test_text}")
print(f"EXPECTED (Ground Truth): {expected_triples}\n")

# Apply the model to the text
extracted_entities = nlp_pipeline(test_text)

print("EXTRACTED ENTITIES (Detected by the model):")
for ent in extracted_entities:
    # The model uses standard labels like PER (Person), LOC (Location), ORG (Organization)
    print(f"- {ent['word']} (Confidence: {ent['score']:.2f}, Type: {ent['entity_group']})")

print("-" * 40)
print("NOTE: To detect custom classes like 'Weapon' or 'Lightsaber', we need to fine-tune the model or use a zero-shot approach like GLiNER in the next step!")