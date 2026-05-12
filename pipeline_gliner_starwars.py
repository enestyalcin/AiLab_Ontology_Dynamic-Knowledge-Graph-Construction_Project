import json
from gliner import GLiNER

# 1. LOAD DATASET
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

starwars_data = load_dataset("dataset_starwars.jsonl")
print(f"Loaded {len(starwars_data)} stories for advanced extraction.\n")

# 2. LOAD GLiNER MODEL
# We use a medium-sized GLiNER model which is very capable for Zero-Shot NER
model_name = "urchade/gliner_medium-v2.1"
print(f"Loading GLiNER model: {model_name}...")
model = GLiNER.from_pretrained(model_name)

ontology_labels = ["Character", "Place", "Weapon"]

# 3. ADVANCED ENTITY EXTRACTION
print("\n" + "="*50)
print("GLiNER ENTITY EXTRACTION TEST")
print("="*50)

test_story = starwars_data[0]
text = test_story["text"]
expected_triples = test_story["triples"]

print(f"TEXT: {text}\n")

# Predict entities based on our custom ontology labels
entities = model.predict_entities(text, ontology_labels)

print("EXTRACTED ENTITIES:")
extracted_entities_dict = {"Character": [], "Place": [], "Weapon": []}

for entity in entities:
    word = entity["text"]
    label = entity["label"]
    score = entity["score"]
    print(f"- {word} (Type: {label}, Confidence: {score:.2f})")
    
    # Store them for relation building
    extracted_entities_dict[label].append(word)


# 4. RELATION BUILDING (Heuristic Baseline)
print("\n" + "="*50)
print("ONTOLOGY-CONSTRAINED TRIPLE GENERATION")
print("="*50)

generated_triples = []

# Constraint 1: locatedIn (Domain: Character, Range: Place)
for char in extracted_entities_dict["Character"]:
    for place in extracted_entities_dict["Place"]:
        generated_triples.append([char, "locatedIn", place])

# Constraint 2: hasWeapon (Domain: Character, Range: Weapon)
for char in extracted_entities_dict["Character"]:
    for weapon in extracted_entities_dict["Weapon"]:
        generated_triples.append([char, "hasWeapon", weapon])

print("GENERATED TRIPLES (Based on extracted entities & ontology rules):")
for t in generated_triples:
    print(t)

print("\nEXPECTED TRIPLES (Ground Truth from JSONL):")
for t in expected_triples:
    print(t)

print("\nNext Phase: To improve relation accuracy, we would replace the heuristic rules")
print("in Step 4 with a Transformer sequence classifier (like BERT) or an LLM validator.")