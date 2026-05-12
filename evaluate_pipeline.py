import json
from gliner import GLiNER


# 1. LOAD DATASET & MODEL
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

starwars_data = load_dataset("dataset_starwars.jsonl")
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
ontology_labels = ["Character", "Place", "Weapon"]


# 2. METRICS INITIALIZATION
true_positives = 0  # Correctly extracted triples
false_positives = 0 # Extracted triples that are WRONG
false_negatives = 0 # Expected triples that were MISSED

print("="*50)
print("STARTING BENCHMARK EVALUATION (10 Stories)")
print("="*50)

# 3. EXTRACTION & EVALUATION LOOP
for i, story in enumerate(starwars_data):
    text = story["text"]
    expected_triples = [tuple(t) for t in story["triples"]] # Convert to tuples for set comparison
    
    # Predict Entities
    entities = model.predict_entities(text, ontology_labels)
    extracted_entities_dict = {"Character": [], "Place": [], "Weapon": []}
    for entity in entities:
        extracted_entities_dict[entity["label"]].append(entity["text"])

    # Generate Triples (Heuristic / Rule-based)
    generated_triples = []
    
    # Rule 1: locatedIn
    for char in extracted_entities_dict["Character"]:
        for place in extracted_entities_dict["Place"]:
            generated_triples.append((char, "locatedIn", place))
            
    # Rule 2: hasWeapon
    for char in extracted_entities_dict["Character"]:
        for weapon in extracted_entities_dict["Weapon"]:
            generated_triples.append((char, "hasWeapon", weapon))
            
    # Rule 3: relativeOf (Heuristic: connect all characters found in the same sentence)
    chars = extracted_entities_dict["Character"]
    for j in range(len(chars)):
        for k in range(j + 1, len(chars)):
            generated_triples.append((chars[j], "relativeOf", chars[k]))
            generated_triples.append((chars[k], "relativeOf", chars[j]))

    # Compare generated vs expected using Sets
    gen_set = set(generated_triples)
    exp_set = set(expected_triples)
    
    tp = len(gen_set.intersection(exp_set))
    fp = len(gen_set - exp_set)
    fn = len(exp_set - gen_set)
    
    true_positives += tp
    false_positives += fp
    false_negatives += fn
    
    # Optional: Print progress for each sentence
    # print(f"Story {i+1}: TP={tp}, FP={fp}, FN={fn}")


# 4. CALCULATE FINAL SCORES
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print("FINAL BENCHMARKING RESULTS (GLiNER + Rule-Based)")
print("="*50)
print(f"Total True Positives (Correct):   {true_positives}")
print(f"Total False Positives (Wrong):    {false_positives}")
print(f"Total False Negatives (Missed):   {false_negatives}")
print("-" * 50)
print(f"Precision: {precision:.2%}  (How many of our predictions were actually correct?)")
print(f"Recall:    {recall:.2%}  (How many of the true triples did we manage to find?)")
print(f"F1-Score:  {f1_score:.2%}  (The harmonic mean of Precision and Recall)")
print("="*50)