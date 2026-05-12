# Capstone Project: Dynamic Knowledge Graph Construction

**Course:** AI-LAB
**Professors:** Mario Lezoche & Avola  
**Author:** Enes-Tarik Yalcin  

## Project Overview
This repository contains an end-to-end "Data Refinery Pipeline" designed to transform unstructured text into structured RDF triples (a Knowledge Graph). The extraction process is strictly governed by a custom domain-specific ontology (Star Wars). 

The project benchmarks a baseline NLP architecture (Zero-Shot NER via GLiNER combined with heuristic rule-based relation extraction) to highlight the limitations of deterministic boundaries and justify the transition to advanced probabilistic ML models (Transformers/LLMs).

## Repository Structure

### 1. Domain Ontology
* `MeineOntologie.owl`: The semantic foundation defining core classes (`Character`, `Place`, `Weapon`) and explicit relations (`locatedIn`, `hasWeapon`, `relativeOf`) with strict domain and range constraints.

### 2. Synthetic Dataset
* `dataset_starwars.jsonl`: A synthetic dataset comprising 10 tiered-complexity Star Wars stories, generated via LLM. It includes the raw text and the expected "Ground Truth" triples.

### 3. Extraction & Evaluation Pipeline (Python)
* `pipeline_gliner_starwars.py`: The baseline extraction script utilizing `GLiNER (Medium-v2.1)` for zero-shot entity detection and heuristic rules for relation building.
* `evaluate_pipeline.py`: The benchmarking script that evaluates the pipeline against the synthetic dataset, calculating True Positives, False Positives, False Negatives, Precision, Recall, and F1-Score.
* `export_rdf.py`: The RDF exporter that converts the extracted Python triples into a machine-readable Knowledge Graph using `rdflib`.

### 4. Outputs & Reports
* `starwars_knowledge_graph.ttl`: The final generated Knowledge Graph in Turtle format, ready to be visualized in Protégé.
* `Benchmarking-Report_Enes-Tarik_Yalcin.pdf`: The comprehensive analysis of the baseline architecture, including an in-depth error analysis and proposed ML improvements.

## Installation & Setup
To run the scripts locally, ensure you have Python installed and install the required dependencies:

```bash
pip install gliner rdflib transformers torch datasets