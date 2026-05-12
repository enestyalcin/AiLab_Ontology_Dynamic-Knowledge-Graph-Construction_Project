from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, OWL

# 1. SETUP NAMESPACE & GRAPH

STARWARS = Namespace("http://www.example.org/starwars#")

# Create a new RDF Graph
kg = Graph()
kg.bind("sw", STARWARS)

# 2. DUMMY DATA (From our Pipeline output)

extracted_triples = [
    ("Luke Skywalker", "locatedIn", "Tatooine"),
    ("Luke Skywalker", "relativeOf", "Leia Organa"),
    ("Boba Fett", "hasWeapon", "EE-3 blaster")
]

# 3. BUILD THE KNOWLEDGE GRAPH
print("Building RDF Knowledge Graph...")

for subject_str, predicate_str, object_str in extracted_triples:
    # Format strings to valid URI formats (replace spaces with underscores)
    subj_uri = URIRef(STARWARS[subject_str.replace(" ", "_")])
    pred_uri = URIRef(STARWARS[predicate_str])
    
    # For objects, decide if it's another entity (URI) or a string value (Literal)
    if predicate_str in ["locatedIn", "relativeOf", "hasWeapon"]:
        obj_uri = URIRef(STARWARS[object_str.replace(" ", "_")])
        kg.add((subj_uri, pred_uri, obj_uri))
        
        # Explicitly declare them as Named Individuals (Instances) in OWL
        kg.add((subj_uri, RDF.type, OWL.NamedIndividual))
        kg.add((obj_uri, RDF.type, OWL.NamedIndividual))


# 4. EXPORT TO FILE
output_file = "starwars_knowledge_graph.ttl"
kg.serialize(destination=output_file, format="turtle")

print(f"Success! Exported {len(kg)} RDF statements to '{output_file}'")
print("This file can now be opened in Protégé or queried with SPARQL.")