import json
import os

# === INPUT SETTINGS ===
input_jsonl_path = "/Users/snigdhashrivastav/Desktop/NLP/project/code/causal_lm_dataset_healthcareAI.jsonl"  
output_jsonl_path = "/Users/snigdhashrivastav/Desktop/NLP/project/code/causal_lm_dataset_healthcareAI_instruction.jsonl" 

# === DUMMY PLACEHOLDER FOR RESPONSE ===
DUMMY_RESPONSE = "A possible new research idea is to explore novel approaches based on the given limitations and future work."

# === PROCESSING ===

# Read original jsonl
entries = []
with open(input_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        entries.append(obj)

# Convert into Instruction-Input-Response style
instructional_entries = []
for entry in entries:
    paper_text = entry.get("text", "")

    new_entry = {
        "text": f"### Instruction:\nGiven the following paper sections, generate a new research idea.\n\n### Input:\n{paper_text}\n\n### Response:\n{DUMMY_RESPONSE}"
    }
    instructional_entries.append(new_entry)

# Write new jsonl
with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    for item in instructional_entries:
        f.write(json.dumps(item) + "\n")

print(f"\u2705 Successfully created instruction-style dataset with {len(instructional_entries)} entries!")
print(f"Saved to: {output_jsonl_path}")
