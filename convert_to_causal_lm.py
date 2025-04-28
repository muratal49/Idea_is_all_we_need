import json

input_path = "/Users/snigdhashrivastav/Desktop/NLP/project/core_fulltext_dataset_filtered_healthcareAI.jsonl"
output_path = "/Users/snigdhashrivastav/Desktop/NLP/project/causal_lm_dataset_healthcareAI.jsonl"

def safe_strip(val):
    return val.strip() if isinstance(val, str) else ""

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    line_count = 0
    kept_count = 0

    for line in fin:
        line_count += 1
        try:
            paper = json.loads(line)
        except json.JSONDecodeError:
            print(f"❌ Skipping malformed JSON at line {line_count}")
            continue

        # Extract sections safely
        sections = {
            "Abstract": paper.get("abstract"),
            "Conclusions": paper.get("conclusions"),
            "Limitations": paper.get("limitations"),
            "Future Work": paper.get("future_work")
        }

        combined = ""
        for section_name, section_text in sections.items():
            if not isinstance(section_text, str):
                continue
            section_text = section_text.strip()
            if section_text.lower().startswith("no ") or len(section_text) < 30:
                continue
            section_text = section_text.replace("Auto-extracted mentions:", "").strip()
            combined += f"{section_name}:\n{section_text}\n\n"

        # Skip this sample if everything got filtered out
        if not combined.strip():
            continue

        combined += "### Suggest two research ideas:"
        fout.write(json.dumps({"text": combined}, ensure_ascii=False) + "\n")
        kept_count += 1
        
print(f"✅ Done. Kept {kept_count} papers out of {line_count} lines.")
print(f"📁 Output saved to: {output_path}")
