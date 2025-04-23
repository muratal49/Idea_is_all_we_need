import requests
import json
import re
import nltk
import time
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
apikey = "YOUR API KEY"

def query_api(search_url, query, offset=0, limit=20):
    headers = {"Authorization": "Bearer " + apikey}
    url = f"{search_url}?q={query}&limit={limit}&offset={offset}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"⚠️ Error {response.status_code} for query='{query}', offset={offset}")
        return {"results": []}, 0

    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error at offset {offset} for query '{query}'")
        return {"results": []}, 0

    print(f"Query: {query} | Offset: {offset} | Results: {len(data.get('results', []))}")
    return data, response.elapsed.total_seconds()


def extract_sections(full_text):
    if not full_text:
        return {
            "conclusions": "Full text not available",
            "limitations": "Full text not available",
            "future_work": "Full text not available"
        }

    full_text = re.sub(r'\s+', ' ', full_text).strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\r\n\s*\r\n', full_text) if p.strip()]
    sections = []
    current_section = {"heading": "", "content": ""}
    for p in paragraphs:
        if len(p) < 100 and (p.isupper() or re.match(r'^\d+[\.\s]+\w+|^[IVX]+[\.\s]+\w+', p)):
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"heading": p, "content": ""}
        else:
            current_section["content"] += (" " + p) if current_section["content"] else p
    if current_section["content"]:
        sections.append(current_section)

    patterns = {
        "limitations": [],

        "future_work": [],

        "conclusions": [ ]
    }

    output = {"limitations": "", "future_work": "", "conclusions": ""}
    for section in sections:
        heading_lower = section["heading"].lower()
        for key, regex_list in patterns.items():
            if any(re.search(p, heading_lower) for p in regex_list):
                output[key] += f"Section: {section['heading']}\n{section['content']}\n\n"
    for key in output:
        if not output[key]:
            output[key] = f"No explicit {key.replace('_', ' ')} section found."
    return output

def extract_sections(full_text):
    if not full_text:
        return {
            "conclusions": "Full text not available",
            "limitations": "Full text not available",
            "future_work": "Full text not available"
        }

    full_text = re.sub(r'\s+', ' ', full_text).strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\r\n\s*\r\n', full_text) if p.strip()]
    sections = []
    current_section = {"heading": "", "content": ""}
    for p in paragraphs:
        if len(p) < 100 and (p.isupper() or re.match(r'^\d+[\.\s]+\w+|^[IVX]+[\.\s]+\w+', p)):
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"heading": p, "content": ""}
        else:
            current_section["content"] += (" " + p) if current_section["content"] else p
    if current_section["content"]:
        sections.append(current_section)

    patterns = {
        "limitations": [r'\b(?:limitation|shortcoming|drawback|weakness|constraint)s?\b',
    r'\bcurrent\s+(?:limitation|constraint|shortcoming)s?\b',
    r'\b(?:limitation|shortcoming|drawback|weakness|constraint)s?\s+of\s+(?:the|this|our)\s+(?:study|approach|method|work|research|analysis|model|system|framework)\b',
    r'\blimitations\s+of\s+(?:our|the)\s+(?:method|model|work|approach|research)\b',
    r'\blimiting\s+factor[s]?\b',
    r'\bsources?\s+of\s+error\b',
    r'\bdifficult[y|ies]\s+in\s+(?:implementing|applying|scaling)\b'],  

        "future_work": [r'\bfuture\s+(?:work|research|direction|study|investigation|exploration|development|improvement|enhancement)s?\b',
    r'\bfurther\s+(?:work|research|study|investigation|development|improvement)s?\b',
    r'\bfuture\s+(?:scope|perspective|outlook|avenue|plan|goal|opportunity|possibility)\b',
    r'\bopen\s+(?:question|issue|challenge|problem|area)s?\b',
    r'\bnext\s+step[s]?\b',
    r'\bcan\s+be\s+(?:extended|explored|investigated)\b',
    r'\bwe\s+plan\s+to\b',
    r'\bwe\s+aim\s+to\b',
    r'\bongoing\s+(?:research|study|investigation)\b'], 

        "conclusions": [r'\bconclusion[s]?\b',
    r'\bconcluding\s+remarks\b',
    r'\bsummary\s+and\s+conclusion[s]?\b',
    r'\bto\s+conclude\b',
    r'\bin\s+conclusion\b',
    r'\bwe\s+conclude\s+that\b',
    r'\bthis\s+study\s+(?:shows|demonstrates|confirms|indicates)\b',
    r'\bthe\s+results\s+suggest\b']   
    }

    output = {"limitations": "", "future_work": "", "conclusions": ""}

    # Pass 1: Section heading detection
    for section in sections:
        heading_lower = section["heading"].lower()
        for key, regex_list in patterns.items():
            if any(re.search(p, heading_lower) for p in regex_list):
                output[key] += f"Section: {section['heading']}\n{section['content']}\n\n"

    # Pass 2: Fallback to inline sentence matches (if still empty)
    for key, regex_list in patterns.items():
        if not output[key]:
            matched_sentences = []
            for section in sections:
                sentences = sent_tokenize(section["content"])
                for i, sentence in enumerate(sentences):
                    if any(re.search(p, sentence.lower()) for p in regex_list):
                        # Get sentence ± 1 context
                        context = sentences[max(0, i - 1):min(len(sentences), i + 2)]
                        matched_sentences.append(" ".join(context))
            if matched_sentences:
                output[key] = f"Auto-extracted mentions:\n" + "\n".join(matched_sentences)

    # Final fallback message
    for key in output:
        if not output[key]:
            output[key] = f"No {key.replace('_', ' ')} content found"

    return output


def main():
    search_url = "https://api.core.ac.uk/v3/search/works"
    topic_queries = [
    # 🔬 General ML/AI
    "artificial intelligence", "machine learning", "deep learning", "data science", "AI applications",
    
    # 🧠 NLP / Language
    "natural language processing", "language models", "NLP", "text mining", "information extraction",
    
    # 🔍 Vision & Multimodal
    "computer vision", "image recognition", "object detection", "vision transformers",
    
    # 🧬 Health / Bio / Clinical
    "biomedical informatics", "health informatics", "clinical AI", "medical imaging", "EHR", "genomics",
    
    # 🏛️ Society, ethics, education
    "AI ethics", "explainable AI", "fairness in machine learning", "AI in education", "social computing",
    
    # 📊 Classic ML / Theory
    "support vector machines", "random forests", "decision trees", "unsupervised learning", "feature selection",
    
    # 🛠️ Engineering / Systems
    "AI systems", "distributed learning", "edge AI", "federated learning", "hardware-aware ML"
]


    max_papers = 2000
    limit = 20
    all_papers = []
    seen_ids = set()


    # Inside your topic_queries loop
    for query in topic_queries:
        offset = 0
        while len(all_papers) < max_papers:
            data, _ = query_api(search_url, query, offset=offset, limit=limit)
            results = data.get("results", [])
            if not results:
                break

            for paper in results:
                if paper.get("fullText") and paper.get("id") not in seen_ids:
                    seen_ids.add(paper["id"])
                    all_papers.append(paper)

            offset += limit
            time.sleep(2)  # wait 1 second before next offset page

        # ✅ Add a pause between queries too
        time.sleep(5)


        if len(all_papers) >= max_papers:
                break

    print(f"\n✅ Total collected papers with full text: {len(all_papers)}")

    with open("core_fulltext_dataset_filtered.jsonl", "a", encoding="utf-8") as f:
        for paper in all_papers:
            full_text = paper.get("fullText", "")
            abstract = paper.get("abstract", "No abstract available")
            sections = extract_sections(full_text)

            # Filter: must have at least one of the three key sections
            if all(sections[k].startswith("No ") for k in ["conclusions", "future_work", "limitations"]):
                continue


            record = {
                "abstract": abstract,
                "conclusions": sections["conclusions"],
                "limitations": sections["limitations"],
                "future_work": sections["future_work"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    print("💾 Saved to core_fulltext_dataset.jsonl")

if __name__ == "__main__":
    main()




# response = requests.get("https://api.core.ac.uk/v3/search/works?q=covid&limit=5", headers={"Authorization": "Bearer kgUzxD4K2Z3QjIp6wT1qJBMbVCAfvahn"})
# print(response.status_code)
# print(response.text[:200])
