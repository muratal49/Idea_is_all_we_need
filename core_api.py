import requests
import json
import re
import nltk
from nltk.tokenize import sent_tokenize

# You might need to download nltk resources if running for first time
nltk.download('punkt')

# API key - leave blank for you to fill in
apikey = "kgUzxD4K2Z3QjIp6wT1qJBMbVCAfvahn"

def query_api(search_url, query, scrollId=None):
    headers = {"Authorization": "Bearer " + apikey}
    
    # Check if the URL already has query parameters
    separator = "&" if "?" in search_url else "?"
    
    if not scrollId:
        response = requests.get(f"{search_url}{separator}q={query}&limit=10&scroll=true", headers=headers)
    else:
        response = requests.get(f"{search_url}{separator}q={query}&limit=10&scrollId={scrollId}", headers=headers)        
    return response.json(), response.elapsed.total_seconds()

def extract_sections(full_text):
    """
    Extract future work and limitations sections from the full text.
    
    Args:
        full_text (str): The full text of a research paper
        
    Returns:
        dict: Dictionary containing extracted limitations and future work sections
    """
    if not full_text:
        return {"limitations": "Full text not available", "future_work": "Full text not available"}
    
    # Preprocess text: normalize whitespace and split into paragraphs
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\r\n\s*\r\n', full_text) if p.strip()]
    
    # Combine paragraphs into potential sections
    sections = []
    current_section = {"heading": "", "content": ""}
    
    for p in paragraphs:
        # Check if paragraph looks like a section heading
        if len(p) < 100 and (p.isupper() or re.match(r'^\d+[\.\s]+\w+|^[IVX]+[\.\s]+\w+', p)):
            # If we already have content in the current section, save it
            if current_section["content"]:
                sections.append(current_section)
            # Start a new section
            current_section = {"heading": p, "content": ""}
        else:
            # Add to current section's content
            if current_section["content"]:
                current_section["content"] += " " + p
            else:
                current_section["content"] = p
    
    # Add the last section if it has content
    if current_section["content"]:
        sections.append(current_section)
    
    # Now look for relevant sections
    limitation_sections = []
    future_work_sections = []
    
    # Define more comprehensive patterns
    limitation_patterns = [
        r'\b(?:limitation|shortcoming|drawback|weakness|constraint)s?\b',
        r'\bcurrent\s+(?:limitation|constraint|shortcoming)s?\b',
        r'\b(?:limitation|shortcoming|drawback|weakness|constraint)s?\s+of\s+(?:the|this|our)\s+(?:study|approach|method|work|research|analysis|model|system|framework)\b'
    ]
    
    future_patterns = [
        r'\bfuture\s+(?:work|research|direction|study|investigation|exploration|development|improvement|enhancement)\b',
        r'\bfurther\s+(?:work|research|study|investigation|development|improvement)\b',
        r'\bfuture\s+(?:scope|perspective|outlook|avenue|plan|goal|opportunity|possibility)\b',
        r'\bopen\s+(?:question|issue|challenge|problem|area)\b',
        r'\bnext\s+step'
    ]
    
    # Find sections with these patterns
    for section in sections:
        heading = section["heading"].lower()
        content = section["content"]
        
        # Check for limitations in heading
        if any(re.search(pattern, heading) for pattern in limitation_patterns):
            limitation_sections.append({"heading": section["heading"], "content": content})
        
        # Check for future work in heading
        elif any(re.search(pattern, heading) for pattern in future_patterns):
            future_work_sections.append({"heading": section["heading"], "content": content})
    
    # If we didn't find dedicated sections, try to extract paragraphs discussing these topics
    if not limitation_sections:
        for section in sections:
            content = section["content"].lower()
            sentences = sent_tokenize(section["content"])
            
            limitation_content = []
            for i, sentence in enumerate(sentences):
                if any(re.search(pattern, sentence.lower()) for pattern in limitation_patterns):
                    # Capture context: the sentence plus surrounding sentences
                    start = max(0, i-1)
                    end = min(len(sentences), i+2)
                    limitation_content.append(" ".join(sentences[start:end]))
            
            if limitation_content:
                limitation_sections.append({
                    "heading": section["heading"],
                    "content": "\n".join(limitation_content)
                })
    
    if not future_work_sections:
        for section in sections:
            content = section["content"].lower()
            sentences = sent_tokenize(section["content"])
            
            future_content = []
            for i, sentence in enumerate(sentences):
                if any(re.search(pattern, sentence.lower()) for pattern in future_patterns):
                    # Capture context: the sentence plus surrounding sentences
                    start = max(0, i-1)
                    end = min(len(sentences), i+2)
                    future_content.append(" ".join(sentences[start:end]))
            
            if future_content:
                future_work_sections.append({
                    "heading": section["heading"],
                    "content": "\n".join(future_content)
                })
    
    # Prepare return values
    limitation_text = ""
    if limitation_sections:
        for section in limitation_sections:
            limitation_text += f"Section: {section['heading']}\n{section['content']}\n\n"
    else:
        limitation_text = "No explicit limitations section found"
    
    future_text = ""
    if future_work_sections:
        for section in future_work_sections:
            future_text += f"Section: {section['heading']}\n{section['content']}\n\n"
    else:
        future_text = "No explicit future work section found"
    
    return {
        "limitations": limitation_text.strip(),
        "future_work": future_text.strip()
    }

def main():
    # CORE API base URL
    search_url = "https://api.core.ac.uk/v3/search/works"
    
    # Example query - you can change this
    query = "healthcare"
    
    # You can search specifically for papers with full text available
    # query = 'fullText:"limitations" AND fullText:"future work"'
    
    print(f"Searching for: {query}")
    
    # Make initial request
    response_data, elapsed_time = query_api(search_url, query)
    
    # Print response information
    print(f"Request took {elapsed_time} seconds")
    print(f"Total results found: {response_data.get('totalHits', 0)}")
    
    # Process only papers with full text
    papers_with_fulltext = [paper for paper in response_data.get('results', []) if paper.get('fullText')]
    
    print(f"Found {len(papers_with_fulltext)} papers with full text")
    
    # Display results in a readable format
    if papers_with_fulltext:
        print("\n--- PAPERS WITH EXTRACTED SECTIONS ---")
        for i, paper in enumerate(papers_with_fulltext[:5], 1):
            print(f"\n{'='*50}")
            print(f"Paper {i}:")
            print(f"Title: {paper.get('title', 'No title')}")
            print(f"Authors: {', '.join([author.get('name', 'Unknown') for author in paper.get('authors', [])])}")
            print(f"Published: {paper.get('publishedDate', 'Unknown date')}")
            
            # Print abstract
            abstract = paper.get('abstract', 'No abstract available')
            print(f"\nABSTRACT:\n{abstract[:500] + '...' if len(abstract) > 500 else abstract}")
        
            # Extract sections from full text
            full_text = paper.get('fullText', '')
            sections = extract_sections(full_text)
            
            print("\nLIMITATIONS:")
            print(sections["limitations"][:500] + "..." if len(sections["limitations"]) > 500 else sections["limitations"])
            
            print("\nFUTURE WORK:")
            print(sections["future_work"][:500] + "..." if len(sections["future_work"]) > 500 else sections["future_work"])
            
            print(f"\nURL: {paper.get('downloadUrl', 'No URL available')}")
    else:
        print("No papers with full text found.")

if __name__ == "__main__":
    main()