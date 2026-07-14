import os
import re

RESEARCH_DIR = r"."
PY_FILES = [
    "complexity_compression.py",
    "datacleaning.py",
    "docxtotxt.py",
    "hallvsinnovation.py",
    "ideologicalbias.py",
    "pdftotxt.py",
    "sarvamdata.py",
    "sarvamnatdata.py",
    "segPDF.py",
    "semantic.py",
]

def clean_comments(content):
    lines = content.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Only process standalone full-line comments and keeping data/method comments heuristically
        if stripped.startswith('#'):
            lower_line = stripped.lower()
            # keep if it mentions data or method/def
            if 'data' in lower_line or 'method' in lower_line or 'def ' in lower_line or 'function' in lower_line or 'argument' in lower_line or 'param' in lower_line or 'return' in lower_line:
                cleaned_lines.append(line)
                continue
                
            # Remove purely decorative or generic section headers
            if re.match(r'^#\s*[-=~_]+$', stripped) or re.match(r'^#\s*\d+\.\s*', stripped):
                continue
            
            # Remove generic structural comments
            if re.match(r'^#\s*(imports?|setup|main|configuration|helper|load|compute|visualization|script|run|execution)\b', lower_line):
                continue
                
            # If it's a normal explanatory comment, we might normally keep it, but user specifically asked to remove ALL unnecessary, keeping ONLY data/method.
            # To be aggressive as requested:
            continue
            
        cleaned_lines.append(line)
        
    # Remove consecutive blank lines
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned_lines))
    return result

def main():
    os.chdir(RESEARCH_DIR)
    for py_file in PY_FILES:
        filepath = os.path.join(RESEARCH_DIR, py_file)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()
            
        cleaned = clean_comments(original)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"Cleaned {py_file}")

if __name__ == "__main__":
    main()
