import io
import pandas as pd
from docx import Document
from typing import List, Tuple

def parse_single_story(text: str) -> List[str]:
    clean = text.strip()
    return [clean] if clean else []

def parse_multiple_stories(text: str) -> List[str]:
    blocks = [block.strip() for block in text.split('\n\n')]
    return [block for block in blocks if block]

def parse_csv_upload(file_bytes: bytes) -> List[str]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    target_col = None
    for col in df.columns:
        if col.lower().strip() in ['storytext', 'story', 'text']:
            target_col = col
            break
    if not target_col and not df.empty:
        target_col = df.columns[0]
        
    if not target_col:
        return []
        
    stories = df[target_col].dropna().astype(str).str.strip().tolist()
    return [s for s in stories if s]

def parse_txt_upload(file_bytes: bytes) -> List[str]:
    text = file_bytes.decode('utf-8')
    return parse_multiple_stories(text)

def parse_docx_upload(file_bytes: bytes) -> List[str]:
    doc = Document(io.BytesIO(file_bytes))
    stories = [p.text.strip() for p in doc.paragraphs]
    return [s for s in stories if s]

def validate_stories(stories: List[str], max_count: int = 50, 
                     min_len: int = 10, max_len: int = 2000) -> Tuple[List[str], List[str]]:
    valid_stories = []
    warnings = []
    
    for i, s in enumerate(stories):
        if len(s) < min_len:
            warnings.append(f"Story {i+1} dropped: shorter than {min_len} characters.")
        elif len(s) > max_len:
            warnings.append(f"Story {i+1} dropped: longer than {max_len} characters.")
        else:
            valid_stories.append(s)
            
    if len(valid_stories) > max_count:
        dropped = len(valid_stories) - max_count
        warnings.append(f"Batch exceeded {max_count} limit. {dropped} stories truncated.")
        valid_stories = valid_stories[:max_count]
        
    return valid_stories, warnings
