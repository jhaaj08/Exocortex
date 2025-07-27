import re
from typing import List, Tuple
from collections import Counter

class TextProcessor:
    """Service class to clean and chunk extracted text"""
    
    def __init__(self):
        self.chunk_size = 350  # Target words per chunk
        self.overlap_size = 50  # Words to overlap between chunks
    
    def clean_text(self, raw_text: str) -> str:
        """Main method to clean extracted text"""
        if not raw_text:
            return ""
        
        # Step 1: Remove repeated headers/footers
        text = self.remove_repeated_elements(raw_text)
        
        # Step 2: Fix hyphen breaks
        text = self.fix_hyphen_breaks(text)
        
        # Step 3: Normalize whitespace and formatting
        text = self.normalize_formatting(text)
        
        return text.strip()
    
    def remove_repeated_elements(self, text: str) -> str:
        """Remove repeated headers, footers, and page numbers"""
        lines = text.split('\n')
        if len(lines) < 10:  # Too short to have meaningful repetition
            return text
        
        # Find lines that appear frequently (likely headers/footers)
        line_counts = Counter(line.strip() for line in lines if line.strip())
        
        # Consider lines repeated if they appear more than 3 times
        repeated_lines = {line for line, count in line_counts.items() 
                         if count > 3 and len(line) < 100}
        
        # Remove repeated lines, but keep if they're part of actual content
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            
            # Skip if it's a repeated line and looks like header/footer
            if (stripped_line in repeated_lines and 
                (self.looks_like_header_footer(stripped_line) or 
                 self.is_page_number(stripped_line))):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def looks_like_header_footer(self, line: str) -> bool:
        """Check if line looks like a header or footer"""
        line = line.strip().lower()
        
        # Common header/footer patterns
        header_footer_patterns = [
            r'^\d+$',  # Just a number
            r'^page \d+',  # Page X
            r'^\d+ of \d+$',  # X of Y
            r'^chapter \d+',  # Chapter X
            r'©.*copyright',  # Copyright notices
            r'^\d{4}-\d{2}-\d{2}$',  # Dates
            r'^confidential|^proprietary',  # Confidentiality notices
        ]
        
        for pattern in header_footer_patterns:
            if re.search(pattern, line):
                return True
        
        # Very short lines at start/end of pages
        if len(line) < 5:
            return True
            
        return False
    
    def is_page_number(self, line: str) -> bool:
        """Check if line is just a page number"""
        line = line.strip()
        return re.match(r'^\d{1,4}$', line) is not None
    
    def fix_hyphen_breaks(self, text: str) -> str:
        """Fix words broken across lines with hyphens"""
        # Pattern: word- followed by newline and lowercase letter
        # Example: "inter-\nesting" -> "interesting"
        
        # Fix simple hyphen breaks
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Fix hyphen breaks with potential whitespace
        text = re.sub(r'(\w)-\s*\n+\s*([a-z])', r'\1\2', text)
        
        return text
    
    def normalize_formatting(self, text: str) -> str:
        """Normalize whitespace and basic formatting"""
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)
        
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def chunk_text(self, cleaned_text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks of approximately chunk_size words
        Returns list of (chunk_text, start_word, end_word) tuples
        """
        if not cleaned_text:
            return []
        
        words = cleaned_text.split()
        if len(words) <= self.chunk_size:
            return [(cleaned_text, 0, len(words))]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            # Calculate end index
            end_idx = min(start_idx + self.chunk_size, len(words))
            
            # If not the last chunk, try to break at sentence boundary
            if end_idx < len(words):
                # Look for sentence ending in the last 50 words of the chunk
                search_start = max(end_idx - 50, start_idx + 100)  # Don't make chunks too small
                
                for i in range(end_idx - 1, search_start - 1, -1):
                    if words[i].endswith(('.', '!', '?')):
                        end_idx = i + 1
                        break
            
            # Extract chunk
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append((chunk_text, start_idx, end_idx))
            
            # Move to next chunk with overlap
            start_idx = max(end_idx - self.overlap_size, end_idx)
            
            # Prevent infinite loop
            if start_idx >= len(words):
                break
        
        return chunks
    
    def get_text_stats(self, text: str) -> dict:
        """Get statistics about the text"""
        if not text:
            return {'words': 0, 'characters': 0, 'lines': 0, 'paragraphs': 0}
        
        words = len(text.split())
        characters = len(text)
        lines = len(text.split('\n'))
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        
        return {
            'words': words,
            'characters': characters,
            'lines': lines,
            'paragraphs': paragraphs
        }

# Convenience function for easy import
def process_text(raw_text: str) -> Tuple[str, List[Tuple[str, int, int]], dict]:
    """
    Process raw text: clean and chunk
    Returns: (cleaned_text, chunks, stats)
    """
    processor = TextProcessor()
    cleaned_text = processor.clean_text(raw_text)
    chunks = processor.chunk_text(cleaned_text)
    stats = processor.get_text_stats(cleaned_text)
    
    return cleaned_text, chunks, stats 