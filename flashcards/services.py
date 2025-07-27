import os
import re
import markdown
from typing import List, Dict, Any
from .models import Folder

class MarkdownProcessor:
    """Service for processing markdown files and extracting content for flashcard generation"""
    
    def __init__(self, folder: Folder):
        self.folder = folder
        self.markdown_parser = markdown.Markdown(extensions=['codehilite', 'fenced_code'])
    
    def get_markdown_files(self) -> List[str]:
        """Get list of markdown files in the folder and all subfolders"""
        markdown_files = []
        try:
            # Walk through all directories and subdirectories
            for root, dirs, files in os.walk(self.folder.path):
                for file in files:
                    if file.lower().endswith(('.md', '.markdown')):
                        # Get relative path from the base folder
                        relative_path = os.path.relpath(os.path.join(root, file), self.folder.path)
                        markdown_files.append(relative_path)
        except OSError as e:
            print(f"Error reading folder {self.folder.path}: {e}")
        return sorted(markdown_files)
    
    def read_file_content(self, filename: str) -> str:
        """Read content from a markdown file"""
        filepath = os.path.join(self.folder.path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Error reading file {filepath}: {e}")
            return ""
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract logical sections from markdown content for flashcard generation"""
        sections = []
        
        # Split content by main headings (# and ##)
        heading_pattern = re.compile(r'^(#{1,2})\s+(.+)$', re.MULTILINE)
        parts = heading_pattern.split(content)
        
        current_section = None
        
        for i in range(0, len(parts), 3):
            if i + 2 < len(parts):
                heading_level = parts[i + 1]
                heading_text = parts[i + 2].strip()
                section_content = parts[i + 3] if i + 3 < len(parts) else ""
                
                # Clean up the content
                section_content = self._clean_content(section_content)
                
                if section_content.strip():
                    section = {
                        'heading': heading_text,
                        'level': len(heading_level),
                        'content': section_content,
                        'type': self._classify_content_type(section_content)
                    }
                    sections.append(section)
            else:
                # Handle content without headings
                remaining_content = parts[i].strip()
                if remaining_content:
                    section = {
                        'heading': 'Introduction',
                        'level': 1,
                        'content': self._clean_content(remaining_content),
                        'type': self._classify_content_type(remaining_content)
                    }
                    sections.append(section)
        
        return sections
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize markdown content"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Normalize code blocks
        content = re.sub(r'```(\w+)?\n', '```\n', content)
        
        return content.strip()
    
    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content to help with flashcard generation"""
        content_lower = content.lower()
        
        # Check for code blocks
        if '```' in content or '`' in content:
            return 'code'
        
        # Check for lists
        if re.search(r'^\s*[-*+]\s', content, re.MULTILINE) or re.search(r'^\s*\d+\.\s', content, re.MULTILINE):
            return 'list'
        
        # Check for definitions (lines with : or --)
        if re.search(r'^\s*\*\*[^*]+\*\*:\s', content, re.MULTILINE) or '--' in content:
            return 'definition'
        
        # Check for examples
        if any(word in content_lower for word in ['example', 'for instance', 'such as', 'like']):
            return 'example'
        
        # Check for procedures/steps
        if any(word in content_lower for word in ['step', 'first', 'then', 'next', 'finally']):
            return 'procedure'
        
        return 'concept'
    
    def extract_key_concepts(self, section: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key concepts from a section that would make good flashcards"""
        concepts = []
        content = section['content']
        content_type = section['type']
        
        if content_type == 'definition':
            # Extract bold terms and their definitions
            definition_pattern = re.compile(r'\*\*([^*]+)\*\*:\s*([^\n]+)')
            for match in definition_pattern.finditer(content):
                concepts.append({
                    'question': f"What is {match.group(1)}?",
                    'answer': match.group(2).strip(),
                    'context': section['heading']
                })
        
        elif content_type == 'list':
            # Create flashcards for list items
            list_items = re.findall(r'^\s*[-*+]\s*(.+)$', content, re.MULTILINE)
            if len(list_items) > 1:
                concepts.append({
                    'question': f"List the main points about {section['heading']}",
                    'answer': '\n'.join([f"• {item.strip()}" for item in list_items]),
                    'context': section['heading']
                })
        
        elif content_type == 'code':
            # Extract code examples
            code_blocks = re.findall(r'```[^\n]*\n(.*?)```', content, re.DOTALL)
            for i, code in enumerate(code_blocks):
                concepts.append({
                    'question': f"How do you implement {section['heading']}? (Code example)",
                    'answer': f"```\n{code.strip()}\n```",
                    'context': section['heading']
                })
        
        elif content_type == 'concept':
            # Create a general concept question
            # Remove markdown formatting for cleaner content
            clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            clean_content = re.sub(r'`([^`]+)`', r'\1', clean_content)
            
            if len(clean_content.strip()) > 50:  # Only create if substantial content
                concepts.append({
                    'question': f"Explain {section['heading']}",
                    'answer': clean_content.strip()[:500] + ('...' if len(clean_content) > 500 else ''),
                    'context': section['heading']
                })
        
        return concepts
    
    def process_all_files(self) -> List[Dict[str, Any]]:
        """Process all markdown files and extract content for flashcard generation"""
        all_content = []
        
        markdown_files = self.get_markdown_files()
        
        for filename in markdown_files:
            print(f"Processing file: {filename}")
            
            # Read file content
            content = self.read_file_content(filename)
            if not content:
                continue
            
            # Extract sections
            sections = self.extract_sections(content)
            
            # Extract concepts from each section
            for section in sections:
                concepts = self.extract_key_concepts(section)
                
                for concept in concepts:
                    all_content.append({
                        'source_file': filename,
                        'question': concept['question'],
                        'answer': concept['answer'],
                        'context': concept['context'],
                        'original_content': section['content'][:200] + ('...' if len(section['content']) > 200 else ''),
                        'content_type': section['type']
                    })
        
        return all_content

def process_folder_content(folder: Folder) -> List[Dict[str, Any]]:
    """Main function to process a folder and extract flashcard content"""
    processor = MarkdownProcessor(folder)
    return processor.process_all_files() 