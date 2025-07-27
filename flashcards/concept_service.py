import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI  # ✅ Correct import
from django.conf import settings
from .models import PDFDocument, TextChunk, ChunkLabel, ConceptUnit
from .reading_time_service import optimize_reading_time

# ✅ Set up detailed logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ConceptAnalyzer:
    """Service class to analyze text chunks and create concept units using LLM"""
    
    def __init__(self):
        # Initialize OpenAI client with defensive error handling
        api_key = os.getenv('OPENAI_API_KEY')
        
        print(f"🔍 DEBUG: API key exists: {bool(api_key)}")
        
        if not api_key or api_key == 'your_openai_api_key_here':
            self.client = None
            self.api_available = False
            print("❌ DEBUG: OpenAI API not available")
        else:
            try:
                # Create client with minimal parameters to avoid proxy issues
                self.client = OpenAI(api_key=api_key)
                self.api_available = True
                print("✅ DEBUG: OpenAI client initialized successfully")
            except TypeError as e:
                if 'proxies' in str(e):
                    print(f"🔧 DEBUG: Proxy parameter issue detected: {e}")
                    print("🔧 DEBUG: Trying alternative initialization...")
                    try:
                        # Alternative initialization without any optional parameters
                        import openai
                        openai.api_key = api_key
                        self.client = openai
                        self.api_available = True
                        print("✅ DEBUG: OpenAI legacy client initialized")
                    except Exception as e2:
                        print(f"❌ DEBUG: All OpenAI initialization methods failed: {e2}")
                        self.client = None
                        self.api_available = False
                else:
                    print(f"❌ DEBUG: OpenAI client initialization failed: {e}")
                    self.client = None
                    self.api_available = False
            except Exception as e:
                print(f"❌ DEBUG: Unexpected OpenAI error: {e}")
                self.client = None
                self.api_available = False
        
        self.model = "gpt-3.5-turbo"
        
        self.concept_labels = [
            'Definition', 'Intuition', 'Example', 'Procedure', 'Algorithm',
            'Derivation', 'Proof', 'Statement', 'Theorem', 'Lemma',
            'Assumption', 'Property', 'Discussion', 'Contrast', 'Warning',
            'NewTopic', 'Recap', 'WrapUp', 'DataSpec', 'Code', 'Pseudocode',
            'FigureCaption', 'Exercise', 'ProblemStatement', 'Introduction',
            'Conclusion', 'Summary', 'Background', 'Methodology', 'Results',
            'Analysis', 'Interpretation', 'Application', 'Limitation',
            'FutureWork', 'Related', 'Comparison', 'Classification',
            'Explanation', 'Elaboration', 'Other'
        ]
    
    def analyze_pdf_concepts(self, pdf_document: PDFDocument) -> Tuple[bool, str]:
        """
        Main method to analyze all chunks in a PDF and create concept units
        Returns: (success, error_message)
        """
        print(f"🚀 DEBUG: Starting concept analysis for PDF {pdf_document.id}")  # ✅ Debug print
        
        if not self.api_available:
            error_msg = "OpenAI API key not configured. Set OPENAI_API_KEY in your .env file"
            print(f"❌ DEBUG: {error_msg}")  # ✅ Debug print
            return False, error_msg
        
        try:
            # Step 1: Get all text chunks for this PDF
            chunks = pdf_document.text_chunks.all().order_by('chunk_order')
            print(f"📊 DEBUG: Found {chunks.count()} chunks to analyze")  # ✅ Debug print
            
            if not chunks:
                return False, "No text chunks found for this PDF"
            
            # Step 2: Analyze each chunk and assign labels
            chunk_labels = []
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks, 1):
                print(f"🧠 DEBUG: Analyzing chunk {i}/{chunks.count()} (ID: {chunk.id})")  # ✅ Debug print
                
                success, label_data = self.analyze_chunk(chunk)
                if success:
                    chunk_labels.append(label_data)
                    print(f"✅ DEBUG: Chunk {i} labeled as '{label_data.get('primary_label', 'Unknown')}'")  # ✅ Debug print
                else:
                    failed_chunks += 1
                    print(f"❌ DEBUG: Chunk {i} analysis failed: {label_data}")  # ✅ Debug print
                    logger.warning(f"Failed to analyze chunk {chunk.id}: {label_data}")
            
            print(f"📈 DEBUG: Analysis complete. Success: {len(chunk_labels)}, Failed: {failed_chunks}")  # ✅ Debug print
            
            # ✅ Strict validation
            if not chunk_labels:
                error_msg = f"LLM labeling failed for all {len(chunks)} chunks. Cannot create concept units without labels."
                print(f"❌ DEBUG: {error_msg}")  # ✅ Debug print
                return False, error_msg
            
            # Step 3: Create ChunkLabel objects
            print(f"💾 DEBUG: Creating chunk labels in database...")  # ✅ Debug print
            self.create_chunk_labels(chunks, chunk_labels)
            
            # Verify labels were created
            created_labels = ChunkLabel.objects.filter(text_chunk__pdf_document=pdf_document).count()
            print(f"💾 DEBUG: Created {created_labels} chunk labels in database")  # ✅ Debug print
            
            if created_labels == 0:
                return False, "Failed to create chunk labels in database"
            
            # Step 4: Group chunks into concept units
            print(f"🔗 DEBUG: Grouping chunks into concept units...")  # ✅ Debug print
            concept_units = self.group_chunks_into_concepts(pdf_document)
            print(f"🔗 DEBUG: Created {len(concept_units)} concept units")  # ✅ Debug print
            
            if not concept_units:
                return False, "Failed to create concept units from labeled chunks"
            
            # Step 5: Optimize concept units by reading time ✅
            print(f"⏱️ DEBUG: Starting time optimization step...")  # ✅ Debug print
            
            try:
                from .reading_time_service import optimize_reading_time
                time_success, time_message = optimize_reading_time(pdf_document)
                print(f"⏱️ DEBUG: Time optimization result: success={time_success}, message='{time_message}'")  # ✅ Debug print
            except Exception as e:
                time_success = False
                time_message = f"Time optimization failed: {str(e)}"
                print(f"💥 DEBUG: Time optimization exception: {time_message}")  # ✅ Debug print
            
            # Step 6: Update PDF document status
            pdf_document.concepts_analyzed = True
            pdf_document.save()
            
            success_message = f"Successfully created {len(concept_units)} concept units"
            if time_success:
                success_message += f". {time_message}"
            else:
                success_message += f". Warning: Time optimization failed - {time_message}"
            
            return True, success_message
            
        except Exception as e:
            error_msg = f"Error in concept analysis: {str(e)}"
            print(f"💥 DEBUG: Exception occurred: {error_msg}")  # ✅ Debug print
            logger.error(error_msg)
            return False, error_msg
    
    def analyze_chunk(self, chunk: TextChunk) -> Tuple[bool, Dict]:
        """
        Analyze a single chunk and assign concept labels using LLM
        Returns: (success, label_data)
        """
        print(f"🤖 DEBUG: Making OpenAI API call for chunk {chunk.id}")  # ✅ Debug print
        
        try:
            prompt = self.create_chunk_analysis_prompt(chunk.chunk_text)
            print(f"📝 DEBUG: Prompt length: {len(prompt)} characters")  # ✅ Debug print
            
            # Test API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert academic content analyzer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            print(f"✅ DEBUG: OpenAI API call successful")  # ✅ Debug print
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            print(f"📋 DEBUG: LLM Response: {response_text[:200]}...")  # ✅ Debug print (first 200 chars)
            
            label_data = self.parse_chunk_analysis_response(response_text)
            print(f"🏷️ DEBUG: Parsed label: {label_data.get('primary_label', 'Unknown')}")  # ✅ Debug print
            
            return True, label_data
            
        except Exception as e:
            error_msg = f"Error analyzing chunk {chunk.id}: {str(e)}"
            print(f"💥 DEBUG: API call failed: {error_msg}")  # ✅ Debug print
            logger.error(error_msg)
            return False, error_msg
    
    def create_chunk_analysis_prompt(self, chunk_text: str) -> str:
        """Create prompt for analyzing a text chunk"""
        labels_str = ", ".join(self.concept_labels)
        
        prompt = f"""
Analyze the following text chunk and classify it according to its primary conceptual purpose.

Available labels: {labels_str}

Text chunk:
---
{chunk_text[:1500]}  # Truncate very long chunks
---

Please respond with a JSON object containing:
1. "primary_label": The single most appropriate label from the list above
2. "secondary_labels": Array of 0-3 additional relevant labels
3. "confidence": Float between 0-1 indicating your confidence
4. "reasoning": Brief explanation for your choice
5. "keywords": Array of 3-5 key concepts/terms from this chunk
6. "concept_topic": Brief description of the main topic/concept

Example response:
{{
    "primary_label": "Definition",
    "secondary_labels": ["Example"],
    "confidence": 0.9,
    "reasoning": "This chunk provides a clear definition followed by an illustrative example",
    "keywords": ["machine learning", "algorithm", "training data"],
    "concept_topic": "Machine Learning Definition"
}}

Respond only with the JSON object.
"""
        return prompt
    
    def parse_chunk_analysis_response(self, response_text: str) -> Dict:
        """Parse LLM response for chunk analysis"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['primary_label', 'confidence', 'reasoning', 'keywords', 'concept_topic']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate primary label
            if data['primary_label'] not in self.concept_labels:
                data['primary_label'] = 'Other'
            
            # Ensure secondary_labels is a list
            if 'secondary_labels' not in data:
                data['secondary_labels'] = []
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing chunk analysis response: {str(e)}")
            # Return default response
            return {
                'primary_label': 'Other',
                'secondary_labels': [],
                'confidence': 0.5,
                'reasoning': 'Failed to parse LLM response',
                'keywords': [],
                'concept_topic': 'Unknown'
            }
    
    def create_chunk_labels(self, chunks: List[TextChunk], chunk_labels: List[Dict]):
        """Create ChunkLabel objects from analysis results"""
        # Delete existing chunk labels
        ChunkLabel.objects.filter(text_chunk__in=chunks).delete()
        
        # Create new chunk labels
        label_objects = []
        for chunk, label_data in zip(chunks, chunk_labels):
            chunk_label = ChunkLabel(
                text_chunk=chunk,
                primary_label=label_data['primary_label'],
                secondary_labels=label_data.get('secondary_labels', []),
                confidence_score=label_data['confidence'],
                reasoning=label_data['reasoning'],
                concept_keywords=label_data['keywords']
            )
            label_objects.append(chunk_label)
        
        # Bulk create for efficiency
        ChunkLabel.objects.bulk_create(label_objects)
    
    def group_chunks_into_concepts(self, pdf_document: PDFDocument) -> List[ConceptUnit]:
        """
        Group related chunks into concept units based on their labels and content
        """
        # Delete existing concept units
        pdf_document.concept_units.all().delete()
        
        # Get all chunk labels for this PDF
        chunk_labels = ChunkLabel.objects.filter(
            text_chunk__pdf_document=pdf_document
        ).select_related('text_chunk').order_by('text_chunk__chunk_order')
        
        if not chunk_labels:
            return []
        
        # Group chunks using LLM-assisted analysis
        concept_groups = self.create_concept_groups(list(chunk_labels))
        
        # Create ConceptUnit objects
        concept_units = []
        for order, group in enumerate(concept_groups):
            concept_unit = self.create_concept_unit(pdf_document, group, order + 1)
            concept_units.append(concept_unit)
        
        return concept_units
    
    def create_concept_groups(self, chunk_labels: List[ChunkLabel]) -> List[List[ChunkLabel]]:
        """
        Group chunk labels into concept units based on semantic similarity and transitions
        """
        if not chunk_labels:
            return []
        
        groups = []
        current_group = [chunk_labels[0]]
        
        for i in range(1, len(chunk_labels)):
            current_chunk = chunk_labels[i]
            previous_chunk = chunk_labels[i-1]
            
            # Check if this chunk should start a new concept unit
            should_start_new_group = self.should_start_new_concept_group(
                previous_chunk, current_chunk, current_group
            )
            
            if should_start_new_group:
                groups.append(current_group)
                current_group = [current_chunk]
            else:
                current_group.append(current_chunk)
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def should_start_new_concept_group(self, prev_chunk: ChunkLabel, curr_chunk: ChunkLabel, current_group: List[ChunkLabel]) -> bool:
        """
        Determine if a new concept group should be started
        """
        # Always start new group for these labels
        new_topic_labels = ['NewTopic', 'Introduction', 'Conclusion', 'Summary', 'WrapUp']
        if curr_chunk.primary_label in new_topic_labels:
            return True
        
        # Don't let groups get too large (max 5 chunks)
        if len(current_group) >= 5:
            return True
        
        # Check for topic shift based on keywords
        prev_keywords = set(prev_chunk.concept_keywords)
        curr_keywords = set(curr_chunk.concept_keywords)
        
        # If keywords have very little overlap, might be new concept
        if prev_keywords and curr_keywords:
            overlap = len(prev_keywords & curr_keywords) / len(prev_keywords | curr_keywords)
            if overlap < 0.2:  # Less than 20% keyword overlap
                return True
        
        # Check label compatibility
        compatible_sequences = [
            ['Definition', 'Intuition', 'Example'],
            ['Statement', 'Proof', 'Example'],
            ['Theorem', 'Derivation', 'Proof'],
            ['ProblemStatement', 'Procedure', 'Algorithm'],
            ['Introduction', 'Background', 'Methodology'],
            ['Discussion', 'Analysis', 'Interpretation'],
        ]
        
        # Check if current sequence makes sense
        group_labels = [chunk.primary_label for chunk in current_group]
        extended_sequence = group_labels + [curr_chunk.primary_label]
        
        # If the extended sequence doesn't match any known pattern and is getting long, split
        if len(extended_sequence) > 3:
            return True
        
        return False
    
    def create_concept_unit(self, pdf_document: PDFDocument, chunk_group: List[ChunkLabel], order: int) -> ConceptUnit:
        """
        Create a ConceptUnit from a group of related chunks
        """
        # Generate title and description
        primary_labels = [chunk.primary_label for chunk in chunk_group]
        all_keywords = []
        for chunk in chunk_group:
            all_keywords.extend(chunk.concept_keywords)
        
        # Get most common keywords
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(5)]
        
        # Generate title from keywords and labels
        if top_keywords:
            title = f"Concept {order}: {', '.join(top_keywords[:3])}"
        else:
            title = f"Concept {order}: {primary_labels[0]}"
        
        # Create description
        description = f"Covers {', '.join(set(primary_labels))} with focus on {', '.join(top_keywords[:3])}"
        
        # Calculate total word count
        total_words = sum(chunk.text_chunk.word_count for chunk in chunk_group)
        
        # Create the concept unit
        concept_unit = ConceptUnit.objects.create(
            pdf_document=pdf_document,
            title=title[:255],  # Ensure it fits in CharField
            description=description,
            primary_labels=list(set(primary_labels)),
            concept_order=order,
            word_count=total_words
        )
        
        # Associate chunks with this concept unit
        for chunk_label in chunk_group:
            chunk_label.concept_unit = concept_unit
            chunk_label.save()
        
        return concept_unit

# Convenience function for easy import
def analyze_pdf_concepts(pdf_document: PDFDocument) -> Tuple[bool, str]:
    """Analyze PDF chunks, create concept units, and optimize by reading time"""
    analyzer = ConceptAnalyzer()
    return analyzer.analyze_pdf_concepts(pdf_document)
