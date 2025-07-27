import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from django.conf import settings
from .models import PDFDocument, ConceptUnit, ChunkLabel, TextChunk

logger = logging.getLogger(__name__)

class ReadingTimeEstimator:
    """Service to estimate reading time and optimize concept units using LLM"""
    
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        
        print(f"⏱️ DEBUG: ReadingTimeEstimator - API key exists: {bool(api_key)}")  # ✅ Debug print
        
        if not api_key or api_key == 'your_openai_api_key_here':
            self.client = None
            self.api_available = False
            print("❌ DEBUG: ReadingTimeEstimator - OpenAI API not available")  # ✅ Debug print
        else:
            self.client = OpenAI(api_key=api_key)
            self.api_available = True
            print("✅ DEBUG: ReadingTimeEstimator - OpenAI client initialized")  # ✅ Debug print
        
        self.model = "gpt-3.5-turbo"
        self.target_time = 10.0  # Target 10 minutes per concept unit
        self.min_time = 5.0      # Minimum viable time
        self.max_time = 10.0     # Maximum time before splitting
    
    def optimize_concept_units_by_time(self, pdf_document: PDFDocument) -> Tuple[bool, str]:
        """
        Main method to optimize concept units based on reading time
        Returns: (success, message)
        """
        print(f"⏱️ DEBUG: Starting time optimization for PDF {pdf_document.id}")  # ✅ Debug print
        
        if not self.api_available:
            print("❌ DEBUG: Time optimization skipped - API not available")  # ✅ Debug print
            # ✅ Use fallback time estimation instead of failing
            return self.optimize_with_fallback_times(pdf_document)
        
        try:
            # Step 1: Get all concept units
            concept_units = list(pdf_document.concept_units.all().order_by('concept_order'))
            print(f"📊 DEBUG: Found {len(concept_units)} concept units to optimize")  # ✅ Debug print
            
            if not concept_units:
                return False, "No concept units found for optimization"
            
            # Step 2: Calculate reading time for each unit
            successful_estimates = 0
            
            for i, unit in enumerate(concept_units, 1):
                print(f"⏱️ DEBUG: Estimating time for unit {i}/{len(concept_units)} (ID: {unit.id})")  # ✅ Debug print
                
                success, time_data = self.estimate_reading_time(unit)
                if success:
                    unit.estimated_reading_time = time_data['reading_time']
                    unit.complexity_score = time_data['complexity_score']
                    unit.cognitive_load = time_data['cognitive_load']
                    unit.original_order = unit.concept_order
                    unit.save()
                    successful_estimates += 1
                    print(f"✅ DEBUG: Unit {i} time estimated: {time_data['reading_time']:.1f} minutes")  # ✅ Debug print
                else:
                    # Fallback time estimation
                    fallback_time = self.fallback_time_estimate(unit)
                    unit.estimated_reading_time = fallback_time
                    unit.complexity_score = 0.5
                    unit.cognitive_load = 'medium'
                    unit.original_order = unit.concept_order
                    unit.save()
                    print(f"⚠️ DEBUG: Unit {i} using fallback time: {fallback_time:.1f} minutes")  # ✅ Debug print
            
            print(f"📈 DEBUG: Time estimation complete. Success: {successful_estimates}/{len(concept_units)}")  # ✅ Debug print
            
            # Step 3: Optimize units (combine short ones, split long ones)
            print(f"🔄 DEBUG: Starting time-based optimization...")  # ✅ Debug print
            optimized_units = self.optimize_time_distribution(concept_units)
            print(f"🔄 DEBUG: Optimization complete. {len(optimized_units)} final units")  # ✅ Debug print
            
            # Step 4: Update database with optimized units
            self.save_optimized_units(pdf_document, optimized_units)
            
            return True, f"Optimized into {len(optimized_units)} time-balanced concept units"
            
        except Exception as e:
            error_msg = f"Error in reading time optimization: {str(e)}"
            print(f"💥 DEBUG: Time optimization exception: {error_msg}")  # ✅ Debug print
            logger.error(error_msg)
            return False, error_msg
    
    def optimize_with_fallback_times(self, pdf_document: PDFDocument) -> Tuple[bool, str]:
        """Optimize using only fallback time estimation when API is not available"""
        print(f"🔄 DEBUG: Using fallback time optimization")  # ✅ Debug print
        
        try:
            concept_units = list(pdf_document.concept_units.all().order_by('concept_order'))
            
            for unit in concept_units:
                fallback_time = self.fallback_time_estimate(unit)
                unit.estimated_reading_time = fallback_time
                unit.complexity_score = 0.5
                unit.cognitive_load = 'medium'
                unit.time_optimized = True  # Mark as optimized even with fallback
                unit.save()
                print(f"📊 DEBUG: Unit {unit.concept_order} fallback time: {fallback_time:.1f} minutes")  # ✅ Debug print
            
            return True, f"Optimized {len(concept_units)} units using fallback time estimation"
            
        except Exception as e:
            error_msg = f"Error in fallback time optimization: {str(e)}"
            print(f"💥 DEBUG: Fallback optimization failed: {error_msg}")  # ✅ Debug print
            return False, error_msg
    
    def estimate_reading_time(self, concept_unit: ConceptUnit) -> Tuple[bool, Dict]:
        """
        Estimate reading time for a concept unit using LLM
        Returns: (success, time_data)
        """
        try:
            content_text = concept_unit.get_combined_text()
            print(f"📝 DEBUG: Estimating time for content length: {len(content_text)} chars")  # ✅ Debug print
            
            if not content_text:
                print(f"⚠️ DEBUG: No content found for unit {concept_unit.id}")  # ✅ Debug print
                return False, "No content found for this concept unit"
            
            prompt = self.create_time_estimation_prompt(content_text, concept_unit.primary_labels)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational psychologist specializing in reading comprehension and learning time estimation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"🤖 DEBUG: LLM time response: {response_text[:100]}...")  # ✅ Debug print
            
            time_data = self.parse_time_estimation_response(response_text)
            print(f"⏱️ DEBUG: Parsed time data: {time_data}")  # ✅ Debug print
            
            return True, time_data
            
        except Exception as e:
            error_msg = f"Error estimating reading time for concept unit {concept_unit.id}: {str(e)}"
            print(f"💥 DEBUG: Time estimation failed: {error_msg}")  # ✅ Debug print
            logger.error(error_msg)
            return False, error_msg
    
    def create_time_estimation_prompt(self, content_text: str, labels: List[str]) -> str:
        """Create prompt for reading time estimation"""
        prompt = f"""
Estimate the reading and comprehension time for the following educational content.

Content Type: {', '.join(labels)}
Word Count: {len(content_text.split())}

Content:
---
{content_text[:2000]}  # Limit content length for API
---

Consider these factors for time estimation:
1. Content complexity (definitions vs proofs vs examples)
2. Cognitive load required for understanding
3. Need for re-reading or deeper thinking
4. Mathematical formulas, code, or technical diagrams
5. Typical student reading and comprehension speed

Please respond with a JSON object:
{{
    "reading_time": 8.5,  // Estimated time in minutes for average student
    "complexity_score": 0.7,  // 0-1 scale (0=simple, 1=very complex)
    "cognitive_load": "high",  // "low", "medium", "high", "very_high"
    "reasoning": "Complex mathematical proofs require slower reading and multiple passes",
    "factors": ["mathematical_notation", "abstract_concepts", "proof_steps"]
}}

Base reading speed: 200-250 words per minute for technical content.
Adjust for complexity, comprehension needs, and content type.
"""
        return prompt
    
    def parse_time_estimation_response(self, response_text: str) -> Dict:
        """Parse LLM response for time estimation"""
        try:
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            data = json.loads(response_text)
            
            # Validate and sanitize response
            reading_time = float(data.get('reading_time', 5.0))
            complexity_score = max(0.0, min(1.0, float(data.get('complexity_score', 0.5))))
            cognitive_load = data.get('cognitive_load', 'medium')
            
            if cognitive_load not in ['low', 'medium', 'high', 'very_high']:
                cognitive_load = 'medium'
            
            return {
                'reading_time': reading_time,
                'complexity_score': complexity_score,
                'cognitive_load': cognitive_load,
                'reasoning': data.get('reasoning', ''),
                'factors': data.get('factors', [])
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing time estimation response: {str(e)}")
            return {
                'reading_time': 5.0,
                'complexity_score': 0.5,
                'cognitive_load': 'medium',
                'reasoning': 'Failed to parse LLM response',
                'factors': []
            }
    
    def fallback_time_estimate(self, concept_unit: ConceptUnit) -> float:
        """Fallback time estimation based on word count and labels"""
        print(f"🔄 DEBUG: Calculating fallback time for unit {concept_unit.concept_order}")  # ✅ Debug print
        
        base_wpm = 200  # Base words per minute for technical content
        
        # Adjust reading speed based on content type
        speed_modifiers = {
            'Definition': 1.2,      # Slightly faster
            'Example': 1.1,         # Normal pace
            'Theorem': 0.6,         # Much slower
            'Proof': 0.5,           # Very slow
            'Algorithm': 0.7,       # Slow
            'Code': 0.8,           # Slower
            'Exercise': 0.9,        # Slightly slower
            'Discussion': 1.0,      # Normal
        }
        
        # Calculate average speed modifier
        if concept_unit.primary_labels:
            modifiers = [speed_modifiers.get(label, 1.0) for label in concept_unit.primary_labels]
            avg_modifier = sum(modifiers) / len(modifiers) if modifiers else 1.0
        else:
            avg_modifier = 1.0
        
        print(f"📊 DEBUG: Word count: {concept_unit.word_count}, Speed modifier: {avg_modifier:.2f}")  # ✅ Debug print
        
        # Adjust WPM
        adjusted_wpm = base_wpm * avg_modifier
        
        # Calculate time
        reading_time = concept_unit.word_count / adjusted_wpm
        
        # Add comprehension time (30% extra)
        comprehension_factor = 1.3
        total_time = reading_time * comprehension_factor
        
        final_time = max(1.0, total_time)  # Minimum 1 minute
        print(f"⏱️ DEBUG: Fallback time calculated: {final_time:.1f} minutes")  # ✅ Debug print
        
        return final_time
    
    def optimize_time_distribution(self, concept_units: List[ConceptUnit]) -> List[List[ConceptUnit]]:
        """
        Optimize concept units by combining short ones and splitting long ones
        Returns list of optimized unit groups
        """
        optimized_groups = []
        i = 0
        
        while i < len(concept_units):
            current_unit = concept_units[i]
            
            # Case 1: Unit is too long (> 10 minutes) - needs splitting
            if current_unit.needs_splitting():
                split_groups = self.split_concept_unit(current_unit)
                optimized_groups.extend(split_groups)
                i += 1
            
            # Case 2: Unit is short enough to potentially combine
            elif i + 1 < len(concept_units):
                next_unit = concept_units[i + 1]
                
                # Check if we can combine with next unit
                if current_unit.can_be_combined(next_unit):
                    combined_group = [current_unit, next_unit]
                    combined_time = current_unit.estimated_reading_time + next_unit.estimated_reading_time
                    
                    # Try to add more units to the group if possible
                    j = i + 2
                    while j < len(concept_units) and combined_time < self.max_time:
                        candidate_unit = concept_units[j]
                        if combined_time + candidate_unit.estimated_reading_time <= self.max_time:
                            combined_group.append(candidate_unit)
                            combined_time += candidate_unit.estimated_reading_time
                            j += 1
                        else:
                            break
                    
                    optimized_groups.append(combined_group)
                    i = j  # Skip all units we just combined
                else:
                    # Unit stands alone
                    optimized_groups.append([current_unit])
                    i += 1
            else:
                # Last unit, stands alone
                optimized_groups.append([current_unit])
                i += 1
        
        return optimized_groups
    
    def split_concept_unit(self, unit: ConceptUnit) -> List[List[ConceptUnit]]:
        """
        Split a concept unit that's too long into smaller units
        Returns list of unit groups (each group will become a new unit)
        """
        # Get all chunks in this unit
        chunk_labels = list(unit.chunk_labels.select_related('text_chunk').order_by('text_chunk__chunk_order'))
        
        if len(chunk_labels) <= 1:
            # Can't split further, return as is
            return [[unit]]
        
        # Split chunks roughly in half
        mid_point = len(chunk_labels) // 2
        
        # Try to find a good split point (avoid breaking related content)
        split_point = self.find_optimal_split_point(chunk_labels, mid_point)
        
        # Create two groups
        group1_chunks = chunk_labels[:split_point]
        group2_chunks = chunk_labels[split_point:]
        
        # Estimate time for each group
        group1_time = unit.estimated_reading_time * (len(group1_chunks) / len(chunk_labels))
        group2_time = unit.estimated_reading_time * (len(group2_chunks) / len(chunk_labels))
        
        # Create temporary units for further optimization
        unit1 = self.create_temp_unit_from_chunks(unit, group1_chunks, group1_time)
        unit2 = self.create_temp_unit_from_chunks(unit, group2_chunks, group2_time)
        
        # Recursively optimize if still too long
        result = []
        if unit1.needs_splitting():
            result.extend(self.split_concept_unit(unit1))
        else:
            result.append([unit1])
        
        if unit2.needs_splitting():
            result.extend(self.split_concept_unit(unit2))
        else:
            result.append([unit2])
        
        return result
    
    def find_optimal_split_point(self, chunk_labels: List, preferred_point: int) -> int:
        """Find the best place to split chunks to maintain logical flow"""
        # Look for natural break points around the preferred split
        search_range = min(2, len(chunk_labels) // 4)  # Search within reasonable range
        
        for offset in range(search_range + 1):
            # Check points around the preferred split
            for direction in [0, 1, -1]:  # Check preferred point first, then before/after
                point = preferred_point + (direction * offset)
                if 0 < point < len(chunk_labels):
                    if self.is_good_split_point(chunk_labels, point):
                        return point
        
        # Fallback to preferred point
        return max(1, min(preferred_point, len(chunk_labels) - 1))
    
    def is_good_split_point(self, chunk_labels: List, point: int) -> bool:
        """Check if a point is a good place to split content"""
        if point <= 0 or point >= len(chunk_labels):
            return False
        
        current_label = chunk_labels[point - 1].primary_label
        next_label = chunk_labels[point].primary_label
        
        # Good split points
        good_transitions = [
            ('WrapUp', 'NewTopic'),
            ('Conclusion', 'Introduction'),
            ('Example', 'Definition'),
            ('Proof', 'Statement'),
            ('Exercise', 'Theorem'),
        ]
        
        # Check if this is a natural transition
        if (current_label, next_label) in good_transitions:
            return True
        
        # Avoid splitting related content
        bad_splits = [
            ('Definition', 'Example'),
            ('Statement', 'Proof'),
            ('Theorem', 'Derivation'),
            ('Algorithm', 'Pseudocode'),
        ]
        
        if (current_label, next_label) in bad_splits:
            return False
        
        return True
    
    def create_temp_unit_from_chunks(self, original_unit: ConceptUnit, chunk_labels: List, estimated_time: float):
        """Create a temporary concept unit from a subset of chunks"""
        # This creates a temporary unit for optimization calculations
        # It won't be saved to the database until the final optimization step
        temp_unit = ConceptUnit(
            pdf_document=original_unit.pdf_document,
            title=f"{original_unit.title} (part)",
            description=original_unit.description,
            primary_labels=list(set([chunk.primary_label for chunk in chunk_labels])),
            concept_order=original_unit.concept_order,
            word_count=sum(chunk.text_chunk.word_count for chunk in chunk_labels),
            estimated_reading_time=estimated_time,
            complexity_score=original_unit.complexity_score,
            cognitive_load=original_unit.cognitive_load
        )
        
        # Store chunk references temporarily
        temp_unit._temp_chunks = chunk_labels
        
        return temp_unit
    
    def save_optimized_units(self, pdf_document: PDFDocument, optimized_groups: List[List[ConceptUnit]]):
        """Save the optimized concept units to database"""
        print(f"💾 DEBUG: Saving {len(optimized_groups)} optimized unit groups")  # ✅ Debug print
        
        # ✅ Fix: Delete existing concept units and create new optimized ones
        # This avoids UNIQUE constraint violations
        
        # Step 1: Collect all chunk labels before deleting concept units
        from .models import ChunkLabel
        all_chunk_data = []
        
        for order, group in enumerate(optimized_groups, 1):
            group_chunk_data = []
            
            for unit in group:
                # Get all chunk labels for this unit
                chunk_labels = ChunkLabel.objects.filter(concept_unit=unit).select_related('text_chunk')
                for chunk_label in chunk_labels:
                    group_chunk_data.append({
                        'chunk_label': chunk_label,
                        'text_chunk': chunk_label.text_chunk,
                        'primary_label': chunk_label.primary_label,
                        'secondary_labels': chunk_label.secondary_labels,
                        'confidence_score': chunk_label.confidence_score,
                        'reasoning': chunk_label.reasoning,
                        'concept_keywords': chunk_label.concept_keywords,
                    })
            
            # Calculate combined metadata for the new unit
            combined_labels = []
            total_words = 0
            total_time = 0
            all_keywords = []
            avg_complexity = 0
            cognitive_loads = []
            
            for unit in group:
                combined_labels.extend(unit.primary_labels)
                total_words += unit.word_count
                total_time += unit.estimated_reading_time
                avg_complexity += unit.complexity_score
                cognitive_loads.append(unit.cognitive_load)
            
            # Calculate averages
            avg_complexity = avg_complexity / len(group) if group else 0.5
            
            # Choose highest cognitive load
            load_hierarchy = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
            max_load = max(cognitive_loads, key=lambda x: load_hierarchy.get(x, 2)) if cognitive_loads else 'medium'
            
            # Create title
            unique_labels = list(set(combined_labels))
            if len(group) == 1:
                title = group[0].title
            else:
                title = f"Combined Concept {order}: {', '.join(unique_labels[:3])}"
            
            all_chunk_data.append({
                'order': order,
                'chunk_data': group_chunk_data,
                'title': title[:255],
                'description': f"Time-optimized unit covering {', '.join(unique_labels)}",
                'primary_labels': unique_labels,
                'word_count': total_words,
                'estimated_reading_time': total_time,
                'complexity_score': avg_complexity,
                'cognitive_load': max_load,
            })
        
        print(f"📊 DEBUG: Collected data for {len(all_chunk_data)} optimized units")
        
        # Step 2: Delete existing concept units (this will also unlink chunk labels)
        deleted_count = pdf_document.concept_units.all().delete()[0]
        print(f"🗑️ DEBUG: Deleted {deleted_count} existing concept units")
        
        # Step 3: Create new optimized concept units
        for unit_data in all_chunk_data:
            # Create the new concept unit
            new_unit = ConceptUnit.objects.create(
                pdf_document=pdf_document,
                title=unit_data['title'],
                description=unit_data['description'],
                primary_labels=unit_data['primary_labels'],
                concept_order=unit_data['order'],
                word_count=unit_data['word_count'],
                estimated_reading_time=unit_data['estimated_reading_time'],
                complexity_score=unit_data['complexity_score'],
                cognitive_load=unit_data['cognitive_load'],
                time_optimized=True
            )
            
            print(f"✅ DEBUG: Created optimized unit {unit_data['order']}: {unit_data['title']} ({unit_data['estimated_reading_time']:.1f} min)")
            
            # Step 4: Re-link chunk labels to the new unit
            for chunk_info in unit_data['chunk_data']:
                chunk_label = chunk_info['chunk_label']
                chunk_label.concept_unit = new_unit
                chunk_label.save()
            
            print(f"🔗 DEBUG: Linked {len(unit_data['chunk_data'])} chunks to unit {unit_data['order']}")
        
        print(f"💾 DEBUG: Time optimization save complete - created {len(all_chunk_data)} optimized units")

# Convenience function for easy import
def optimize_reading_time(pdf_document: PDFDocument) -> Tuple[bool, str]:
    """Optimize concept units based on reading time"""
    estimator = ReadingTimeEstimator()
    return estimator.optimize_concept_units_by_time(pdf_document)
