import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from django.conf import settings
from .models import PDFDocument, ConceptUnit, FocusBlock
import re

logger = logging.getLogger(__name__)

class FocusBlockGenerator:
    """Service to generate Compact7 format focus blocks using LLM"""
    
    def __init__(self):
        # Initialize OpenAI client with Render proxy handling
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            self.client = None
            self.api_available = False
        else:
            try:
                # RENDER FIX: Remove proxy environment variables
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                original_proxies = {}
                
                for var in proxy_vars:
                    if var in os.environ:
                        original_proxies[var] = os.environ[var]
                        del os.environ[var]
                
                self.client = OpenAI(
                    api_key=api_key,
                    timeout=120.0,
                    max_retries=3
                )
                self.api_available = True
                
                # Restore proxy variables
                for var, value in original_proxies.items():
                    os.environ[var] = value
                    
            except Exception as e:
                print(f"❌ Focus block service OpenAI init failed: {e}")
                self.client = None
                self.api_available = False
        
        self.model = "gpt-4o-mini"  # Using the model specified in metadata
        self.target_duration = 420  # 7 minutes in seconds
    
    def generate_focus_blocks(self, pdf_document: PDFDocument) -> Tuple[bool, str]:
        """
        Generate Compact7 format focus blocks for all concept units in a PDF
        Returns: (success, message)
        """
        if not self.api_available:
            return False, "OpenAI API key not configured for focus block generation"
        
        try:
            # Get all concept units in order
            concept_units = list(pdf_document.concept_units.all().order_by('concept_order'))
            if not concept_units:
                return False, "No concept units found for focus block generation"
            
            print(f"🎯 DEBUG: Generating Compact7 focus blocks for {len(concept_units)} concept units")
            
            # Delete existing focus blocks
            pdf_document.focus_blocks.all().delete()
            
            # Generate focus blocks
            created_blocks = []
            for order, concept_unit in enumerate(concept_units, 1):
                # Find most related prior concept unit
                revision_unit = self.find_related_prior_unit(concept_unit, concept_units[:order-1])
                
                # Generate the focus block
                success, focus_block = self.generate_compact7_focus_block(
                    pdf_document, concept_unit, revision_unit, order
                )
                
                if success:
                    created_blocks.append(focus_block)
                    print(f"✅ DEBUG: Created Compact7 focus block {order}: {focus_block.title}")
                else:
                    print(f"❌ DEBUG: Failed to create focus block {order}")
            
            return True, f"Successfully created {len(created_blocks)} Compact7 focus blocks"
            
        except Exception as e:
            error_msg = f"Error generating focus blocks: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def find_related_prior_unit(self, current_unit: ConceptUnit, prior_units: List[ConceptUnit]) -> Optional[ConceptUnit]:
        """Find the most related prior concept unit for revision"""
        if not prior_units:
            return None
        
        # Simple similarity based on shared keywords and labels
        current_keywords = set(kw.lower() for kw in current_unit.primary_labels)
        current_keywords.update(kw.lower() for chunk_label in current_unit.chunk_labels.all() 
                               for kw in chunk_label.concept_keywords)
        
        best_match = None
        best_score = 0
        
        for prior_unit in reversed(prior_units[-3:]):  # Check last 3 units
            prior_keywords = set(kw.lower() for kw in prior_unit.primary_labels)
            prior_keywords.update(kw.lower() for chunk_label in prior_unit.chunk_labels.all() 
                                 for kw in chunk_label.concept_keywords)
            
            # Calculate similarity score
            if current_keywords and prior_keywords:
                intersection = len(current_keywords & prior_keywords)
                union = len(current_keywords | prior_keywords)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = prior_unit
        
        print(f"🔍 DEBUG: Found revision unit for '{current_unit.title}': '{best_match.title if best_match else 'None'}' (similarity: {best_score:.2f})")
        return best_match
    
    def generate_compact7_focus_block(self, pdf_document: PDFDocument, main_unit: ConceptUnit, 
                                     revision_unit: Optional[ConceptUnit], order: int) -> Tuple[bool, Optional[FocusBlock]]:
        """Generate a single Compact7 format focus block with better titles"""
        try:
            # Get content for the main concept unit
            main_content = main_unit.get_combined_text()
            revision_content = revision_unit.get_combined_text() if revision_unit else ""
            
            # Generate Compact7 JSON using LLM
            success, compact7_data = self.generate_compact7_json(
                main_content, revision_content, main_unit, revision_unit, order
            )
            
            if not success:
                return False, None
            
            # ✅ IMPROVED: Better title generation
            title = self.generate_focus_block_title(compact7_data, main_unit, order)
            
            # Create the focus block
            focus_block = FocusBlock.objects.create(
                pdf_document=pdf_document,
                main_concept_unit=main_unit,
                revision_concept_unit=revision_unit,
                block_order=order,
                title=title,
                target_duration=self.target_duration,
                compact7_data=compact7_data,
                difficulty_level='intermediate',
                learning_objectives=[compact7_data.get('core', {}).get('goal', '')],
            )
            
            return True, focus_block
            
        except Exception as e:
            logger.error(f"Error generating Compact7 focus block {order}: {str(e)}")
            return False, None
    
    def generate_compact7_json(self, main_content: str, revision_content: str, 
                              main_unit: ConceptUnit, revision_unit: Optional[ConceptUnit], 
                              order: int) -> Tuple[bool, Dict]:
        """Generate Compact7 JSON structure using LLM"""
        try:
            prompt = self.create_compact7_prompt(main_content, revision_content, main_unit, revision_unit, order)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational psychologist and instructional designer. Create focused 7-minute learning blocks following cognitive science principles."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"🤖 DEBUG: LLM Response length: {len(response_text)} characters")
            
            compact7_data = self.parse_compact7_response(response_text, main_unit, revision_unit, order)
            
            return True, compact7_data
            
        except Exception as e:
            logger.error(f"Error generating Compact7 JSON: {str(e)}")
            return False, {}
    
    def create_compact7_prompt(self, main_content: str, revision_content: str, 
                              main_unit: ConceptUnit, revision_unit: Optional[ConceptUnit], order: int) -> str:
        """Create prompt for Compact7 format generation"""
        
        revision_context = ""
        if revision_unit and revision_content:
            revision_context = f"""
PRIOR RELATED CONCEPT (for revision):
Title: {revision_unit.title}
Content: {revision_content[:600]}
"""
        
        prompt = f"""
Create a 7-minute focused learning block in Compact7 JSON format following educational psychology principles.

MAIN CONCEPT TO TEACH:
Title: {main_unit.title}
Labels: {', '.join(main_unit.primary_labels)}
Content: {main_content[:2000]}

{revision_context}

Generate a JSON object with this EXACT structure:

{{
  "template": "compact7",
  "revision": {{
    "nano_recap": "1-sentence summary of prior concept for context",
    "question": {{
      "type": "recall",
      "prompt": "Quick recall question about prior concept",
      "answer": "Expected answer for quick validation"
    }}
  }},
  "core": {{
    "goal": "Clear learning objective - what student will achieve",
    "segments": [
      {{
        "title": "First concept segment",
        "time_sec": 120,
        "teach": [
          {{"mode":"socratic","text":"Thought-provoking question to start"}},
          {{"mode":"explain","text":"Core concept explanation"}},
          {{"mode":"example","text":"Concrete example"}},
          {{"mode":"warning","text":"Common pitfall or misconception"}}
        ],
        "diagram_suggestion": "Visual aid description"
      }},
      {{
        "title": "Second concept segment", 
        "time_sec": 110,
        "teach": [
          {{"mode":"explain","text":"Build on previous segment"}},
          {{"mode":"example","text":"Another example"}},
          {{"mode":"benefit","text":"Why this matters"}}
        ]
      }},
      {{
        "title": "Third concept segment",
        "time_sec": 90,
        "teach": [
          {{"mode":"explain","text":"Complete the concept"}},
          {{"mode":"example","text":"Final example"}},
          {{"mode":"contrast","text":"Compare with alternatives"}}
        ]
      }}
    ],
    "misconceptions": [
      "Common misconception 1",
      "Common misconception 2"
    ],
    "fixes": [
      "How to correct misconception 1",
      "How to correct misconception 2"
    ]
  }},
  "qa": [
    {{
      "type": "concept",
      "prompt": "Conceptual understanding question",
      "ideal_answer": "What a good answer should include",
      "rubric": ["key_point_1", "key_point_2", "key_point_3"]
    }},
    {{
      "type": "applied", 
      "prompt": "Application/problem-solving question",
      "ideal_answer": "Expected solution approach",
      "solution_steps": ["step1", "step2", "step3"]
    }}
  ],
  "recap": {{
    "bullets": [
      "Key takeaway 1",
      "Key takeaway 2"  
    ],
    "confidence_prompt": "How confident are you with this concept (1-5)?",
    "next_review": "same-day-evening"
  }},
  "rescue": {{
    "nano_summary": "Ultra-brief concept summary for students who got lost",
    "one_min_reset": [
      "Simple question 1 to rebuild understanding",
      "Simple question 2",
      "Simple question 3",
      "Simple question 4"
    ]
  }},
  "metadata": {{ 
    "model": "gpt-4o-mini", 
    "prompt_version": "fb_v1_2025",
    "concept_unit_id": "{main_unit.id}",
    "block_order": {order}
  }}
}}

REQUIREMENTS:
- Total time for segments should be ~320 seconds (leaving room for QA and transitions)
- Use diverse teaching modes: socratic, explain, example, warning, benefit, contrast
- Create engaging, psychology-aware content that respects cognitive load
- Questions should test understanding and application
- Rescue content should be simpler and more direct
- Focus on the specific content provided, not generic concepts

Respond with ONLY the JSON object, no other text.
"""
        return prompt
    
    def parse_compact7_response(self, response_text: str, main_unit: ConceptUnit, 
                               revision_unit: Optional[ConceptUnit], order: int) -> Dict:
        """Parse LLM response for Compact7 format"""
        try:
            # Clean up response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            
            data = json.loads(response_text)
            
            # Validate that required structure exists
            required_keys = ['template', 'core', 'qa', 'recap', 'rescue']
            for key in required_keys:
                if key not in data:
                    print(f"⚠️ DEBUG: Missing required key '{key}' in LLM response")
            
            # Ensure metadata exists
            if 'metadata' not in data:
                data['metadata'] = {
                    "model": "gpt-4o-mini",
                    "prompt_version": "fb_v1_2025", 
                    "concept_unit_id": str(main_unit.id),
                    "block_order": order
                }
            
            print(f"✅ DEBUG: Successfully parsed Compact7 JSON with {len(data.get('core', {}).get('segments', []))} segments")
            return data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing Compact7 response: {str(e)}")
            print(f"💥 DEBUG: Failed to parse JSON. Response was: {response_text[:200]}...")
            
            # Return fallback Compact7 structure
            return self.create_fallback_compact7(main_unit, revision_unit, order)
    
    def create_fallback_compact7(self, main_unit: ConceptUnit, revision_unit: Optional[ConceptUnit], order: int) -> Dict:
        """Create fallback Compact7 structure when LLM parsing fails"""
        return {
            "template": "compact7",
            "revision": {
                "nano_recap": f"Previous concept: {revision_unit.title if revision_unit else 'N/A'}",
                "question": {
                    "type": "recall",
                    "prompt": "What was the main idea from the previous concept?",
                    "answer": "Review the key points from the prior material."
                }
            },
            "core": {
                "goal": f"Understand the concepts in {main_unit.title}",
                "segments": [
                    {
                        "title": "Core Concepts",
                        "time_sec": 180,
                        "teach": [
                            {"mode": "explain", "text": f"Let's explore {main_unit.title}"},
                            {"mode": "example", "text": "Consider the key examples in this material"},
                            {"mode": "warning", "text": "Pay attention to the important details"}
                        ]
                    },
                    {
                        "title": "Applications",
                        "time_sec": 140,
                        "teach": [
                            {"mode": "explain", "text": "How to apply these concepts"},
                            {"mode": "example", "text": "Practical applications"}
                        ]
                    }
                ],
                "misconceptions": ["Students might confuse key terms"],
                "fixes": ["Clarify definitions and provide examples"]
            },
            "qa": [
                {
                    "type": "concept",
                    "prompt": "What are the main concepts in this material?",
                    "ideal_answer": "The key ideas and their relationships",
                    "rubric": ["key_concepts", "relationships", "examples"]
                },
                {
                    "type": "applied",
                    "prompt": "How would you apply these concepts?",
                    "ideal_answer": "Through practical examples and exercises",
                    "solution_steps": ["identify_problem", "apply_concept", "verify_solution"]
                }
            ],
            "recap": {
                "bullets": [
                    f"Learned about {main_unit.title}",
                    "Understood key applications"
                ],
                "confidence_prompt": "How confident are you (1-5)?",
                "next_review": "same-day-evening"
            },
            "rescue": {
                "nano_summary": f"Key concept: {main_unit.title}",
                "one_min_reset": [
                    "What is the main topic?",
                    "What are the key components?",
                    "How do they work together?",
                    "What are real-world examples?"
                ]
            },
            "metadata": {
                "model": "gpt-4o-mini",
                "prompt_version": "fb_v1_2025_fallback",
                "concept_unit_id": str(main_unit.id),
                "block_order": order
            }
        }

    def generate_focus_block_title(self, compact7_data: Dict, main_unit: ConceptUnit, order: int) -> str:
        """Generate a meaningful title for the focus block"""
        
        # Try different sources for title, in order of preference
        title_sources = []
        
        # 1. Use core goal (cleaned up)
        if 'core' in compact7_data and 'goal' in compact7_data['core']:
            goal = compact7_data['core']['goal']
            # Clean and shorten the goal
            clean_goal = self.clean_goal_for_title(goal)
            if clean_goal:
                title_sources.append(clean_goal)
        
        # 2. Use first segment title
        if 'core' in compact7_data and 'segments' in compact7_data['core']:
            segments = compact7_data['core']['segments']
            if segments and 'title' in segments[0]:
                segment_title = segments[0]['title']
                title_sources.append(segment_title)
        
        # 3. Use concept unit title (cleaned)
        if main_unit.title:
            clean_unit_title = main_unit.title.replace('Combined Concept', '').strip()
            if clean_unit_title:
                title_sources.append(clean_unit_title)
        
        # 4. Use primary labels
        if main_unit.primary_labels:
            labels_title = ' & '.join(main_unit.primary_labels[:2])  # First 2 labels
            title_sources.append(f"Learn {labels_title}")
        
        # 5. Fallback
        title_sources.append(f"Focus Session {order}")
        
        # Pick the best title
        for potential_title in title_sources:
            if potential_title and len(potential_title.strip()) > 5:
                # Format nicely
                final_title = f"{order}. {potential_title}"
                # Truncate if too long
                if len(final_title) > 100:
                    final_title = final_title[:97] + "..."
                return final_title
        
        return f"Focus Session {order}"

    def clean_goal_for_title(self, goal: str) -> str:
        """Clean and shorten the goal text to make a good title"""
        if not goal:
            return ""
        
        # Remove common prefixes
        goal = goal.replace("Understand ", "").replace("Learn about ", "").replace("Explain ", "")
        goal = goal.replace("Students will ", "").replace("The student will ", "")
        
        # Split on punctuation and take first meaningful part
        parts = re.split(r'[.;,]', goal)
        clean_part = parts[0].strip()
        
        # Remove action words
        clean_part = re.sub(r'^(understand|learn|know|identify|explain|describe|analyze)\s+', '', clean_part, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if clean_part:
            clean_part = clean_part[0].upper() + clean_part[1:]
        
        # Truncate if too long
        if len(clean_part) > 60:
            clean_part = clean_part[:57] + "..."
            
        return clean_part

# Convenience function
def generate_focus_blocks(pdf_document: PDFDocument) -> Tuple[bool, str]:
    """Generate Compact7 format focus blocks for all concept units in a PDF"""
    print(f"🎯 Starting focus block generation for PDF: {pdf_document.name}")
    
    generator = FocusBlockGenerator()
    
    if not generator.api_available:
        print("❌ OpenAI API not available")
        return False, "OpenAI API key not configured for focus block generation"
    
    try:
        # Get all concept units in order
        concept_units = list(pdf_document.concept_units.all().order_by('concept_order'))
        if not concept_units:
            print("❌ No concept units found")
            return False, "No concept units found for focus block generation"
        
        print(f"🎯 Found {len(concept_units)} concept units to process")
        
        # Delete existing focus blocks
        existing_count = pdf_document.focus_blocks.count()
        pdf_document.focus_blocks.all().delete()
        print(f"🗑️ Deleted {existing_count} existing focus blocks")
        
        # ✅ FIXED: Direct call without signal timeout
        success, message = generator.generate_focus_blocks(pdf_document)
        print(f"🎯 Focus block generation completed: {success}")
        
        return success, message
            
    except Exception as e:
        print(f"💥 Exception in focus block generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error generating focus blocks: {str(e)}"
