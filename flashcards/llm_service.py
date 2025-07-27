import os
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from django.conf import settings
from .models import Folder, Flashcard

class FlashcardLLMService:
    """Service for generating enhanced flashcards using OpenAI"""
    
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"  # Using GPT-3.5 for cost efficiency
    
    def enhance_flashcards(self, raw_content: List[Dict[str, Any]], folder: Folder) -> List[Dict[str, Any]]:
        """
        Enhance raw extracted content into educational flashcards using LLM
        """
        enhanced_flashcards = []
        
        # Process in batches to manage API costs and rate limits
        batch_size = 3  # Process 3 concepts at a time
        
        for i in range(0, len(raw_content), batch_size):
            batch = raw_content[i:i + batch_size]
            
            try:
                # Create a comprehensive prompt for the batch
                prompt = self._create_batch_prompt(batch, folder.name)
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert educational content creator specializing in creating effective flashcards for learning and retention."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                # Parse the response
                enhanced_batch = self._parse_llm_response(response.choices[0].message.content, batch)
                enhanced_flashcards.extend(enhanced_batch)
                
                # Rate limiting - wait between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Fallback to original content if LLM fails
                for item in batch:
                    enhanced_flashcards.append(self._create_fallback_flashcard(item))
        
        return enhanced_flashcards
    
    def _create_batch_prompt(self, batch: List[Dict[str, Any]], topic: str) -> str:
        """Create a comprehensive prompt for a batch of content"""
        
        prompt = f"""Create high-quality educational flashcards from the following content about "{topic}".

For each content item below, create ONE improved flashcard that:
1. Has a clear, specific question that tests understanding
2. Provides a comprehensive but concise answer
3. Uses educational best practices for spaced repetition learning
4. Is appropriate for the content type (code, concept, list, etc.)

IMPORTANT: Return your response as a JSON array where each object has exactly these fields:
- "question": The improved question
- "answer": The comprehensive answer
- "difficulty": "easy", "medium", or "hard"
- "tags": Array of relevant tags/keywords

Content to process:
"""
        
        for i, item in enumerate(batch, 1):
            prompt += f"""
--- Content {i} ---
Source: {item['source_file']}
Content Type: {item['content_type']}
Context: {item['context']}
Original Question: {item['question']}
Original Answer: {item['answer']}
Original Content: {item['original_content']}
"""
        
        prompt += """

Return ONLY a JSON array with the improved flashcards. Do not include any other text."""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, original_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the LLM response and combine with original metadata"""
        enhanced_flashcards = []
        
        try:
            # Try to parse JSON response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            llm_flashcards = json.loads(response_text)
            
            # Combine LLM enhancements with original metadata
            for i, (llm_card, original) in enumerate(zip(llm_flashcards, original_batch)):
                enhanced_card = {
                    'source_file': original['source_file'],
                    'question': llm_card.get('question', original['question']),
                    'answer': llm_card.get('answer', original['answer']),
                    'context': original['context'],
                    'content_type': original['content_type'],
                    'difficulty_level': llm_card.get('difficulty', 'medium'),
                    'tags': llm_card.get('tags', []),
                    'original_content': original['original_content']
                }
                enhanced_flashcards.append(enhanced_card)
        
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response_text[:200]}...")
            
            # Fallback to original content
            for item in original_batch:
                enhanced_flashcards.append(self._create_fallback_flashcard(item))
        
        return enhanced_flashcards
    
    def _create_fallback_flashcard(self, original: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback flashcard when LLM processing fails"""
        return {
            'source_file': original['source_file'],
            'question': original['question'],
            'answer': original['answer'],
            'context': original['context'],
            'content_type': original['content_type'],
            'difficulty_level': 'medium',
            'tags': [original['context'].lower().replace(' ', '_')],
            'original_content': original['original_content']
        }
    
    def save_flashcards_to_db(self, enhanced_flashcards: List[Dict[str, Any]], folder: Folder) -> int:
        """Save enhanced flashcards to the database"""
        saved_count = 0
        
        for card_data in enhanced_flashcards:
            try:
                flashcard = Flashcard.objects.create(
                    folder=folder,
                    question=card_data['question'],
                    answer=card_data['answer'],
                    source_file=card_data['source_file'],
                    source_content=card_data['original_content'],
                    difficulty_level=card_data['difficulty_level']
                )
                saved_count += 1
                
            except Exception as e:
                print(f"Error saving flashcard: {e}")
                continue
        
        # Mark folder as processed
        folder.processed = True
        folder.save()
        
        return saved_count

def process_folder_with_llm(folder: Folder, raw_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to process folder content with LLM enhancement
    """
    try:
        llm_service = FlashcardLLMService()
        
        # Enhance content with LLM
        enhanced_flashcards = llm_service.enhance_flashcards(raw_content, folder)
        
        # Save to database
        saved_count = llm_service.save_flashcards_to_db(enhanced_flashcards, folder)
        
        return {
            'success': True,
            'processed_count': len(enhanced_flashcards),
            'saved_count': saved_count,
            'message': f"Successfully created {saved_count} flashcards from {len(raw_content)} content items"
        }
        
    except ValueError as e:
        # API key not configured
        return {
            'success': False,
            'error': 'api_key_missing',
            'message': str(e)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': 'processing_failed',
            'message': f"Error processing with LLM: {str(e)}"
        } 