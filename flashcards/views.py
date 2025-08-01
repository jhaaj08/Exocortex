from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.views.decorators.http import require_http_methods
from django.db import models  # Add this import
from .models import Folder, Flashcard, StudySession, PDFDocument, TextChunk, ChunkLabel, ConceptUnit, FocusBlock, FocusBlockRelationship
from .forms import FolderInputForm, PDFUploadForm
from .services import process_folder_content
from .llm_service import process_folder_with_llm
from .pdf_service import extract_pdf_text
from .focus_block_service import generate_focus_blocks
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from django.conf import settings
import hashlib
from difflib import SequenceMatcher
from django.utils import timezone
import uuid
from .models import FocusSession
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
from datetime import timedelta
from collections import defaultdict
from collections import deque

logger = logging.getLogger(__name__)

def home(request):
    """Home page with PDF upload functionality and ALL processed PDFs"""
    pdf_form = PDFUploadForm()
    extracted_text = None
    pdf_document = None
    processing_error = None
    processing_stats = None
    
    print(f"🏠 Home view called - Method: {request.method}")  # Debug
    
    # Handle PDF upload
    if request.method == 'POST' and 'pdf_upload' in request.POST:
        print("📤 PDF upload detected")  # Debug
        
        pdf_form = PDFUploadForm(request.POST, request.FILES)
        print(f"📋 Form valid: {pdf_form.is_valid()}")  # Debug
        
        if pdf_form.is_valid():
            try:
                print("💾 Creating PDF document...")  # Debug
                pdf_document = pdf_form.save(commit=False)
                pdf_document.save()
                print(f"✅ PDF saved with ID: {pdf_document.id}")  # Debug
                
                # Use the simplified processing
                success, result_message, stats = process_pdf_complete(pdf_document)
                
                if success:
                    messages.success(request, f'✅ {pdf_document.name}: {result_message}')
                    print("✅ Processing completed successfully")
                    # Redirect to progress page to show what happened
                    return redirect('flashcards:pdf_progress', pdf_id=pdf_document.id)
                else:
                    messages.error(request, f'❌ {pdf_document.name}: {result_message}')
                    print(f"❌ Processing failed: {result_message}")
                    
            except Exception as e:
                error_msg = f"Upload error: {str(e)}"
                print(f"💥 Exception in upload: {error_msg}")
                messages.error(request, error_msg)
        else:
            print(f"❌ Form errors: {pdf_form.errors}")  # Debug
            messages.error(request, 'Please check the form and try again.')
    
    # Get ALL processed PDFs with calculated fields
    recent_pdfs = PDFDocument.objects.all().order_by('-created_at')
    
    # Calculate stats
    for pdf in recent_pdfs:
        pdf.focus_blocks_count = pdf.focus_blocks.count()
        pdf.chunks_count = pdf.text_chunks.count()
        pdf.concepts_count = pdf.concept_units.count()
        pdf.has_focus_blocks = pdf.focus_blocks_count > 0
        pdf.total_study_time = pdf.focus_blocks_count * 7
        
        # ✅ Add duplicate status
        pdf.is_duplicate_display = getattr(pdf, 'is_duplicate', False)
        pdf.duplicate_source = getattr(pdf, 'duplicate_of', None)
    
    context = {
        'pdf_form': pdf_form,
        'extracted_text': extracted_text,
        'pdf_document': pdf_document,
        'processing_error': processing_error,
        'processing_stats': processing_stats,
        'recent_pdfs': recent_pdfs,
        'total_pdfs': recent_pdfs.count(),
        'ready_to_study': recent_pdfs.filter(focus_blocks__isnull=False).distinct().count(),
        'duplicates_count': recent_pdfs.filter(is_duplicate=True).count(),
    }
    
    print(f"📄 Returning context with {recent_pdfs.count()} PDFs")  # Debug
    return render(request, 'flashcards/home.html', context)

def add_folder(request):
    """View to add a new folder containing markdown files"""
    if request.method == 'POST':
        form = FolderInputForm(request.POST)
        if form.is_valid():
            folder = form.save()
            messages.success(
                request, 
                f'Folder "{folder.name}" added successfully with {folder.total_files} markdown files!'
            )
            return redirect('flashcards:process_folder', folder_id=folder.id)
    else:
        form = FolderInputForm()
    
    return render(request, 'flashcards/add_folder.html', {'form': form})

def folder_list(request):
    """View to list all folders"""
    folders = Folder.objects.all()
    return render(request, 'flashcards/folder_list.html', {'folders': folders})

def folder_detail(request, folder_id):
    """View to show details of a specific folder"""
    folder = get_object_or_404(Folder, id=folder_id)
    flashcards = folder.flashcards.all()
    
    context = {
        'folder': folder,
        'flashcards': flashcards,
        'total_cards': flashcards.count(),
    }
    return render(request, 'flashcards/folder_detail.html', context)

def process_folder(request, folder_id):
    """View to process markdown files and generate flashcards"""
    folder = get_object_or_404(Folder, id=folder_id)
    
    if folder.processed:
        messages.info(request, f'Folder "{folder.name}" has already been processed.')
        return redirect('flashcards:folder_detail', folder_id=folder.id)
    
    context = {
        'folder': folder,
    }
    return render(request, 'flashcards/process_folder.html', context)

@require_http_methods(["POST"])
def process_folder_ajax(request, folder_id):
    """AJAX endpoint to process folder and generate flashcards"""
    folder = get_object_or_404(Folder, id=folder_id)
    
    if folder.processed:
        return JsonResponse({
            'success': False,
            'error': 'already_processed',
            'message': 'This folder has already been processed.'
        })
    
    try:
        # Step 1: Extract content from markdown files
        raw_content = process_folder_content(folder)
        
        if not raw_content:
            return JsonResponse({
                'success': False,
                'error': 'no_content',
                'message': 'No content could be extracted from the markdown files.'
            })
        
        # Step 2: Process with LLM (if API key is configured)
        result = process_folder_with_llm(folder, raw_content)
        
        if result['success']:
            return JsonResponse({
                'success': True,
                'message': result['message'],
                'processed_count': result['processed_count'],
                'saved_count': result['saved_count'],
                'redirect_url': f'/folder/{folder.id}/'
            })
        else:
            # If LLM fails, save basic flashcards without enhancement
            if result.get('error') == 'api_key_missing':
                saved_count = save_basic_flashcards(raw_content, folder)
                return JsonResponse({
                    'success': True,
                    'message': f'Created {saved_count} basic flashcards (AI enhancement not available)',
                    'processed_count': len(raw_content),
                    'saved_count': saved_count,
                    'redirect_url': f'/folder/{folder.id}/',
                    'warning': 'AI enhancement requires OpenAI API key configuration'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'processing_failed',
                    'message': result['message']
                })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': 'unexpected_error',
            'message': f'An unexpected error occurred: {str(e)}'
        })

def save_basic_flashcards(raw_content, folder):
    """Save basic flashcards without LLM enhancement"""
    saved_count = 0
    
    for item in raw_content:
        try:
            Flashcard.objects.create(
                folder=folder,
                question=item['question'],
                answer=item['answer'],
                source_file=item['source_file'],
                source_content=item['original_content'],
                difficulty_level='medium'
            )
            saved_count += 1
        except Exception as e:
            print(f"Error saving basic flashcard: {e}")
            continue
    
    # Mark folder as processed
    folder.processed = True
    folder.save()
    
    return saved_count

def process_pdf_complete(pdf_document):
    """Check duplicates FIRST with lightweight text extraction"""
    import hashlib
    import time
    
    start_time = time.time()
    try:
        print(f"🚀 Processing: {pdf_document.name}")
        
        # ✅ STEP 1: LIGHTWEIGHT text extraction (no concept analysis!)
        print("📄 Step 1: Extracting text for duplicate check...")
        
        from .pdf_service import PDFTextExtractor
        extractor = PDFTextExtractor()
        
        # ✅ JUST extract text, no processing
        pdf_path = pdf_document.pdf_file.path
        text, page_count = extractor.extract_text_pdfplumber(pdf_path)
        
        # if not text, try again with pypdf2
        if not text:
            text, page_count = extractor.extract_text_pypdf2(pdf_path)
        
        if not text:
            return False, "Could not extract text from PDF", {}
        
        # ✅ STEP 2: Immediate hash check
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        print(f"🔍 Hash: {content_hash[:12]}...")
        
        # ✅ STEP 3: Check for duplicates BEFORE any processing
        existing_pdf = PDFDocument.objects.filter(
            content_hash=content_hash,
            processed=True,
            focus_blocks__isnull=False
        ).exclude(id=pdf_document.id).first()
        
        if existing_pdf:
            print(f"🛑 DUPLICATE FOUND: {existing_pdf.name} - STOPPING ALL PROCESSING")
            
            # Mark as duplicate and exit immediately
            pdf_document.content_hash = content_hash
            pdf_document.page_count = page_count
            pdf_document.word_count = len(text.split())
            pdf_document.is_duplicate = True
            pdf_document.duplicate_of = existing_pdf
            pdf_document.processed = True
            pdf_document.save()
            
            focus_count = existing_pdf.focus_blocks.count()
            return True, f"📚 Content already exists as '{existing_pdf.name}' with {focus_count} focus blocks!", {}
        
        # ✅ STEP 4: No duplicates - NOW do the heavy processing
        print("🆕 Unique content - starting heavy processing...")
        
        # Store basic info
        pdf_document.extracted_text = text
        pdf_document.content_hash = content_hash
        pdf_document.page_count = page_count
        pdf_document.word_count = len(text.split())
        pdf_document.save()
        
        # ✅ STEP 5: NEW - Generate new format focus blocks directly
        blocks_success, blocks_message, focus_blocks = generate_new_format_focus_blocks(pdf_document)
        if not blocks_success:
            return False, f"New focus blocks failed: {blocks_message}", {}
        
        # ✅ STEP 6: NEW - Auto-update knowledge graph
        try:
            update_knowledge_graph_with_new_blocks(focus_blocks)
            print("🕸️ Knowledge graph updated with new blocks")
        except Exception as e:
            print(f"⚠️ Knowledge graph update failed: {str(e)} (blocks still created)")
        
        pdf_document.processed = True
       
        end_time = time.time()
        processing_duration = round(end_time - start_time, 2)
        print(f"🕐 Processing ended at: {end_time}")
        pdf_document.processing_duration = processing_duration
        print(f"💾 About to save processing_duration: {pdf_document.processing_duration}")
        pdf_document.save()
        print(f"✅ Saved! PDF processing_duration in DB: {pdf_document.processing_duration}")
        # Store processing info in cache for progress page
        from django.core.cache import cache
        processing_info = {
            'pdf_name': pdf_document.name,
            'pdf_size': pdf_document.pdf_file.size if pdf_document.pdf_file else 0,
            'page_count': pdf_document.page_count or 0,
            'word_count': pdf_document.word_count or 0,
            'total_duration': processing_duration,
            'focus_blocks_created': [
                {
                    'title': block.title,
                    'id': str(block.id),
                    'segments_count': len(block.compact7_data.get('segments', [])),
                    'difficulty': block.difficulty_level,
                    'duration': block.target_duration
                }
                for block in focus_blocks
            ],
            'success': True
        }
        cache.set(f'pdf_progress_{pdf_document.id}', processing_info, timeout=3600)


        return True, f"✅ Processed! Generated {len(focus_blocks)} new format focus blocks.", {}
        
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def generate_new_format_focus_blocks(pdf_document):
    """
    Generate new format focus blocks with segments using AI analysis
    Returns: (success, message, focus_blocks_list)
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    try:
        print(f"🧠 Generating new format focus blocks for: {pdf_document.name}")
        
        # Step 1: Chunk the text into manageable pieces
        chunks_data = generate_text_chunks_with_concepts(pdf_document.extracted_text)
        print(f"📝 Created {len(chunks_data)} text chunks")
        
        # Step 2: Group similar chunks into focus block concepts
        grouped_concepts = group_similar_concepts(chunks_data, pdf_document.name)
        print(f"🔗 Grouped into {len(grouped_concepts)} concept groups")
        
        # Step 3: Generate interactive focus blocks for each group
        focus_blocks = []
        for i, group in enumerate(grouped_concepts, 1):
            try:
                focus_block = create_interactive_focus_block_from_group(group, pdf_document, i)
                focus_blocks.append(focus_block)
                print(f"✅ Created focus block {i}: {focus_block.title[:50]}")
            except Exception as e:
                print(f"❌ Failed to create focus block {i}: {str(e)}")
                continue
        
        if not focus_blocks:
            return False, "No focus blocks could be created", []
        
        return True, f"Successfully created {len(focus_blocks)} interactive focus blocks", focus_blocks
        
    except Exception as e:
        print(f"❌ Focus block generation error: {str(e)}")
        return False, f"Failed to generate focus blocks: {str(e)}", []

def generate_text_chunks_with_concepts(text, chunk_size=300):
    """
    Split text into chunks of specified word count and prepare for concept analysis
    Returns list of chunk data with placeholders for concepts and roles
    """
    import re
    
    # Clean and split text into words
    words = text.split()
    chunks_data = []
    
    # Create chunks of specified size
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        # Basic cleaning
        chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
        
        chunk_info = {
            'chunk_id': i // chunk_size + 1,
            'text': chunk_text,
            'word_count': len(chunk_words),
            'start_word': i + 1,
            'end_word': min(i + chunk_size, len(words)),
            # Placeholders for AI analysis
            'core_concept': f"[To be analyzed - Chunk {i // chunk_size + 1}]",
            'role': 'pending_analysis',
            'confidence': 0.0,
            'keywords': [],
            'analysis_status': 'pending'
        }
        
        chunks_data.append(chunk_info)
    
    return chunks_data

def group_similar_concepts(analyzed_chunks, pdf_name, similarity_threshold=0.60):
    """
    Group chunks with similar core concepts using embeddings + AI
    Returns list of concept groups with their chunks
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    print(f"🔍 Starting concept grouping for {len(analyzed_chunks)} chunks...")
    
    # Step 1: Generate embeddings for all core concepts
    concepts = [chunk['core_concept'] for chunk in analyzed_chunks]
    
    print(f"📊 Generating embeddings for {len(concepts)} concepts...")
    print(f"📝 Concepts to analyze: {concepts[:3]}..." if len(concepts) > 3 else f"📝 Concepts: {concepts}")
    
    embeddings_response = client.embeddings.create(
        model="text-embedding-3-small",  # Efficient embedding model
        input=concepts
    )
    
    embeddings = [item.embedding for item in embeddings_response.data]
    embeddings_matrix = np.array(embeddings)
    
    # Step 2: Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    print(f"🔗 Calculated similarity matrix: {similarity_matrix.shape}")
    
    # DEBUG: Print all similarity scores
    print(f"🔍 SIMILARITY SCORES (threshold: {similarity_threshold}):")
    similar_pairs = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            similarity = similarity_matrix[i][j]
            print(f"  Chunk {analyzed_chunks[i]['chunk_id']} vs {analyzed_chunks[j]['chunk_id']}: {similarity:.3f}")
            if similarity >= similarity_threshold:
                similar_pairs.append({
                    'chunk1_id': analyzed_chunks[i]['chunk_id'],
                    'chunk2_id': analyzed_chunks[j]['chunk_id'],
                    'concept1': concepts[i],
                    'concept2': concepts[j],
                    'similarity': float(similarity)
                })
                print(f"    ✅ GROUPED: {similarity:.3f} >= {similarity_threshold}")
    
    print(f"🔗 Found {len(similar_pairs)} similar concept pairs")
    if similar_pairs:
        for pair in similar_pairs:
            print(f"  📎 Pair: '{pair['concept1'][:50]}...' + '{pair['concept2'][:50]}...' (sim: {pair['similarity']:.3f})")
    
    # Step 4: Use AI to make final grouping decisions
    if similar_pairs:
        grouped_concepts = ai_assisted_concept_grouping(analyzed_chunks, similar_pairs, pdf_name)
    else:
        print("❌ No similar pairs found - creating individual groups")
        # No similar concepts found - each chunk is its own group
        grouped_concepts = []
        for chunk in analyzed_chunks:
            grouped_concepts.append({
                'group_id': chunk['chunk_id'],
                'unified_concept': chunk['core_concept'],
                'confidence': chunk['confidence'],
                'chunks': [chunk],
                'similarity_scores': [],
                'grouping_reason': f'No similar concepts found (highest similarity was below {similarity_threshold})'
            })
    
    return grouped_concepts

def ai_assisted_concept_grouping(analyzed_chunks, similar_pairs, pdf_name):
    """
    Use AI to make intelligent grouping decisions based on similarity pairs
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Prepare data for AI analysis
    concepts_info = []
    for chunk in analyzed_chunks:
        concepts_info.append({
            'chunk_id': chunk['chunk_id'],
            'core_concept': chunk['core_concept'],
            'role': chunk['role'],
            'keywords': chunk['keywords'],
            'confidence': chunk['confidence']
        })
    
    # Prepare similarity pairs for AI
    pairs_info = []
    for pair in similar_pairs:
        pairs_info.append({
            'chunk1_id': pair['chunk1_id'],
            'chunk2_id': pair['chunk2_id'],
            'similarity_score': pair['similarity'],
            'concept1': pair['concept1'],
            'concept2': pair['concept2']
        })
    
    prompt = f"""
    You are analyzing concept groupings for the document "{pdf_name}".
    
    Below are {len(concepts_info)} chunks with their core concepts, and {len(pairs_info)} similar concept pairs.
    
    Your task: Create MULTIPLE focused concept groups for effective learning.
    
    CONCEPTS:
    {json.dumps(concepts_info, indent=2)}
    
    SIMILAR PAIRS (high semantic similarity):
    {json.dumps(pairs_info, indent=2)}
    
    CRITICAL GROUPING RULES:
    1. **CREATE 2-4 SEPARATE GROUPS** - Don't put everything in one massive group
    2. **EACH GROUP = ONE FOCUSED TOPIC** - Each group should cover ONE specific concept/topic
    3. **AIM FOR 2-4 chunks per group** - Smaller, focused groups are better than one big group
    4. **SEPARATE DIFFERENT TOPICS** - Even if related, different topics should be separate groups
    5. **ONLY group chunks about the EXACT SAME specific topic**
    6. **PREFER MULTIPLE GROUPS** over one big group
    
    EXAMPLE OF GOOD GROUPING (for 8 chunks):
    - Group 1: "HTTP Request Methods" (chunks about GET/POST/PUT) 
    - Group 2: "HTTP Status Codes" (chunks about 200/404/500)
    - Group 3: "HTTP Headers" (chunks about headers)
    
    EXAMPLE OF BAD GROUPING:
    - Group 1: "HTTP Protocols" (all 8 chunks) ❌ TOO BIG!
    
    IMPORTANT CONSTRAINTS:
    - CREATE AT LEAST 2 GROUPS (preferably 3-4 groups)
    - Each group should have 2-4 chunks maximum
    - Focus on SPECIFIC topics, not broad themes
    - If you only create 1 group, you're doing it wrong!
    
    Return JSON response:
    {{
        "concept_groups": [
            {{
                "group_id": 1,
                "unified_concept": "Specific topic name (not broad theme)",
                "chunk_ids": [1, 3],
                "confidence": 0.85,
                "grouping_reason": "Why these specific chunks form one focused topic"
            }},
            {{
                "group_id": 2,
                "unified_concept": "Another specific topic name",
                "chunk_ids": [2, 5],
                "confidence": 0.80,
                "grouping_reason": "Different focused topic"
            }}
        ]
    }}
    
    CRITICAL: You MUST create multiple groups. Single group responses will be rejected.
    """
    
    print(f"🤖 Sending {len(concepts_info)} concepts to AI for intelligent grouping...")
    print(f"📊 Similar pairs to consider: {len(pairs_info)}")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at organizing academic content. Your goal is to create cohesive, comprehensive study units by grouping related content together. Err on the side of grouping rather than separating."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=3000,
        temperature=0.2  # Slightly higher for more creative grouping
    )
    
    ai_response = response.choices[0].message.content.strip()
    
    try:
        # Parse AI grouping response
        if ai_response.startswith('```json'):
            ai_response = ai_response.replace('```json', '').replace('```', '').strip()
        
        ai_data = json.loads(ai_response)
        concept_groups_ai = ai_data.get('concept_groups', [])
        
        print(f"✅ AI created {len(concept_groups_ai)} concept groups")
        for group in concept_groups_ai:
            print(f"  📚 Group {group.get('group_id')}: '{group.get('unified_concept')}' ({len(group.get('chunk_ids', []))} chunks)")
        
        # Convert AI response to our format
        grouped_concepts = []
        for group in concept_groups_ai:
            # Get chunks for this group
            group_chunks = []
            for chunk_id in group.get('chunk_ids', []):
                for chunk in analyzed_chunks:
                    if chunk['chunk_id'] == chunk_id:
                        group_chunks.append(chunk)
                        break
            
            # Find similarity scores for this group
            group_similarities = []
            chunk_ids = group.get('chunk_ids', [])
            for pair in similar_pairs:
                if pair['chunk1_id'] in chunk_ids and pair['chunk2_id'] in chunk_ids:
                    group_similarities.append(pair['similarity'])
            
            grouped_concepts.append({
                'group_id': group.get('group_id', len(grouped_concepts) + 1),
                'unified_concept': group.get('unified_concept', 'Unknown Concept'),
                'confidence': group.get('confidence', 0.5),
                'chunks': group_chunks,
                'similarity_scores': group_similarities,
                'grouping_reason': group.get('grouping_reason', 'AI-determined grouping')
            })
        
        # ✅ ADD: Check for single massive group and split it
        if len(concept_groups_ai) == 1 and len(concept_groups_ai[0].get('chunk_ids', [])) > 4:
            print(f"⚠️ AI created single massive group - splitting it automatically")
            large_group = concept_groups_ai[0]
            chunk_ids = large_group.get('chunk_ids', [])
            
            # Split into smaller groups
            groups_to_create = min(4, (len(chunk_ids) + 2) // 3)  # 3-4 chunks per group
            chunk_size = len(chunk_ids) // groups_to_create
            
            concept_groups_ai = []
            for i in range(groups_to_create):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < groups_to_create - 1 else len(chunk_ids)
                group_chunk_ids = chunk_ids[start_idx:end_idx]
                
                concept_groups_ai.append({
                    'group_id': i + 1,
                    'unified_concept': f"{large_group['unified_concept']} - Part {i + 1}",
                    'chunk_ids': group_chunk_ids,
                    'confidence': 0.75,
                    'grouping_reason': f'Auto-split from large group (part {i + 1} of {groups_to_create})'
                })
            
            print(f"✅ Split into {len(concept_groups_ai)} focused groups")
        
        return grouped_concepts
        
    except json.JSONDecodeError as e:
        print(f"❌ AI Grouping JSON Parse Error: {e}")
        print(f"Raw AI Response: {ai_response[:500]}...")
        
        # Fallback: create individual groups
        grouped_concepts = []
        for chunk in analyzed_chunks:
            grouped_concepts.append({
                'group_id': chunk['chunk_id'],
                'unified_concept': chunk['core_concept'],
                'confidence': chunk['confidence'],
                'chunks': [chunk],
                'similarity_scores': [],
                'grouping_reason': f'Fallback grouping due to AI error: {str(e)}'
            })
        
        return grouped_concepts

def create_interactive_focus_block_from_group(group, pdf_document, block_order):
    """
    Create an interactive focus block from a concept group using AI
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    from django.db import models
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise Exception("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    # Prepare chunk content for AI
    chunk_content = []
    for chunk in group['chunks']:
        chunk_content.append({
            'text': chunk['text'],
            'role': chunk.get('role', 'content'),
            'concept': chunk.get('core_concept', 'Unknown')
        })
    
    # [Keep the existing AI prompt - same as before]
    prompt = f"""
    Based on the following grouped concept chunks:
    {json.dumps(chunk_content, indent=2, default=str)}
    
    Create a new format focus block with this structure:
    {{
        "title": "Shortest complete concept name (2-7 words, prefer minimal)",
        "learning_objectives": ["objective 1", "objective 2", "objective 3"],
        "segments": [
            {{
                "type": "recap",
                "title": "Quick Review",
                "content": "Brief recap of any prerequisites",
                "duration_seconds": 60
            }},
            {{
                "type": "socratic_intro",
                "title": "Discovering the Concept",
                "content": "Socratic questions to guide understanding",
                "duration_seconds": 120
            }},
            {{
                "type": "definition",
                "title": "Core Definition",
                "content": "Clear, precise definition",
                "duration_seconds": 90
            }},
            {{
                "type": "example",
                "title": "Concrete Example",
                "content": "Real-world example with details",
                "duration_seconds": 120
            }},
            {{
                "type": "practice",
                "title": "Apply It",
                "content": "Practice problem or exercise",
                "duration_seconds": 90
            }}
        ],
        "qa_items": [
            {{
                "question": "Understanding check question about the core concept",
                "answer": "Comprehensive answer explaining the concept",
                "difficulty": "basic"
            }},
            {{
                "question": "Application question requiring practical thinking",
                "answer": "Detailed answer with real-world application",
                "difficulty": "intermediate"
            }},
            {{
                "question": "Synthesis question connecting to other concepts",
                "answer": "Advanced answer showing deeper understanding",
                "difficulty": "advanced"
            }}
        ],
        "total_duration": 480,
        "difficulty_level": "intermediate"
    }}
    
    Guidelines:
    - TITLE: 2-7 words. PREFER SHORTEST COMPLETE TITLE. Core concept only. NO academic phrases like "Understanding", "Analysis", "Study", "Guide", "Introduction", "Overview", "Comprehensive"
    - GOOD EXAMPLES: "HTTP Protocols", "Linear Regression", "Market Supply and Demand", "Cell Mitosis Process"
    - BAD EXAMPLES: "Understanding HTTP Protocols"
    - Use Socratic method for engagement
    - Make content interactive and time-bounded
    - Base content on the provided chunks
    - Create clear learning progression
    - Generate 3 Q&A items of increasing difficulty
    - Return ONLY valid JSON
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are an expert educational content designer. Create engaging, interactive focus blocks."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=2000,
            temperature=0.3
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Clean response
        if response_content.startswith('```json'):
            response_content = response_content[7:]
        if response_content.endswith('```'):
            response_content = response_content[:-3]
        
        block_data = json.loads(response_content.strip())
        
        # ✅ FIXED: Create concept unit with correct field names
        max_order = ConceptUnit.objects.filter(pdf_document=pdf_document).aggregate(
            max_order=models.Max('concept_order')
        )['max_order']
        next_order = (max_order or 0) + 1
        
        concept_unit = ConceptUnit.objects.create(
            pdf_document=pdf_document,
            title=group['unified_concept'],  # ✅ FIXED: Use 'title' not 'concept_title'
            concept_order=next_order,
            complexity_score=group.get('confidence', 0.75)  # ✅ FIXED: Use 'complexity_score'
        )
        
        # Create focus block
        focus_block = FocusBlock.objects.create(
            pdf_document=pdf_document,
            main_concept_unit=concept_unit,
            block_order=block_order,
            title=block_data.get('title', group['unified_concept']),
            target_duration=block_data.get('total_duration', 420),
            compact7_data=block_data,
            difficulty_level=block_data.get('difficulty_level', 'intermediate'),
            learning_objectives=block_data.get('learning_objectives', [])
        )
        
        return focus_block
        
    except Exception as e:
        raise Exception(f"Failed to create interactive focus block: {str(e)}")

def update_knowledge_graph_with_new_blocks(new_focus_blocks):
    """
    Update the knowledge graph to include relationships with newly created blocks
    """
    try:
        if not new_focus_blocks:
            print("📊 No new focus blocks to process")
            return

        print(f"🕸️ Updating knowledge graph with {len(new_focus_blocks)} new blocks...")
        
        # Get all existing focus blocks (including new ones)
        all_blocks = list(FocusBlock.objects.filter(
            compact7_data__has_key='segments'
        ).exclude(
            title__startswith='[MIGRATED→'
        ))
        
        if len(all_blocks) < 2:
            print("📊 Not enough blocks for knowledge graph (need at least 2)")
            return
        
        # Generate relationships for new blocks with all existing blocks
        new_relationships = []
        
        for new_block in new_focus_blocks:
            # Find relationships between this new block and all other blocks
            other_blocks = [b for b in all_blocks if b.id != new_block.id]
            
            if other_blocks:
                relationships = analyze_new_block_relationships(new_block, other_blocks)
                new_relationships.extend(relationships)
        
        # Store relationships in database
        created_count = 0
        for rel_data in new_relationships:
            try:
                relationship, created = FocusBlockRelationship.objects.get_or_create(
                    from_block_id=rel_data['from_block_id'],
                    to_block_id=rel_data['to_block_id'],
                    relationship_type=rel_data['relationship_type'],
                    defaults={
                        'confidence': rel_data['confidence'],
                        'similarity_score': rel_data.get('similarity_score', 0.0),
                        'edge_strength': rel_data.get('edge_strength', rel_data['confidence']),
                        'description': rel_data['description'],
                        'educational_reasoning': rel_data.get('educational_reasoning', '')
                    }
                )
                
                if created:
                    created_count += 1
                    
                    # For bidirectional relationships, create reverse relationship
                    if rel_data['relationship_type'] in ['related', 'compares_with']:
                        reverse_rel, reverse_created = FocusBlockRelationship.objects.get_or_create(
                            from_block_id=rel_data['to_block_id'],
                            to_block_id=rel_data['from_block_id'],
                            relationship_type=rel_data['relationship_type'],
                            defaults={
                                'confidence': rel_data['confidence'],
                                'similarity_score': rel_data.get('similarity_score', 0.0),
                                'edge_strength': rel_data.get('edge_strength', rel_data['confidence']),
                                'description': f"Reverse: {rel_data['description']}",
                                'educational_reasoning': rel_data.get('educational_reasoning', '')
                            }
                        )
                        if reverse_created:
                            created_count += 1
                            
            except Exception as e:  # ✅ FIXED: Proper indentation to match the 'try'
                print(f"❌ Failed to create relationship: {str(e)}")
                continue
        
        print(f"🕸️ Knowledge graph updated: {created_count} new relationships created")
        
        # Optional: Clean up low-confidence relationships periodically
        cleanup_weak_relationships()
        
    except Exception as e:
        print(f"❌ Knowledge graph update failed: {str(e)}")
        # Don't fail the whole process if knowledge graph fails


def analyze_new_block_relationships(new_block, existing_blocks):
    """
    Analyze relationships between a new focus block and existing blocks
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import uuid
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        print("⚠️ OpenAI API key not configured, skipping knowledge graph update")
        return []
    
    client = OpenAI(api_key=api_key)
    
    try:
        # Add debug prints
        print(f"🔍 Analyzing relationships for: {new_block.title}")
        print(f"🔍 Against {len(existing_blocks)} other blocks")
        
        # Step 1: Get embeddings for the new block and existing blocks
        all_blocks = [new_block] + existing_blocks
        block_texts = []
        
        for block in all_blocks:
            # Create comprehensive text representation
            text_parts = [block.title]
            
            # Add content from compact7_data if available
            if hasattr(block, 'compact7_data') and block.compact7_data:
                segments = block.compact7_data.get('segments', [])
                for segment in segments:
                    if 'content' in segment:
                        text_parts.append(segment['content'][:200])  # Limit content length
                    if 'explanation' in segment:
                        text_parts.append(segment['explanation'][:200])
            
            # Add learning objectives
            if block.learning_objectives:
                text_parts.extend(block.learning_objectives)
            
            block_texts.append(' '.join(text_parts))
        
        # Get embeddings with better error handling
        try:
            embeddings_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=block_texts
            )
            print(f"✅ Got embeddings for {len(block_texts)} blocks")
        except Exception as e:
            print(f"❌ Embedding API failed: {str(e)}")
            return []
        
        embeddings = [item.embedding for item in embeddings_response.data]
        embeddings_matrix = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Step 2: Create simplified block mapping for AI
        # Use simple indices instead of UUIDs to avoid truncation
        block_mapping = {}
        existing_blocks_simple = []
        
        for i, existing_block in enumerate(existing_blocks):
            block_id = f"block_{i}"
            block_mapping[block_id] = str(existing_block.id)
            
            existing_blocks_simple.append({
                'id': block_id,  # Simple ID for AI
                'title': existing_block.title,
                'difficulty_level': existing_block.difficulty_level,
            })
        
        new_block_simple = {
            'id': 'new_block',
            'title': new_block.title,
            'difficulty_level': new_block.difficulty_level,
        }
        
        # Step 3: Simplified AI analysis for relationships
        prompt = f"""
        Analyze educational relationships between blocks.
        
        NEW BLOCK:
        - ID: new_block
        - Title: {new_block.title}
        - Level: {new_block.difficulty_level}
        
        EXISTING BLOCKS:
        {chr(10).join([f"- ID: {block['id']}, Title: {block['title']}, Level: {block['difficulty_level']}" for block in existing_blocks_simple[:5]])}
        
        Return JSON with relationships. Use EXACT IDs shown above:
        {{
            "relationships": [
                {{
                    "from_block_id": "block_0",
                    "to_block_id": "new_block",
                    "relationship_type": "prerequisite",
                    "confidence": 0.8,
                    "description": "Short description",
                    "educational_reasoning": "Why this helps learning"
                }}
            ]
        }}
        
        Rules:
        - Use ONLY the IDs shown above: {', '.join([b['id'] for b in existing_blocks_simple[:5]])} and "new_block"
        - Relationship types: prerequisite, builds_on, related, applies_to
        - Only confidence >= 0.5
        - Maximum 3 relationships
        - Return valid JSON only
        """
        
        print(f"🤖 Analyzing relationships for new block: {new_block.title}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800  # Limit tokens
            )
            
            # Debug the response
            response_content = response.choices[0].message.content
            print(f"🔍 AI returned {len(response_content)} characters")
            
            if not response_content or response_content.strip() == "":
                print("❌ Empty response from OpenAI")
                return []
                
        except Exception as e:
            print(f"❌ OpenAI API call failed: {str(e)}")
            return []
        
        # Parse response with better error handling
        try:
            # Clean the response (remove any markdown formatting)
            cleaned_content = response_content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content.replace('```json', '').replace('```', '').strip()
            elif cleaned_content.startswith('```'):
                cleaned_content = cleaned_content.replace('```', '').strip()
            
            relationships_data = json.loads(cleaned_content)
            relationships = relationships_data.get('relationships', [])
            
            # Convert simple IDs back to actual UUIDs and validate
            validated_relationships = []
            for rel in relationships:
                try:
                    # Convert from simple IDs to actual UUIDs
                    from_id = rel['from_block_id']
                    to_id = rel['to_block_id']
                    
                    if from_id == 'new_block':
                        actual_from_id = str(new_block.id)
                    elif from_id in block_mapping:
                        actual_from_id = block_mapping[from_id]
                    else:
                        print(f"❌ Invalid from_block_id: {from_id}")
                        continue
                    
                    if to_id == 'new_block':
                        actual_to_id = str(new_block.id)
                    elif to_id in block_mapping:
                        actual_to_id = block_mapping[to_id]
                    else:
                        print(f"❌ Invalid to_block_id: {to_id}")
                        continue
                    
                    # Validate UUIDs
                    uuid.UUID(actual_from_id)  # Will raise ValueError if invalid
                    uuid.UUID(actual_to_id)
                    
                    # Validate that blocks exist
                    if not FocusBlock.objects.filter(id=actual_from_id).exists():
                        print(f"❌ from_block not found: {actual_from_id}")
                        continue
                    if not FocusBlock.objects.filter(id=actual_to_id).exists():
                        print(f"❌ to_block not found: {actual_to_id}")
                        continue
                    
                    validated_rel = {
                        'from_block_id': actual_from_id,
                        'to_block_id': actual_to_id,
                        'relationship_type': rel['relationship_type'],
                        'confidence': rel['confidence'],
                        'description': rel['description'],
                        'educational_reasoning': rel.get('educational_reasoning', ''),
                        'similarity_score': 0.0,
                        'edge_strength': rel['confidence']
                    }
                    validated_relationships.append(validated_rel)
                    print(f"✅ Validated relationship: {rel['relationship_type']}")
                    
                except (ValueError, KeyError) as e:
                    print(f"❌ Invalid relationship data: {str(e)}")
                    continue
            
            print(f"✅ Generated {len(validated_relationships)} valid relationships")
            return validated_relationships
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse AI response: {str(e)}")
            print(f"❌ Raw response: {response_content[:200]}...")
            return []
        
    except Exception as e:
        print(f"❌ Error analyzing new block relationships: {str(e)}")
        return []


def cleanup_weak_relationships(min_confidence=0.3):
    """
    Clean up relationships with very low confidence scores
    """
    try:
        weak_relationships = FocusBlockRelationship.objects.filter(confidence__lt=min_confidence)
        count = weak_relationships.count()
        if count > 0:
            weak_relationships.delete()
            print(f"🧹 Cleaned up {count} weak relationships (confidence < {min_confidence})")
    except Exception as e:
        print(f"❌ Error cleaning up relationships: {str(e)}")

def check_for_duplicate_content(pdf_document: PDFDocument) -> Optional[PDFDocument]:
    """Ultra-fast duplicate detection using stored extracted text"""
    import hashlib
    
    # ✅ Use the ALREADY extracted text (not extract again!)
    if not pdf_document.extracted_text:
        print("❌ No extracted text available for hash calculation")
        return None
    
    # Calculate hash from the SAME text we just extracted
    content_hash = hashlib.sha256(pdf_document.extracted_text.encode('utf-8')).hexdigest()
    
    # Store the hash
    pdf_document.content_hash = content_hash
    pdf_document.save()
    
    print(f"🔍 Calculated hash: {content_hash[:12]}... for PDF: {pdf_document.name}")
    
    # ✅ Check for duplicates (exclude current PDF)
    duplicate = PDFDocument.objects.filter(
        content_hash=content_hash,
        processed=True,
        focus_blocks__isnull=False  # Only consider PDFs with focus blocks
    ).exclude(id=pdf_document.id).first()
    
    if duplicate:
        print(f"✅ DUPLICATE FOUND: {duplicate.name} (hash: {duplicate.content_hash[:12]}...)")
        return duplicate
    
    print("🆕 No duplicates found - this is unique content")
    return None


def copy_pdf_data(source_pdf: PDFDocument, target_pdf: PDFDocument) -> Dict:
    """Copy all data from source PDF to target PDF"""
    from .models import TextChunk, ConceptUnit, ChunkLabel, FocusBlock
    
    print(f"📋 Copying data from '{source_pdf.name}' to '{target_pdf.name}'")
    
    # ✅ CLEAR existing data first to avoid constraint violations
    print("🧹 Clearing existing data...")
    target_pdf.text_chunks.all().delete()
    target_pdf.concept_units.all().delete() 
    target_pdf.focus_blocks.all().delete()
    
    # Copy basic info
    target_pdf.cleaned_text = source_pdf.cleaned_text
    target_pdf.processed = True
    target_pdf.is_duplicate = True
    target_pdf.duplicate_of = source_pdf
    target_pdf.save()
    
    print("📄 Copying text chunks...")
    # Copy text chunks
    chunks_copied = 0
    source_chunks = source_pdf.text_chunks.all().order_by('chunk_order')
    chunk_mapping = {}
    
    for chunk in source_chunks:
        new_chunk = TextChunk.objects.create(
            pdf_document=target_pdf,
            chunk_text=chunk.chunk_text,
            chunk_order=chunk.chunk_order,
            word_start=chunk.word_start,
            word_end=chunk.word_end,
            word_count=chunk.word_count
        )
        chunk_mapping[chunk.id] = new_chunk
        chunks_copied += 1
    
    print(f"✅ Copied {chunks_copied} text chunks")
    
    # Copy concept units and labels
    print("🧠 Copying concept units...")
    concepts_copied = 0
    for concept in source_pdf.concept_units.all().order_by('concept_order'):
        new_concept = ConceptUnit.objects.create(
            pdf_document=target_pdf,
            title=concept.title,
            description=concept.description,
            concept_order=concept.concept_order,
            primary_labels=concept.primary_labels,
            estimated_reading_time=concept.estimated_reading_time,
            word_count=concept.word_count
        )
        
        # Copy chunk labels
        for label in concept.chunk_labels.all():
            if label.text_chunk.id in chunk_mapping:
                ChunkLabel.objects.create(
                    text_chunk=chunk_mapping[label.text_chunk.id],
                    concept_unit=new_concept,
                    label_type=label.label_type,
                    confidence_score=label.confidence_score,
                    concept_keywords=label.concept_keywords
                )
        
        concepts_copied += 1
    
    print(f"✅ Copied {concepts_copied} concept units")
    
    # Copy focus blocks
    print("🎯 Copying focus blocks...")
    focus_blocks_copied = 0
    for focus_block in source_pdf.focus_blocks.all().order_by('block_order'):
        new_focus_block = FocusBlock.objects.create(
            pdf_document=target_pdf,
            main_concept_unit=None,  # Will link after concept units are created
            title=focus_block.title,
            block_order=focus_block.block_order,
            estimated_duration_minutes=focus_block.estimated_duration_minutes,
            compact7_data=focus_block.compact7_data,
            teacher_script=focus_block.teacher_script,
            rescue_reset=focus_block.rescue_reset,
            recap_summary=focus_block.recap_summary
        )
        focus_blocks_copied += 1
    
    print(f"✅ Copied {focus_blocks_copied} focus blocks")
    
    return {
        'chunks_copied': chunks_copied,
        'concepts_copied': concepts_copied,
        'focus_blocks_copied': focus_blocks_copied,
        'duplicate_source': source_pdf.name
    }

def study_flashcards(request, folder_id):
    """View to study flashcards with pagination"""
    folder = get_object_or_404(Folder, id=folder_id)
    
    if not folder.processed:
        messages.warning(request, 'This folder has not been processed yet.')
        return redirect('flashcards:process_folder', folder_id=folder.id)
    
    # Get or create study session
    session_id = request.session.get(f'study_session_{folder.id}')
    study_session = None
    
    if session_id:
        try:
            study_session = StudySession.objects.get(session_id=session_id, folder=folder)
        except StudySession.DoesNotExist:
            pass
    
    if not study_session:
        study_session = StudySession.objects.create(folder=folder)
        request.session[f'study_session_{folder.id}'] = str(study_session.session_id)
    
    # Get randomized flashcards (exclude already shown ones for variety)
    shown_cards = study_session.cards_shown.all()
    available_cards = folder.flashcards.exclude(id__in=shown_cards.values_list('id', flat=True))
    
    # If all cards have been shown, reset and use all cards
    if not available_cards.exists():
        available_cards = folder.flashcards.all()
        study_session.cards_shown.clear()
    
    # Randomize and paginate
    cards = available_cards.order_by('?')  # Random ordering
    paginator = Paginator(cards, study_session.cards_per_page)
    
    page_number = request.GET.get('page', study_session.current_page)
    page_obj = paginator.get_page(page_number)
    
    # Update study session
    study_session.current_page = page_obj.number
    study_session.save()
    
    # Mark these cards as shown
    for card in page_obj:
        study_session.cards_shown.add(card)
    
    # Calculate progress percentage
    progress_percentage = (page_obj.number / page_obj.paginator.num_pages) * 100 if page_obj.paginator.num_pages > 0 else 0
    
    # Safe page navigation numbers
    prev_page = page_obj.previous_page_number() if page_obj.has_previous() else None
    next_page = page_obj.next_page_number() if page_obj.has_next() else None
    
    # JSON data for JavaScript
    navigation_data = json.dumps({
        'prevPage': prev_page,
        'nextPage': next_page
    })
    
    context = {
        'folder': folder,
        'page_obj': page_obj,
        'study_session': study_session,
        'progress_percentage': progress_percentage,
        'navigation_data': navigation_data,
    }
    return render(request, 'flashcards/study.html', context)

@require_http_methods(["POST"])
def delete_folder(request, folder_id):
    """Delete a folder and all its flashcards"""
    folder = get_object_or_404(Folder, id=folder_id)
    folder_name = folder.name
    total_flashcards = folder.flashcards.count()
    
    # Delete the folder (cascades to flashcards automatically)
    folder.delete()
    
    messages.success(
        request, 
        f'Folder "{folder_name}" and {total_flashcards} flashcards deleted successfully.'
    )
    return redirect('flashcards:folder_list')

@require_http_methods(["POST"])
def delete_flashcard(request, flashcard_id):
    """Delete a single flashcard"""
    flashcard = get_object_or_404(Flashcard, id=flashcard_id)
    folder = flashcard.folder
    question_preview = flashcard.question[:50] + "..." if len(flashcard.question) > 50 else flashcard.question
    
    flashcard.delete()
    
    messages.success(
        request, 
        f'Flashcard "{question_preview}" deleted successfully.'
    )
    return redirect('flashcards:folder_detail', folder_id=folder.id)

@require_http_methods(["POST"])
def bulk_delete_flashcards(request, folder_id):
    """Delete multiple flashcards at once"""
    folder = get_object_or_404(Folder, id=folder_id)
    flashcard_ids = request.POST.getlist('flashcard_ids')
    
    if flashcard_ids:
        deleted_count = Flashcard.objects.filter(
            id__in=flashcard_ids, 
            folder=folder
        ).delete()[0]
        
        messages.success(
            request, 
            f'Successfully deleted {deleted_count} flashcards.'
        )
    else:
        messages.warning(request, 'No flashcards selected for deletion.')
    
    return redirect('flashcards:folder_detail', folder_id=folder.id)

def pdf_processing_details(request, pdf_id):
    """Detailed view showing the complete processing pipeline for a PDF"""
    pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
    
    # Get all processing data
    text_chunks = pdf_document.text_chunks.all().order_by('chunk_order')
    chunk_labels = ChunkLabel.objects.filter(text_chunk__in=text_chunks).select_related('text_chunk', 'concept_unit')
    concept_units = pdf_document.concept_units.all().order_by('concept_order')
    
    # ✅ Add debugging information
    debug_info = {
        'chunks_exist': text_chunks.exists(),
        'chunk_labels_exist': chunk_labels.exists(),
        'concept_units_exist': concept_units.exists(),
        'concepts_analyzed_flag': pdf_document.concept_units.exists(),
        'chunks_with_labels': chunk_labels.count(),
        'chunks_without_labels': text_chunks.exclude(chunk_label__isnull=False).count(),
        'concept_units_with_chunks': concept_units.filter(chunk_labels__isnull=False).distinct().count(),
        'orphaned_concept_units': concept_units.filter(chunk_labels__isnull=True).count(),
    }
    
    # Calculate statistics
    processing_stats = {
        'total_chunks': text_chunks.count(),
        'labeled_chunks': chunk_labels.count(),
        'total_concept_units': concept_units.count(),
        'optimized_units': concept_units.filter(time_optimized=True).count(),
        'total_words': pdf_document.word_count,
        'total_estimated_time': sum(unit.estimated_reading_time for unit in concept_units),
        'avg_time_per_unit': sum(unit.estimated_reading_time for unit in concept_units) / len(concept_units) if concept_units else 0,
        'debug_info': debug_info,  # ✅ Add debug info
    }
    
    # Group data for detailed breakdown
    concept_breakdown = []
    for unit in concept_units:
        unit_chunks = chunk_labels.filter(concept_unit=unit).order_by('text_chunk__chunk_order')
        
        concept_info = {
            'unit': unit,
            'chunks': unit_chunks,
            'chunk_count': unit_chunks.count(),
            'labels_used': list(set(chunk.primary_label for chunk in unit_chunks)),
            'keywords': list(set([kw for chunk in unit_chunks for kw in chunk.concept_keywords])),
            'avg_confidence': sum(chunk.confidence_score for chunk in unit_chunks) / len(unit_chunks) if unit_chunks else 0,
            'has_chunks': unit_chunks.exists(),  # ✅ Debug flag
        }
        concept_breakdown.append(concept_info)
    
    # Calculate label distribution with percentages in view
    label_distribution = {}
    total_labeled = processing_stats['labeled_chunks']
    
    for chunk_label in chunk_labels:
        label = chunk_label.primary_label
        if label not in label_distribution:
            label_distribution[label] = {'count': 0, 'percentage': 0}
        label_distribution[label]['count'] += 1
    
    # Calculate percentages
    for label_data in label_distribution.values():
        if total_labeled > 0:
            label_data['percentage'] = round((label_data['count'] * 100) / total_labeled, 1)
        else:
            label_data['percentage'] = 0
    
    # Sort by count (descending)
    label_distribution = dict(sorted(label_distribution.items(), key=lambda x: x[1]['count'], reverse=True))
    
    context = {
        'pdf_document': pdf_document,
        'processing_stats': processing_stats,
        'concept_breakdown': concept_breakdown,
        'label_distribution': label_distribution,
        'text_chunks': text_chunks,
    }
    
    return render(request, 'flashcards/pdf_processing_details.html', context)

def chunk_detail(request, chunk_id):
    """Detailed view for a specific text chunk"""
    chunk = get_object_or_404(TextChunk, id=chunk_id)
    
    try:
        chunk_label = chunk.chunk_label
    except ChunkLabel.DoesNotExist:
        chunk_label = None
    
    context = {
        'chunk': chunk,
        'chunk_label': chunk_label,
        'pdf_document': chunk.pdf_document,
    }
    
    return render(request, 'flashcards/chunk_detail.html', context)

def generate_focus_blocks_view(request, pdf_id):
    """Generate focus blocks for a PDF"""
    pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
    
    if request.method == 'POST':
        try:
            success, message = generate_focus_blocks(pdf_document)
            if success:
                messages.success(request, message)
            else:
                messages.error(request, f"Focus block generation failed: {message}")
        except Exception as e:
            messages.error(request, f"Error generating focus blocks: {str(e)}")
    
    return redirect('flashcards:pdf_focus_blocks', pdf_id=pdf_id)

def pdf_focus_blocks(request, pdf_id):
    """View all focus blocks for a PDF"""
    pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
    focus_blocks = pdf_document.focus_blocks.all().order_by('block_order')
    
    # Calculate total study time
    total_duration = sum(block.target_duration for block in focus_blocks)
    
    context = {
        'pdf_document': pdf_document,
        'focus_blocks': focus_blocks,
        'total_duration': total_duration,
        'total_blocks': focus_blocks.count(),
    }
    
    return render(request, 'flashcards/pdf_focus_blocks.html', context)

def focus_block_detail(request, block_id):
    """Detailed view of a single focus block"""
    focus_block = get_object_or_404(FocusBlock, id=block_id)
    qa_items = focus_block.get_qa_items()  # Use Compact7 JSON format
    
    context = {
        'focus_block': focus_block,
        'qa_items': qa_items,
        'pdf_document': focus_block.pdf_document,
    }
    
    return render(request, 'flashcards/focus_block_detail.html', context)

def study_focus_blocks(request, pdf_id):
    """Start or continue a focus block study session"""
    pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
    
    # Get or create study session
    session_id = request.session.get(f'study_session_{pdf_id}')
    if session_id:
        try:
            study_session = StudySession.objects.get(session_id=session_id)
        except StudySession.DoesNotExist:
            study_session = None
    else:
        study_session = None
    
    if not study_session:
        study_session = StudySession.objects.create(
            pdf_document=pdf_document,
            session_type='focus_blocks'
        )
        request.session[f'study_session_{pdf_id}'] = str(study_session.session_id)
    
    # Get current focus block
    if not study_session.current_focus_block:
        # Start with first block
        first_block = pdf_document.focus_blocks.first()
        if first_block:
            study_session.current_focus_block = first_block
            study_session.save()
    
    current_block = study_session.current_focus_block
    if not current_block:
        messages.error(request, "No focus blocks found. Generate focus blocks first.")
        return redirect('flashcards:pdf_focus_blocks', pdf_id=pdf_id)
    
    # Get progress information
    total_blocks = pdf_document.focus_blocks.count()
    completed_blocks = study_session.completed_focus_blocks.count()
    progress_percentage = study_session.get_progress_percentage()
    
    context = {
        'pdf_document': pdf_document,
        'study_session': study_session,
        'current_block': current_block,
        'qa_items': current_block.get_qa_items(),  # Use Compact7 JSON format
        'total_blocks': total_blocks,
        'completed_blocks': completed_blocks,
        'progress_percentage': progress_percentage,
    }
    
    return render(request, 'flashcards/study_focus_blocks.html', context)

def complete_focus_block(request, session_id, block_id):
    """Mark a focus block as completed and move to next"""
    study_session = get_object_or_404(StudySession, session_id=session_id)
    focus_block = get_object_or_404(FocusBlock, id=block_id)
    
    if request.method == 'POST':
        # Mark current block as completed
        study_session.completed_focus_blocks.add(focus_block)
        
        # Move to next block
        next_block = study_session.pdf_document.focus_blocks.filter(
            block_order__gt=focus_block.block_order
        ).first()
        
        if next_block:
            study_session.current_focus_block = next_block
            study_session.save()
            messages.success(request, f"Completed {focus_block.title}! Moving to next block.")
        else:
            # All blocks completed
            study_session.current_focus_block = None
            study_session.save()
            messages.success(request, "Congratulations! You've completed all focus blocks!")
        
        return redirect('flashcards:study_focus_blocks', pdf_id=study_session.pdf_document.id)
    
    return redirect('flashcards:study_focus_blocks', pdf_id=study_session.pdf_document.id)

def concept_unit_detail(request, unit_id):
    """Detailed view of a single concept unit with full text"""
    concept_unit = get_object_or_404(ConceptUnit, id=unit_id)
    
    # Get all chunks in this concept unit
    chunk_labels = concept_unit.chunk_labels.select_related('text_chunk').order_by('text_chunk__chunk_order')
    
    # Get combined text
    combined_text = concept_unit.get_combined_text()
    
    # Calculate reading statistics
    word_count = len(combined_text.split()) if combined_text else 0
    char_count = len(combined_text)
    
    # Get related focus blocks
    focus_blocks = concept_unit.main_focus_blocks.all()
    
    context = {
        'concept_unit': concept_unit,
        'chunk_labels': chunk_labels,
        'combined_text': combined_text,
        'word_count': word_count,
        'char_count': char_count,
        'focus_blocks': focus_blocks,
        'pdf_document': concept_unit.pdf_document,
    }
    
    return render(request, 'flashcards/concept_unit_detail.html', context)

def focus_block_debug(request, pdf_id):
    """Debug view to inspect focus block content"""
    pdf_document = get_object_or_404(PDFDocument, id=pdf_id)
    focus_blocks = pdf_document.focus_blocks.all().order_by('block_order')
    
    debug_info = []
    for block in focus_blocks:
        info = {
            'block': block,
            'has_compact7_data': bool(block.compact7_data and block.compact7_data != {}),
            'compact7_keys': list(block.compact7_data.keys()) if block.compact7_data else [],
            'segments_count': len(block.get_segments()),
            'qa_count': len(block.get_qa_items()),
            'has_revision': bool(block.get_revision_data()),
            'has_recap': bool(block.get_recap_data()),
            'has_rescue': bool(block.get_rescue_data()),
            'core_goal': block.get_core_goal()[:100] + '...' if len(block.get_core_goal()) > 100 else block.get_core_goal(),
        }
        debug_info.append(info)
    
    context = {
        'pdf_document': pdf_document,
        'debug_info': debug_info,
        'total_blocks': len(focus_blocks),
    }
    
    return render(request, 'flashcards/focus_block_debug.html', context)

def deduplication_stats(request):
    """Show deduplication statistics and management"""
    # from .deduplication_service import PDFDeduplicationService
    # from .deduplication_service import check_pdf_duplicates
    
    # service = PDFDeduplicationService()
    # stats = service.get_deduplication_stats()
    
    # # Get duplicate groups
    # originals_with_dupes = PDFDocument.objects.filter(
    #     duplicates__isnull=False
    # ).distinct().prefetch_related('duplicates')
    
    # duplicate_groups = []
    # for original in originals_with_dupes:
    #     duplicates = list(original.duplicates.all())
    #     duplicate_groups.append({
    #         'original': original,
    #         'duplicates': duplicates,
    #         'total_size_saved': sum(dup.file_size for dup in duplicates),
    #         'avg_similarity': sum(dup.similarity_score for dup in duplicates) / len(duplicates)
    #     })
    
    context = {
        # 'stats': stats,
        # 'duplicate_groups': duplicate_groups,
    }
    
    return render(request, 'flashcards/deduplication_stats.html', context)

def all_focus_blocks(request):
    """Unified advanced study interface for all focus blocks"""
    
    all_blocks = FocusBlock.objects.select_related('pdf_document').order_by(
        'pdf_document__created_at', 'block_order'
    )

    if not all_blocks.exists():
        return render(request, 'flashcards/all_focus_blocks.html', {'no_blocks': True})

    # Create a single focus session for all blocks if needed
    active_session = None
    if all_blocks.exists():
        # Get or create a session for the first block (we'll use it for session-wide tracking)
        first_block = all_blocks.first()
        active_session, created = FocusSession.objects.get_or_create(
            focus_block=first_block,
            status__in=['active', 'paused'],
            defaults={
                'status': 'active',
                'current_segment': 0,
            }
        )
        
        if created:
            active_session.block_start_times = {str(first_block.id): timezone.now().isoformat()}
            active_session.save()
            print(f"✅ Created new unified FocusSession: {active_session.id}")
        
        # ✅ ADD: Handle completed sessions
        if active_session:
            print(f"🔍 Session segments_completed: {active_session.segments_completed}")
            print(f"🔍 Current segment: {active_session.current_segment}")
            
            # Check what segments should be shown
            current_block = active_session.focus_block
            total_segments_in_current_block = len(current_block.get_segments())
            
            print(f"🔍 Total segments in current block: {total_segments_in_current_block}")
            print(f"🔍 Completed segments: {len(active_session.segments_completed)}")

            # Handle completed sessions
            if len(active_session.segments_completed) >= total_segments_in_current_block:
                print("⚠️ Current block completed - finding next block!")
                
                # Find next uncompleted block
                remaining_blocks = all_blocks.filter(
                    models.Q(pdf_document__created_at__gt=current_block.pdf_document.created_at) |
                    models.Q(
                        pdf_document__created_at=current_block.pdf_document.created_at,
                        block_order__gt=current_block.block_order
                    )
                ).order_by('pdf_document__created_at', 'block_order')
                
                if remaining_blocks.exists():
                    # Move to next block
                    next_block = remaining_blocks.first()
                    print(f"🎯 Moving to next block: {next_block.title}")
                    
                    # Create new session for next block
                    active_session = FocusSession.objects.create(
                        focus_block=next_block,
                        status='active',
                        current_segment=0,
                        segments_completed=[]
                    )
                    print(f"✅ Created new session for next block: {active_session.id}")
                else:
                    # All blocks completed!
                    print("🎉 All blocks completed!")
                    return render(request, 'flashcards/all_focus_blocks.html', {
                        'all_completed': True,
                        'total_blocks': len(all_blocks),
                        'message': 'Congratulations! You have completed all focus blocks!'
                    })

        print(f"Created new unified FocusSession: {created}")
    
    # Format blocks for advanced study
    formatted_blocks = []
    for block in all_blocks:
        # Get segments using the model method (now fixed to handle new data structure)
        segments = block.get_segments()
        qa_items = block.get_qa_items()
        learning_objectives = block.compact7_data.get('learning_objectives', []) if block.compact7_data else []
        
        print(f"🔍 Block: {block.title}")
        print(f"   Segments: {len(segments)}")
        print(f"   Q&A Items: {len(qa_items)}")
        print(f"   Learning Objectives: {len(learning_objectives)}")
        
        formatted_blocks.append({
            'block': block,
            'segments': segments,
            'qa_items': qa_items,
            'learning_objectives': learning_objectives,
            'source_pdf': block.pdf_document.name,
            'total_segments': len(segments),
            'total_qa': len(qa_items),
            'estimated_duration': block.get_estimated_duration_display(),
        })
    
    total_time = sum(block.target_duration or 420 for block in all_blocks) / 60
    
    context = {
        'focus_blocks': formatted_blocks,
        'focus_session': active_session,
        'total_blocks': len(formatted_blocks),
        'total_study_time': total_time,
        'unique_pdfs': len(set(block.pdf_document.name for block in all_blocks)),
        'progress_percentage': active_session.get_completion_percentage() if active_session else 0,
    }
    
    return render(request, 'flashcards/all_focus_blocks.html', context)


def bulk_upload(request):
    """Handle bulk PDF upload"""
    if request.method == 'POST':
        files = request.FILES.getlist('pdf_files')
        if not files:
            messages.error(request, 'Please select at least one PDF file.')
            return render(request, 'flashcards/bulk_upload.html')
        
        uploaded_count = 0
        error_count = 0
        duplicate_count = 0
        
        for file in files:
            if not file.name.lower().endswith('.pdf'):
                error_count += 1
                continue
                
            try:
                # Check file size (limit to 50MB)
                if file.size > 50 * 1024 * 1024:
                    messages.warning(request, f'{file.name}: File too large (max 50MB)')
                    error_count += 1
                    continue
                
                # Create PDF document
                pdf_doc = PDFDocument.objects.create(
                    name=file.name.replace('.pdf', ''),
                    pdf_file=file,
                    file_size=file.size
                )
                
                # Start processing in background
                try:
                    success, message, stats = process_pdf_complete(pdf_doc)
                    if success:
                        uploaded_count += 1
                    else:
                        if 'duplicate' in message.lower():
                            duplicate_count += 1
                        else:
                            error_count += 1
                            messages.warning(request, f'{file.name}: {message}')
                except Exception as e:
                    error_count += 1
                    messages.error(request, f'{file.name}: Processing error - {str(e)}')
                    
            except Exception as e:
                error_count += 1
                messages.error(request, f'{file.name}: Upload error - {str(e)}')
        
        # Show summary
        if uploaded_count > 0:
            messages.success(request, f'Successfully uploaded and processed {uploaded_count} PDFs')
        if duplicate_count > 0:
            messages.info(request, f'Skipped {duplicate_count} duplicate PDFs')
        if error_count > 0:
            messages.warning(request, f'{error_count} files had errors')
            
        return redirect('flashcards:bulk_manage')
    
    return render(request, 'flashcards/bulk_upload.html')


def bulk_manage(request):
    """Manage all uploaded PDFs"""
    # Get filter parameters
    status_filter = request.GET.get('status', 'all')
    search_query = request.GET.get('search', '')
    
    # Base queryset
    pdfs = PDFDocument.objects.all().order_by('-created_at')
    
    # Apply filters
    if status_filter == 'ready':
        pdfs = pdfs.filter(processed=True, concept_units__isnull=False)
    elif status_filter == 'processing':
        pdfs = pdfs.filter(processed=False)
    elif status_filter == 'duplicates':
        pdfs = pdfs.filter(is_duplicate=True)
    elif status_filter == 'errors':
        pdfs = pdfs.exclude(processing_error__isnull=True).exclude(processing_error='')
    
    # Apply search
    if search_query:
        pdfs = pdfs.filter(name__icontains=search_query)
    
    # Get statistics
    total_pdfs = PDFDocument.objects.count()
    ready_pdfs = PDFDocument.objects.filter(processed=True, concept_units__isnull=False).count()
    processing_pdfs = PDFDocument.objects.filter(processed=False).count()
    duplicate_pdfs = PDFDocument.objects.filter(is_duplicate=True).count()
    error_pdfs = PDFDocument.objects.exclude(processing_error__isnull=True).exclude(processing_error='').count()
    
    # Get focus blocks count
    total_focus_blocks = FocusBlock.objects.count()
    
    context = {
        'pdfs': pdfs,
        'status_filter': status_filter,
        'search_query': search_query,
        'stats': {
            'total_pdfs': total_pdfs,
            'ready_pdfs': ready_pdfs,
            'processing_pdfs': processing_pdfs,
            'duplicate_pdfs': duplicate_pdfs,
            'error_pdfs': error_pdfs,
            'total_focus_blocks': total_focus_blocks,
        }
    }
    
    return render(request, 'flashcards/bulk_manage.html', context)


def bulk_delete_pdfs(request):
    """Delete multiple PDFs"""
    if request.method == 'POST':
        pdf_ids = request.POST.getlist('pdf_ids')
        if pdf_ids:
            deleted_count = PDFDocument.objects.filter(id__in=pdf_ids).count()
            PDFDocument.objects.filter(id__in=pdf_ids).delete()
            messages.success(request, f'Successfully deleted {deleted_count} PDFs and their associated data')
        else:
            messages.warning(request, 'No PDFs selected for deletion')
    
    return redirect('flashcards:bulk_manage')

def focus_schedule(request):
    """Focus Schedule dashboard showing new blocks, completed blocks, and repetition schedule"""
    
    # Get all focus blocks
    all_focus_blocks = FocusBlock.objects.select_related(
        'pdf_document', 'main_concept_unit'
    ).order_by('pdf_document__created_at', 'block_order')
    
    # ✅ GET COMPLETION DATA FROM DATABASE (ORIGINAL WORKING VERSION)
    completed_sessions = FocusSession.objects.filter(
        status='completed'
    ).select_related('focus_block').order_by('-completed_at')
    
    # Group by focus block
    completion_stats = {}
    for session in completed_sessions:
        block_id = session.focus_block.id
        if block_id not in completion_stats:
            completion_stats[block_id] = {
                'focus_block': session.focus_block,
                'total_completions': 0,
                'last_completed': None,
                'last_proficiency': None,
                'average_time': 0,
                'sessions': []
            }
        
        completion_stats[block_id]['total_completions'] += 1
        completion_stats[block_id]['sessions'].append({
            'completed_at': session.completed_at,
            'proficiency_score': session.proficiency_score,
            'study_time': session.total_study_time
        })
        
        # Update latest completion info
        if (not completion_stats[block_id]['last_completed'] or 
            session.completed_at > completion_stats[block_id]['last_completed']):
            completion_stats[block_id]['last_completed'] = session.completed_at
            completion_stats[block_id]['last_proficiency'] = session.proficiency_score
    
    # Calculate averages
    for block_id, stats in completion_stats.items():
        if stats['sessions']:
            total_time = sum(s['study_time'] or 0 for s in stats['sessions'])
            stats['average_time'] = total_time / len(stats['sessions'])
    
    # Separate into categories
    completed_blocks = list(completion_stats.values())
    completed_block_ids = set(completion_stats.keys())
    new_blocks = [block for block in all_focus_blocks if block.id not in completed_block_ids]
    
    context = {
        'all_focus_blocks': all_focus_blocks,
        'total_blocks': all_focus_blocks.count(),
        # ✅ DATABASE-BASED COMPLETION DATA
        'completed_blocks': completed_blocks,
        'new_blocks': new_blocks,
        'total_completed': len(completed_blocks),
        'total_new': len(new_blocks),
        'completion_stats': completion_stats,
    }
    
    return render(request, 'flashcards/focus_schedule.html', context)

def start_focus_mode(request, focus_block_id):
    """Start a new focus session"""
    try:
        focus_block = FocusBlock.objects.get(id=focus_block_id)
        
        # Create new session
        session = FocusSession.objects.create(
            focus_block=focus_block,
            status='active'
        )
        
        return redirect('flashcards:focus_mode', session_id=session.id)
        
    except FocusBlock.DoesNotExist:
        messages.error(request, "Focus block not found")
        return redirect('flashcards:all_focus_blocks')

def focus_mode(request, session_id):
    """Focus mode interface with timer"""
    try:
        session = FocusSession.objects.get(id=session_id)
        focus_block = session.focus_block
        segments = focus_block.get_segments()
        
        context = {
            'session': session,
            'focus_block': focus_block,
            'segments': segments,
            'revision_data': focus_block.get_revision_data(),
            'qa_items': focus_block.get_qa_items(),
            'recap_data': focus_block.get_recap_data(),
            'total_segments': len(segments),
            'current_segment': session.current_segment,
        }
        
        return render(request, 'flashcards/focus_mode.html', context)
        
    except FocusSession.DoesNotExist:
        messages.error(request, "Focus session not found")
        return redirect('flashcards:all_focus_blocks')

def update_focus_progress(request, session_id):
    """AJAX endpoint to update focus session progress"""
    if request.method == 'POST':
        try:
            session = FocusSession.objects.get(id=session_id)
            data = json.loads(request.body)
            
            action = data.get('action')
            
            if action == 'complete_segment':
                segment_index = data.get('segment_index')
                time_spent = data.get('time_spent')
                session.mark_segment_completed(segment_index, time_spent)
                
            elif action == 'pause_session':
                session.status = 'paused'
                session.save()
                
            elif action == 'resume_session':
                session.status = 'active'
                session.save()
                
            elif action == 'complete_session':
                session.status = 'completed'
                session.completed_at = timezone.now()
                session.total_study_time = data.get('total_time')
                session.save()
            
            return JsonResponse({'status': 'success'})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def complete_focus_session(request, session_id):
    """Complete session and collect proficiency score"""
    if request.method == 'POST':
        try:
            session = FocusSession.objects.get(id=session_id)
            
            # Get completion data
            proficiency_score = request.POST.get('proficiency_score')
            total_study_time = request.POST.get('total_study_time')  # in seconds
            
            # Update session
            session.proficiency_score = int(proficiency_score) if proficiency_score else None
            session.total_study_time = float(total_study_time) if total_study_time else None
            session.status = 'completed'
            session.completed_at = timezone.now()
            session.save()
            
            # Return success with next review calculation
            return JsonResponse({
                'success': True,
                'proficiency_score': proficiency_score,
                'completion_time': total_study_time
            })
            
        except FocusSession.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Session not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def start_block_session(request, focus_block_id):
    """Start a new FocusSession for tracking"""
    if request.method == 'POST':
        try:
            focus_block = FocusBlock.objects.get(id=focus_block_id)
            
            session = FocusSession.objects.create(
                focus_block=focus_block,
                status='active'
            )
            
            return JsonResponse({
                'success': True,
                'session_id': str(session.id)
            })
            
        except FocusBlock.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Focus block not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

@csrf_exempt  # ✅ FIXED: Use @csrf_exempt directly for function-based views
def complete_focus_block_api(request, focus_block_id):
    """API endpoint to save focus block completion data"""
    if request.method == 'POST':
        try:
            # Parse request data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = request.POST
            
            # ✅ Add validation
            proficiency_score = data.get('proficiency_score')
            completion_time = data.get('completion_time')
            
            if not proficiency_score or not completion_time:
                return JsonResponse({
                    'success': False, 
                    'error': 'Missing proficiency_score or completion_time'
                })
            
            focus_block = FocusBlock.objects.get(id=focus_block_id)
            
            # Create a new FocusSession record (for detailed tracking)
            session = FocusSession.objects.create(
                focus_block=focus_block,
                proficiency_score=int(proficiency_score),
                total_study_time=float(completion_time),
                status='completed',
                completed_at=timezone.now()
            )

            # ✅ DEBUG: Verify the session was created
            print(f"🔍 API COMPLETION DEBUG:")
            print(f"   Created FocusSession ID: {session.id}")
            print(f"   Focus Block ID: {focus_block.id}")
            print(f"   Focus Block Title: {focus_block.title}")
            print(f"   Status: {session.status}")
            print(f"   Proficiency: {session.proficiency_score}")
            print(f"   Completed At: {session.completed_at}")

            # Verify it's in the database
            verification = FocusSession.objects.filter(
                focus_block=focus_block, 
                status='completed'
            ).count()
            print(f"   ✅ Total completed sessions for this block: {verification}")
            
            # ✅ ALSO add to the original working system
            # Find or create a study session for this user/PDF
            study_session, created = StudySession.objects.get_or_create(
                pdf_document=focus_block.pdf_document,
                session_type='focus_blocks',
                defaults={'session_id': uuid.uuid4()}
            )

            # Add to the original ManyToMany relationship
            study_session.completed_focus_blocks.add(focus_block)
            study_session.save()

            print(f"   ✅ Added to StudySession completed_focus_blocks (original working system)")
            
            # Calculate next review date using spaced repetition logic
            proficiency_score = int(proficiency_score)
            completion_time = float(completion_time)
            
            # Simple spaced repetition logic
            base_intervals = [1, 3, 7, 21, 52]  # days
            proficiency_multipliers = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.5}
            
            # Get current repetition level (how many times completed)
            previous_sessions = FocusSession.objects.filter(
                focus_block=focus_block, 
                status='completed'
            ).count() - 1  # Subtract current session
            
            level = min(previous_sessions, len(base_intervals) - 1)
            interval = base_intervals[level] * proficiency_multipliers.get(proficiency_score, 1.0)
            
            # ✅ FIXED: Use timedelta correctly  
            next_review = timezone.now() + timedelta(days=interval)
            
            return JsonResponse({
                'success': True,
                'session_id': str(session.id),
                'next_review_date': next_review.isoformat(),
                'interval_days': int(interval),
                'message': f'Completed! Next review in {int(interval)} days'
            })
            
        except FocusBlock.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Focus block not found'})
        except ValueError as e:
            return JsonResponse({'success': False, 'error': f'Invalid data: {str(e)}'})
        except Exception as e:
            # ✅ Better error logging for debugging
            import traceback
            print(f"API Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return JsonResponse({'success': False, 'error': f'Server error: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@csrf_exempt
def mark_segment_complete_api(request):
    """API endpoint to mark a segment as complete"""
    if request.method == 'POST':
            data = json.loads(request.body)
            session_id = data.get('segment_id')
            segment_index = data.get('segment_id')

            try:
                session = FocusSession.objects.get(id=session_id)
                session.mark_segment_completed(segment_index, time_spent=0)
                session.save()
                return JsonResponse({'success': True, 'progress': session.get_completion_percentage()})
            except FocusSession.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Session not found'})
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@csrf_exempt
def get_focus_completions_api(request):
    """API endpoint to load user's focus block completions"""
    if request.method == 'GET':
        try:
            # Get all completed sessions for this user
            completed_sessions = FocusSession.objects.filter(
                status='completed'
            ).select_related('focus_block').order_by('completed_at')
            
            # Convert to simple format
            completions = []
            for session in completed_sessions:
                # Find block index (position in the sequence)
                block_index = None
                try:
                    # Get all blocks in order
                    all_blocks = FocusBlock.objects.order_by('pdf_document__created_at', 'block_order')
                    for idx, block in enumerate(all_blocks):
                        if block.id == session.focus_block.id:
                            block_index = idx
                            break
                except:
                    block_index = 0
                
                completions.append({
                    'session_id': str(session.id),
                    'block_index': block_index,
                    'block_title': session.focus_block.title,
                    'focus_block_id': str(session.focus_block.id),
                    'proficiency_score': session.proficiency_score,
                    'total_study_time': session.total_study_time,
                    'completed_at': session.completed_at.isoformat(),
                })
            
            return JsonResponse({
                'success': True,
                'completions': completions,
                'total_count': len(completions)
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def debug_focus_chunks(request):
    """Debug page for new focus blocks approach - chunking and concept analysis"""
    # Get all processed PDFs for selection
    processed_pdfs = PDFDocument.objects.filter(
        processed=True,
        extracted_text__isnull=False
    ).order_by('-created_at')
    
    context = {
        'processed_pdfs': processed_pdfs,
        'chunks_data': [],
        'grouped_concepts': [],
        'focus_blocks': [],
        'knowledge_graph': None,
        'selected_pdf': None,
        'chunk_size': 300,
        'analysis_message': None,
        'grouping_message': None,
        'focus_block_message': None,
        'graph_message': None
    }
    
    # If a PDF is selected, process it
    pdf_id = request.GET.get('pdf_id')
    analyze_request = request.GET.get('analyze', False)
    group_request = request.GET.get('group', False)
    generate_blocks_request = request.GET.get('generate_blocks', False)
    generate_graph_request = request.GET.get('generate_graph', False)
    
    if pdf_id:
        try:
            selected_pdf = PDFDocument.objects.get(id=pdf_id)
            context['selected_pdf'] = selected_pdf
            
            # Generate 300-word chunks
            chunks_data = generate_text_chunks_with_concepts(selected_pdf.extracted_text, chunk_size=300)
            
            # If analysis is requested, analyze all chunks with AI
            if analyze_request and analyze_request.lower() == 'true':
                try:
                    analyzed_chunks = analyze_all_chunks_with_ai(chunks_data, selected_pdf.name)
                    context['chunks_data'] = analyzed_chunks
                    context['analysis_message'] = f"✅ Successfully analyzed {len(analyzed_chunks)} chunks with AI!"
                except Exception as e:
                    context['chunks_data'] = chunks_data
                    context['analysis_message'] = f"❌ AI Analysis failed: {str(e)}"
            else:
                context['chunks_data'] = chunks_data
            
            # If grouping is requested, group similar concepts
            if group_request and group_request.lower() == 'true' and context['chunks_data']:
                try:
                    # Only group if chunks have been analyzed
                    analyzed_chunks = [c for c in context['chunks_data'] if c.get('analysis_status') == 'completed']
                    if analyzed_chunks:
                        grouped_concepts = group_similar_concepts(analyzed_chunks, selected_pdf.name)
                        context['grouped_concepts'] = grouped_concepts
                        context['grouping_message'] = f"✅ Grouped {len(analyzed_chunks)} chunks into {len(grouped_concepts)} concept groups!"
                    else:
                        context['grouping_message'] = "❌ No analyzed chunks found. Run AI analysis first."
                except Exception as e:
                    context['grouping_message'] = f"❌ Concept grouping failed: {str(e)}"
            
            # If focus block generation is requested
            if generate_blocks_request and generate_blocks_request.lower() == 'true' and context.get('grouped_concepts'):
                try:
                    focus_blocks = generate_focus_blocks_from_groups(context['grouped_concepts'], selected_pdf.name)
                    context['focus_blocks'] = focus_blocks
                    context['focus_block_message'] = f"✅ Generated {len(focus_blocks)} focus blocks from concept groups!"
                except Exception as e:
                    context['focus_block_message'] = f"❌ Focus block generation failed: {str(e)}"
            
            # If knowledge graph generation is requested
            if generate_graph_request and generate_graph_request.lower() == 'true' and context.get('focus_blocks'):
                try:
                    knowledge_graph = generate_knowledge_graph(context['focus_blocks'], selected_pdf.name)
                    context['knowledge_graph'] = knowledge_graph
                    context['graph_message'] = f"✅ Generated knowledge graph with {len(knowledge_graph['nodes'])} nodes and {len(knowledge_graph['edges'])} connections!"
                except Exception as e:
                    context['graph_message'] = f"❌ Knowledge graph generation failed: {str(e)}"
            
            context['total_chunks'] = len(context['chunks_data'])
            
        except PDFDocument.DoesNotExist:
            pass
    
    return render(request, 'flashcards/debug_focus_chunks.html', context)

def generate_knowledge_graph(focus_blocks, pdf_name):
    """
    Generate a knowledge graph showing relationships between focus blocks
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    print(f"🕸️ Generating knowledge graph for {len(focus_blocks)} focus blocks...")
    
    # Step 1: Create nodes from focus blocks
    nodes = []
    for i, block in enumerate(focus_blocks):
        # Extract chunk roles from compact7_data if available
        chunk_roles = []
        if hasattr(block, 'compact7_data') and block.compact7_data:
            segments = block.compact7_data.get('segments', [])
            chunk_roles = [seg.get('type', 'unknown') for seg in segments]
        
        node = {
            'id': f"block_{block.id}",
            'label': block.title,
            'title': block.title,
            'group_id': str(block.id),
            'estimated_duration': block.target_duration or 420,  # Use target_duration
            'chunk_count': len(chunk_roles) or 1,
            'chunk_roles': chunk_roles,
            'confidence': 0.8,  # Default confidence
            'size': max(20, min(60, len(chunk_roles) * 10)) if chunk_roles else 30,
            'color': get_node_color(chunk_roles),
            'level': i  # For hierarchical layout
        }
        nodes.append(node)

def new_format_knowledge_graph(request):
    """
    Knowledge graph view using persistent FocusBlockRelationship data
    """
    # Get only new format blocks (with segments)
    new_format_blocks = FocusBlock.objects.filter(
        compact7_data__has_key='segments'
    ).exclude(
        title__startswith='[MIGRATED→'  # Exclude migrated blocks
    ).order_by('pdf_document__name', 'block_order')
    
    # Get all persistent relationships
    relationships = FocusBlockRelationship.objects.filter(
        from_block__in=new_format_blocks,
        to_block__in=new_format_blocks
    ).select_related('from_block', 'to_block').order_by('-edge_strength')
    
    context = {
        'focus_blocks': new_format_blocks,
        'relationships': relationships,
        'knowledge_graph': None,
        'total_blocks': new_format_blocks.count(),
        'total_relationships': relationships.count(),
        'message': None
    }
    
    # 🆕 ADD: Handle manual relationship generation for existing blocks
    if request.method == 'POST' and request.POST.get('action') == 'generate_relationships':
        try:
            # Generate relationships for ALL existing blocks
            success_count = generate_relationships_for_existing_blocks(new_format_blocks)
            context['message'] = f"✅ Generated {success_count} new relationships for existing blocks"
            
            # Refresh relationships after generation
            relationships = FocusBlockRelationship.objects.filter(
                from_block__in=new_format_blocks,
                to_block__in=new_format_blocks
            ).select_related('from_block', 'to_block').order_by('-edge_strength')
            context['relationships'] = relationships
            context['total_relationships'] = relationships.count()
            
        except Exception as e:
            context['message'] = f"❌ Relationship generation failed: {str(e)}"
    
    if request.method == 'POST' and request.POST.get('action') == 'generate_graph':
        try:
            # Generate knowledge graph using persistent relationships
            knowledge_graph = generate_knowledge_graph_from_relationships(new_format_blocks, relationships)
            
            # Process graph data for template
            if knowledge_graph and 'edges' in knowledge_graph:
                for edge in knowledge_graph['edges']:
                    # Add calculated values for template
                    edge['strength_percent'] = int(float(edge.get('strength', 0)) * 100)
                    edge['width_value'] = float(edge.get('strength', 0)) * 5
            
            context['knowledge_graph'] = knowledge_graph
            context['message'] = f"✅ Generated knowledge graph with {len(knowledge_graph['nodes'])} nodes and {len(knowledge_graph['edges'])} relationships"
        except Exception as e:
            context['message'] = f"❌ Graph generation failed: {str(e)}"
    
    return render(request, 'flashcards/new_format_knowledge_graph.html', context)

    
def generate_relationships_for_existing_blocks(focus_blocks):
    """
    Generate relationships for all existing focus blocks (one-time operation)
    """
    print(f"🔄 Generating relationships for {focus_blocks.count()} existing blocks...")
    
    if focus_blocks.count() < 2:
        print("📊 Not enough blocks for relationships (need at least 2)")
        return 0
    
    # Convert QuerySet to list for easier processing
    all_blocks = list(focus_blocks)
    total_created = 0
    
    # Generate relationships for each block with all others
    for i, block_a in enumerate(all_blocks):
        # Get other blocks (excluding current one)
        other_blocks = [b for j, b in enumerate(all_blocks) if j != i]
        
        if other_blocks:
            print(f"🔍 Analyzing block {i+1}/{len(all_blocks)}: {block_a.title[:50]}...")
            
            try:
                relationships = analyze_new_block_relationships(block_a, other_blocks)
                
                # Store relationships
                for rel_data in relationships:
                    try:
                        relationship, created = FocusBlockRelationship.objects.get_or_create(
                            from_block_id=rel_data['from_block_id'],
                            to_block_id=rel_data['to_block_id'],
                            relationship_type=rel_data['relationship_type'],
                            defaults={
                                'confidence': rel_data['confidence'],
                                'similarity_score': rel_data.get('similarity_score', 0.0),
                                'edge_strength': rel_data.get('edge_strength', rel_data['confidence']),
                                'description': rel_data['description'],
                                'educational_reasoning': rel_data.get('educational_reasoning', '')
                            }
                        )
                        
                        if created:
                            total_created += 1
                            print(f"  ✅ Created: {rel_data['relationship_type']} relationship")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to create relationship: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"  ❌ Failed to analyze block: {str(e)}")
                continue
    
    print(f"🕸️ Relationship generation complete: {total_created} relationships created")
    return total_created

def analyze_focus_block_relationships(focus_blocks, similarity_matrix, pdf_name, client):
    """
    Use AI to analyze relationships between focus blocks
    """
    import json
    
    # Prepare focus block data for AI analysis
    blocks_info = []
    for i, block in enumerate(focus_blocks):
        # Extract segments from compact7_data
        segments = []
        if hasattr(block, 'compact7_data') and block.compact7_data:
            segments = block.compact7_data.get('segments', [])
        
        blocks_info.append({
            'block_id': str(block.id),
            'title': block.title,
            'estimated_duration': block.target_duration or 420,
            'chunk_roles': [seg.get('type', 'unknown') for seg in segments],
            'segments': [seg.get('title', '') for seg in segments],
            'index': i
        })
    
    # Calculate similarity pairs above threshold
    similarity_pairs = []
    threshold = 0.3  # Lower threshold for knowledge graph
    for i in range(len(focus_blocks)):
        for j in range(i + 1, len(focus_blocks)):
            similarity = similarity_matrix[i][j]
            if similarity >= threshold:
                similarity_pairs.append({
                    'block1_id': focus_blocks[i]['group_id'],
                    'block2_id': focus_blocks[j]['group_id'],
                    'similarity': float(similarity)
                })
    
    prompt = f"""
    Analyze the relationships between focus blocks in the document "{pdf_name}".
    
    FOCUS BLOCKS:
    {json.dumps(blocks_info, indent=2)}
    
    SIMILAR PAIRS (semantic similarity >= {threshold}):
    {json.dumps(similarity_pairs, indent=2)}
    
    Your task: Identify meaningful relationships between focus blocks and create a knowledge graph.
    
    RELATIONSHIP TYPES:
    - "prerequisite" - Block A must be understood before Block B
    - "builds_on" - Block B extends concepts from Block A  
    - "related" - Blocks cover related but independent concepts
    - "applies_to" - Block A provides theory, Block B shows applications
    - "compares_with" - Blocks compare/contrast different approaches
    - "specializes" - Block B is a specific case of Block A
    
    RELATIONSHIP RULES:
    1. Create directed relationships (from → to)
    2. Focus on educational progression and logical flow
    3. Consider chunk roles (definition → example → application)
    4. Estimated duration can indicate complexity level
    5. Only create relationships that make educational sense
    6. Avoid creating too many relationships (quality over quantity)
    
    Return JSON with relationships:
    {{
        "relationships": [
            {{
                "from_block_id": 1,
                "to_block_id": 2,
                "relationship_type": "prerequisite",
                "confidence": 0.85,
                "description": "Understanding Block 1 is essential before learning Block 2",
                "educational_reasoning": "Why this relationship helps learning"
            }}
        ]
    }}
    
    IMPORTANT:
    - Only include relationships with confidence >= 0.4
    - Provide clear educational reasoning for each relationship
    - Consider the learning progression and difficulty
    - Return ONLY valid JSON
    """
    
    print(f"🤖 Analyzing relationships between {len(focus_blocks)} focus blocks...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in educational content structure and learning pathways. Create meaningful connections that enhance the learning experience."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=3000,
        temperature=0.2
    )
    
    ai_response = response.choices[0].message.content.strip()
    
    try:
        # Parse AI response
        if ai_response.startswith('```json'):
            ai_response = ai_response.replace('```json', '').replace('```', '').strip()
        
        ai_data = json.loads(ai_response)
        relationships = ai_data.get('relationships', [])
        
        print(f"✅ AI identified {len(relationships)} meaningful relationships")
        for rel in relationships:
            print(f"  🔗 {rel.get('from_block_id')} → {rel.get('to_block_id')}: {rel.get('relationship_type')} (conf: {rel.get('confidence', 0):.2f})")
        
        return relationships
        
    except json.JSONDecodeError as e:
        print(f"❌ Relationship Analysis JSON Parse Error: {e}")
        print(f"Raw AI Response: {ai_response[:500]}...")
        
        # Fallback: create basic relationships based on similarity
        fallback_relationships = []
        for pair in similarity_pairs[:5]:  # Limit to top 5 similar pairs
            fallback_relationships.append({
                'from_block_id': pair['block1_id'],
                'to_block_id': pair['block2_id'],
                'relationship_type': 'related',
                'confidence': pair['similarity'],
                'description': f"Related concepts (similarity: {pair['similarity']:.2f})",
                'educational_reasoning': 'Fallback relationship based on semantic similarity'
            })
        
        return fallback_relationships

def get_node_color(chunk_roles):
    """Assign colors to nodes based on difficulty/complexity instead of single content type"""
    # Since focus blocks have multiple content types, use different criteria
    complexity_score = len(chunk_roles)  # More roles = more complex
    
    if complexity_score >= 4:
        return '#e74c3c'      # Red - High complexity
    elif complexity_score >= 3:
        return '#f39c12'      # Orange - Medium complexity  
    elif complexity_score >= 2:
        return '#f1c40f'      # Yellow - Low-medium complexity
    else:
        return '#2ecc71'      # Green - Simple concepts

def get_edge_color(relationship_type):
    """Assign colors to edges based on relationship type"""
    edge_colors = {
        'prerequisite': '#e74c3c',      # Red - must learn first
        'builds_on': '#3498db',         # Blue - natural progression
        'related': '#95a5a6',           # Gray - loose connection
        'applies_to': '#2ecc71',        # Green - theory to practice
        'compares_with': '#f39c12',     # Orange - comparison
        'specializes': '#9b59b6'        # Purple - specialization
    }
    
    return edge_colors.get(relationship_type, '#bdc3c7')

def generate_focus_blocks_from_groups(grouped_concepts, pdf_name):
    """
    Generate structured focus blocks from grouped concepts using chunk content
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    print(f"🎯 Generating focus blocks for {len(grouped_concepts)} concept groups...")
    
    focus_blocks = []
    
    for i, group in enumerate(grouped_concepts):
        print(f"📚 Processing group {i+1}: {group['unified_concept']}")
        
        # Prepare chunk content by role
        chunks_by_role = {}
        for chunk in group['chunks']:
            role = chunk.get('role', 'unknown')
            if role not in chunks_by_role:
                chunks_by_role[role] = []
            chunks_by_role[role].append(chunk)
        
        # Create the AI prompt for focus block generation
        prompt = f"""
        Create a comprehensive, engaging focus block for the concept: "{group['unified_concept']}"
        
        You have the following content chunks organized by role:
        {json.dumps(chunks_by_role, indent=2, default=str)}
        
        Create a focus block with this EXACT structure:
        
        {{
            "title": "{group['unified_concept']}",
            "recap": "Brief connection to previous concepts (2-3 sentences)",
            "socratic_intro": {{
                "hook_question": "Engaging question to start thinking about this concept",
                "exploration_questions": ["Question 1", "Question 2", "Question 3"],
                "bridge_to_content": "Transition to the main content"
            }},
            "segments": [
                {{
                    "segment_title": "Core Understanding",
                    "content": "Main explanation using definition chunks",
                    "key_points": ["Point 1", "Point 2", "Point 3"]
                }},
                {{
                    "segment_title": "Mathematical Framework", 
                    "content": "Formulas and calculations using formula/derivation chunks",
                    "key_points": ["Formula 1", "Calculation method", "Key relationships"]
                }},
                {{
                    "segment_title": "Practical Applications",
                    "content": "Real-world examples using example chunks", 
                    "key_points": ["Application 1", "Application 2", "Use cases"]
                }}
            ],
            "qa_items": [
                {{
                    "prompt": "Understanding check question",
                    "ideal_answer": "Comprehensive answer",
                    "type": "conceptual"
                }},
                {{
                    "prompt": "Application question", 
                    "ideal_answer": "Practical answer",
                    "type": "application"
                }}
            ],
            "estimated_duration": 15
        }}
        
        CONTENT GUIDELINES:
        1. **Use chunk content extensively** - don't make up new information
        2. **Socratic method** - start with questions that lead to discovery
        3. **Progressive difficulty** - build from basic to advanced
        4. **Practical relevance** - connect theory to real applications
        5. **Comprehensive coverage** - use all chunk roles effectively
        6. **Engaging tone** - make it interesting and thought-provoking
        
        SEGMENT CREATION RULES:
        - If you have definition chunks → Create "Core Understanding" segment
        - If you have formula/derivation chunks → Create "Mathematical Framework" segment  
        - If you have example chunks → Create "Practical Applications" segment
        - If you have procedure chunks → Create "Method & Process" segment
        - Adapt segment titles based on available content
        
        Return ONLY valid JSON, no other text.
        """
        
        print(f"🤖 Generating focus block for: {group['unified_concept']}")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert educational content creator. Create engaging, well-structured focus blocks that promote deep learning through the Socratic method and comprehensive content coverage."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        try:
            # Parse AI response
            if ai_response.startswith('```json'):
                ai_response = ai_response.replace('```json', '').replace('```', '').strip()
            
            focus_block_data = json.loads(ai_response)
            
            # Add metadata
            focus_block_data.update({
                'group_id': group['group_id'],
                'source_chunks': len(group['chunks']),
                'chunk_roles': list(chunks_by_role.keys()),
                'confidence': group['confidence'],
                'generation_status': 'success'
            })
            
            focus_blocks.append(focus_block_data)
            print(f"✅ Generated focus block: {focus_block_data.get('title', 'Unknown')}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error for group {i+1}: {e}")
            
            # Fallback focus block
            fallback_block = {
                'title': group['unified_concept'],
                'recap': f"Building on previous concepts, we now explore {group['unified_concept']}.",
                'socratic_intro': {
                    'hook_question': f"What do you think {group['unified_concept']} means?",
                    'exploration_questions': ["How might this concept apply?", "What are the key components?"],
                    'bridge_to_content': "Let's explore this systematically."
                },
                'segments': [
                    {
                        'segment_title': 'Content Overview',
                        'content': f"This concept involves {len(group['chunks'])} key aspects.",
                        'key_points': [f"Chunk {chunk['chunk_id']}: {chunk['core_concept']}" for chunk in group['chunks'][:3]]
                    }
                ],
                'qa_items': [
                    {
                        'prompt': f"Explain the main idea of {group['unified_concept']}",
                        'ideal_answer': "Based on the content analysis.",
                        'type': 'conceptual'
                    }
                ],
                'estimated_duration': 10,
                'group_id': group['group_id'],
                'source_chunks': len(group['chunks']),
                'chunk_roles': list(chunks_by_role.keys()),
                'confidence': 0.1,
                'generation_status': 'fallback'
            }
            
            focus_blocks.append(fallback_block)
    
    print(f"🎯 Generated {len(focus_blocks)} focus blocks total")
    return focus_blocks

def focus_block_study(request, block_id):
    focus_block = get_object_or_404(FocusBlock, id=block_id)
    
    # Extract segments from compact7_data
    segments = focus_block.compact7_data.get('segments', [])
    
    context = {
        'focus_block': focus_block,
        'segments': segments,
        'title': focus_block.title,
        'learning_objectives': focus_block.learning_objectives,
        'total_duration': focus_block.target_duration,
    }
    return render(request, 'flashcards/focus_block_study.html', context)

def start_focus_study_session(request):
    """Start an interactive study session with focus blocks"""
    if request.method == 'POST':
        # Get selected focus blocks or use all from a PDF
        pdf_id = request.POST.get('pdf_id')
        if pdf_id:
            # For now, we'll redirect to debug page but later this will be the study interface
            return redirect('flashcards:debug_focus_chunks') 
    
    # Show available PDFs with focus blocks
    processed_pdfs = PDFDocument.objects.filter(
        processed=True,
        extracted_text__isnull=False
    ).order_by('-created_at')
    
    context = {
        'processed_pdfs': processed_pdfs
    }
    return render(request, 'flashcards/start_focus_study.html', context)

def migrate_old_focus_blocks_for_pdf(pdf_document):
    """
    Migrate old focus blocks for a specific PDF to the new interactive format
    First consolidate similar blocks, then convert to interactive format
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    # Get old focus blocks for this PDF
    old_focus_blocks = FocusBlock.objects.filter(pdf_document=pdf_document)
    
    if not old_focus_blocks.exists():
        return 0
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    print(f"🔄 Analyzing {old_focus_blocks.count()} old focus blocks for {pdf_document.name}")
    
    # Step 1: Consolidate similar old focus blocks
    consolidated_groups = consolidate_similar_focus_blocks(old_focus_blocks, client)
    
    print(f"📊 Consolidated {old_focus_blocks.count()} blocks into {len(consolidated_groups)} groups")
    
    # Step 2: Convert each consolidated group to interactive format
    migrated_count = 0
    
    for group in consolidated_groups:
        try:
            # Convert consolidated group to interactive format
            interactive_block = convert_consolidated_group_to_interactive(group, client)
            
            # Store the converted block (you might want to save this somewhere)
            # For now, we'll just count successful conversions
            migrated_count += 1
            print(f"✅ Created interactive block: {interactive_block['title']}")
            
        except Exception as e:
            print(f"❌ Failed to convert group: {str(e)}")
            continue
    
    return migrated_count

def consolidate_similar_focus_blocks(old_focus_blocks, client):
    """
    Group similar old focus blocks together using embeddings + AI analysis
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import json
    
    print(f"🔍 Analyzing similarity between {old_focus_blocks.count()} focus blocks...")
    
    # Prepare focus block data for analysis
    blocks_data = []
    for block in old_focus_blocks:
        block_text = f"{block.title} {getattr(block, 'content', '')} {getattr(block, 'description', '')}"
        blocks_data.append({
            'id': block.id,
            'title': block.title,
            'content': getattr(block, 'content', ''),
            'description': getattr(block, 'description', ''),
            'text_for_embedding': block_text[:1000],  # Limit for embedding
            'target_duration': getattr(block, 'target_duration', 15),
            'difficulty_level': getattr(block, 'difficulty_level', 'medium')
        })
    
    # Generate embeddings for focus block content
    texts_for_embedding = [block['text_for_embedding'] for block in blocks_data]
    
    print(f"📊 Generating embeddings for {len(texts_for_embedding)} focus blocks...")
    
    embeddings_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts_for_embedding
    )
    
    embeddings = [item.embedding for item in embeddings_response.data]
    embeddings_matrix = np.array(embeddings)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # Find similar pairs (more aggressive threshold for old blocks)
    similarity_threshold = 0.5  # Lower threshold to catch repetitive content
    similar_pairs = []
    
    print(f"🔍 FOCUS BLOCK SIMILARITIES (threshold: {similarity_threshold}):")
    for i in range(len(blocks_data)):
        for j in range(i + 1, len(blocks_data)):
            similarity = similarity_matrix[i][j]
            print(f"  '{blocks_data[i]['title'][:40]}...' vs '{blocks_data[j]['title'][:40]}...': {similarity:.3f}")
            
            if similarity >= similarity_threshold:
                similar_pairs.append({
                    'block1_id': blocks_data[i]['id'],
                    'block2_id': blocks_data[j]['id'],
                    'title1': blocks_data[i]['title'],
                    'title2': blocks_data[j]['title'],
                    'similarity': float(similarity)
                })
                print(f"    ✅ SIMILAR: {similarity:.3f} >= {similarity_threshold}")
    
    print(f"🔗 Found {len(similar_pairs)} similar focus block pairs")
    
    # Use AI to make intelligent grouping decisions
    if similar_pairs:
        consolidated_groups = ai_consolidate_focus_blocks(blocks_data, similar_pairs, client)
    else:
        # No similar blocks - each block is its own group
        print("❌ No similar focus blocks found - keeping all separate")
        consolidated_groups = []
        for block_data in blocks_data:
            consolidated_groups.append({
                'group_id': len(consolidated_groups) + 1,
                'unified_title': block_data['title'],
                'block_ids': [block_data['id']],
                'blocks_data': [block_data],
                'consolidation_reason': 'No similar blocks found - standalone concept',
                'estimated_duration': block_data['target_duration']
            })
    
    return consolidated_groups

def ai_consolidate_focus_blocks(blocks_data, similar_pairs, client):
    """
    Use AI to intelligently consolidate similar focus blocks
    """
    import json
    
    # Prepare similarity pairs for AI
    pairs_info = []
    for pair in similar_pairs:
        pairs_info.append({
            'block1_id': pair['block1_id'],
            'block2_id': pair['block2_id'],
            'similarity_score': pair['similarity'],
            'title1': pair['title1'],
            'title2': pair['title2']
        })
    
    prompt = f"""
    You are consolidating repetitive old focus blocks into coherent groups.
    
    FOCUS BLOCKS TO ANALYZE:
    {json.dumps(blocks_data, indent=2, default=str)}
    
    SIMILAR PAIRS (high content similarity):
    {json.dumps(pairs_info, indent=2)}
    
    Your task: Group similar/repetitive focus blocks together to eliminate redundancy.
    
    CONSOLIDATION RULES:
    1. **Aggressive consolidation** - Group blocks covering the same topic/concept
    2. **Eliminate redundancy** - Don't keep separate blocks for the same concept
    3. **Preserve unique content** - Only group if they're truly about the same thing
    4. **Consider titles and content** - Look for conceptual overlap, not just word similarity
    5. **Create meaningful group titles** - More general than individual block titles
    6. **Aim for 3-8 blocks per group** for substantial consolidation
    
    EXAMPLES OF GOOD CONSOLIDATION:
    - "Introduction to Probability" + "Probability Basics" + "Understanding Probability" 
      → GROUP: "Probability Fundamentals"
    - "Normal Distribution" + "Gaussian Distribution" + "Bell Curve Distribution"
      → GROUP: "Normal Distribution Theory"
    
    EXAMPLES OF BAD CONSOLIDATION:
    - "Probability" + "Statistics" (different concepts)
    - Grouping unrelated topics just because they're similar words
    
    Return JSON response:
    {{
        "consolidated_groups": [
            {{
                "group_id": 1,
                "unified_title": "Clear, descriptive title for the consolidated concept",
                "block_ids": [1, 3, 7, 12],
                "consolidation_reason": "Why these blocks were grouped together",
                "estimated_duration": 20
            }}
        ]
    }}
    
    IMPORTANT:
    - Every block must be assigned to exactly one group
    - **Prefer fewer, larger groups** over many small groups
    - If blocks cover genuinely different concepts, keep them separate
    - Unified title should capture the essence of all blocks in the group
    - Estimated duration should be sum of individual blocks (but reasonable)
    - Return ONLY valid JSON
    """
    
    print(f"🤖 AI analyzing {len(blocks_data)} focus blocks for consolidation...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at identifying and consolidating redundant educational content. Your goal is to eliminate repetitive focus blocks while preserving all unique educational value."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=3000,
        temperature=0.1  # Low temperature for consistent grouping
    )
    
    ai_response = response.choices[0].message.content.strip()
    
    try:
        # Parse AI response
        if ai_response.startswith('```json'):
            ai_response = ai_response.replace('```json', '').replace('```', '').strip()
        
        ai_data = json.loads(ai_response)
        consolidated_groups_ai = ai_data.get('consolidated_groups', [])
        
        print(f"✅ AI created {len(consolidated_groups_ai)} consolidated groups")
        for group in consolidated_groups_ai:
            print(f"  📚 Group {group.get('group_id')}: '{group.get('unified_title')}' ({len(group.get('block_ids', []))} blocks)")
        
        # Convert AI response to our format with actual block data
        consolidated_groups = []
        for group in consolidated_groups_ai:
            # Get actual block data for this group
            group_blocks_data = []
            for block_id in group.get('block_ids', []):
                for block_data in blocks_data:
                    if block_data['id'] == block_id:
                        group_blocks_data.append(block_data)
                        break
            
            consolidated_groups.append({
                'group_id': group.get('group_id', len(consolidated_groups) + 1),
                'unified_title': group.get('unified_title', 'Consolidated Focus Block'),
                'block_ids': group.get('block_ids', []),
                'blocks_data': group_blocks_data,
                'consolidation_reason': group.get('consolidation_reason', 'AI-determined consolidation'),
                'estimated_duration': group.get('estimated_duration', 15)
            })
        
        return consolidated_groups
        
    except json.JSONDecodeError as e:
        print(f"❌ AI Consolidation JSON Parse Error: {e}")
        print(f"Raw AI Response: {ai_response[:500]}...")
        
        # Fallback: create individual groups
        fallback_groups = []
        for block_data in blocks_data:
            fallback_groups.append({
                'group_id': len(fallback_groups) + 1,
                'unified_title': block_data['title'],
                'block_ids': [block_data['id']],
                'blocks_data': [block_data],
                'consolidation_reason': f'Fallback due to AI error: {str(e)}',
                'estimated_duration': block_data['target_duration']
            })
        
        return fallback_groups

def convert_consolidated_group_to_interactive(consolidated_group, client):
    """
    Convert a group of consolidated focus blocks into one interactive learning block
    """
    import json
    
    # Combine content from all blocks in the group
    combined_content = ""
    all_titles = []
    total_duration = 0
    
    for block_data in consolidated_group['blocks_data']:
        all_titles.append(block_data['title'])
        combined_content += f"\n\nFrom '{block_data['title']}':\n{block_data['content']}\n"
        total_duration += block_data.get('target_duration', 15)
    
    # Limit combined content for API efficiency
    combined_content = combined_content[:2000]
    
    prompt = f"""
    Create a comprehensive interactive learning block by consolidating these related focus blocks:
    
    UNIFIED CONCEPT: {consolidated_group['unified_title']}
    
    ORIGINAL BLOCKS BEING CONSOLIDATED:
    {json.dumps([{'title': block['title'], 'content': block['content'][:300]} for block in consolidated_group['blocks_data']], indent=2)}
    
    COMBINED CONTENT:
    {combined_content}
    
    CONSOLIDATION CONTEXT:
    - Reason for grouping: {consolidated_group['consolidation_reason']}
    - Original blocks: {len(consolidated_group['blocks_data'])} blocks
    - Estimated total duration: {total_duration} minutes
    
    Create a single, comprehensive interactive focus block with this EXACT structure:
    
    {{
        "title": "{consolidated_group['unified_title']}",
        "recap": "Brief connection showing how this consolidated concept builds on previous learning",
        "socratic_intro": {{
            "hook_question": "Engaging question about the unified concept",
            "exploration_questions": ["Question 1", "Question 2", "Question 3"],
            "bridge_to_content": "Transition to comprehensive content"
        }},
        "segments": [
            {{
                "segment_title": "Fundamental Understanding",
                "content": "Core explanation combining key points from all original blocks",
                "key_points": ["Fundamental point 1", "Fundamental point 2", "Fundamental point 3"]
            }},
            {{
                "segment_title": "Deeper Exploration", 
                "content": "More detailed explanation integrating content from original blocks",
                "key_points": ["Advanced concept 1", "Advanced concept 2", "Advanced concept 3"]
            }},
            {{
                "segment_title": "Comprehensive Applications",
                "content": "Practical applications combining examples from all blocks", 
                "key_points": ["Application 1", "Application 2", "Real-world use"]
            }}
        ],
        "qa_items": [
            {{
                "prompt": "Comprehensive understanding question covering the unified concept",
                "ideal_answer": "Answer drawing from all consolidated content",
                "type": "conceptual"
            }},
            {{
                "prompt": "Application question spanning the full concept", 
                "ideal_answer": "Practical answer using consolidated knowledge",
                "type": "application"
            }}
        ],
        "estimated_duration": {min(total_duration, 25)}
    }}
    
    CONSOLIDATION GUIDELINES:
    1. **Integrate all content** - Don't lose important information from any block
    2. **Eliminate redundancy** - Combine overlapping concepts into unified explanations
    3. **Create coherent flow** - Make it one seamless learning experience
    4. **Enhance with Socratic method** - Add engaging discovery-based learning
    5. **Comprehensive coverage** - Address the full scope of the unified concept
    6. **Logical progression** - Build from basic to advanced understanding
    
    IMPORTANT:
    - This replaces {len(consolidated_group['blocks_data'])} separate blocks
    - Preserve all unique educational value from original blocks
    - Make it more comprehensive than any individual block
    - Ensure the learning experience is cohesive and complete
    - Return ONLY valid JSON, no other text
    """
    
    print(f"🤖 Converting consolidated group: {consolidated_group['unified_title']}")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating comprehensive, engaging educational content by consolidating multiple related learning materials into cohesive interactive experiences."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=4000,
        temperature=0.3
    )
    
    ai_response = response.choices[0].message.content.strip()
    
    try:
        # Parse AI response
        if ai_response.startswith('```json'):
            ai_response = ai_response.replace('```json', '').replace('```', '').strip()
        
        interactive_block = json.loads(ai_response)
        
        # Add consolidation metadata
        interactive_block.update({
            'consolidated_from': consolidated_group['block_ids'],
            'original_blocks_count': len(consolidated_group['blocks_data']),
            'consolidation_reason': consolidated_group['consolidation_reason'],
            'migration_timestamp': str(timezone.now()),
            'migration_status': 'consolidated_success'
        })
        
        return interactive_block
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error for consolidated group: {e}")
        
        # Fallback interactive block
        fallback_block = {
            'title': consolidated_group['unified_title'],
            'recap': f"This concept consolidates information from {len(consolidated_group['blocks_data'])} related focus blocks.",
            'socratic_intro': {
                'hook_question': f"What do you think encompasses the broad concept of {consolidated_group['unified_title']}?",
                'exploration_questions': [
                    "How do the different aspects of this concept relate?",
                    "What are the key components we need to understand?",
                    "Why is it important to see this as a unified concept?"
                ],
                'bridge_to_content': "Let's explore this comprehensive concept systematically."
            },
            'segments': [
                {
                    'segment_title': 'Consolidated Understanding',
                    'content': f"This concept brings together {len(consolidated_group['blocks_data'])} related topics: {', '.join(all_titles)}",
                    'key_points': [f"Aspect from {block['title']}" for block in consolidated_group['blocks_data'][:3]]
                }
            ],
            'qa_items': [
                {
                    'prompt': f"Explain how the different aspects of {consolidated_group['unified_title']} relate to each other",
                    'ideal_answer': "These concepts work together to provide a comprehensive understanding.",
                    'type': 'conceptual'
                }
            ],
            'estimated_duration': min(total_duration, 25),
            'consolidated_from': consolidated_group['block_ids'],
            'original_blocks_count': len(consolidated_group['blocks_data']),
            'consolidation_reason': consolidated_group['consolidation_reason'],
            'migration_timestamp': str(timezone.now()),
            'migration_status': 'fallback'
        }
        
        return fallback_block

def migrate_all_focus_blocks(request):
    """
    Migrate ALL old focus blocks across all PDFs
    """
    # Get statistics for display
    pdfs_with_focus_blocks = PDFDocument.objects.filter(focus_blocks__isnull=False).distinct()
    total_old_blocks = FocusBlock.objects.count()
    total_pdfs_with_blocks = pdfs_with_focus_blocks.count()
    estimated_new_blocks = max(1, total_old_blocks // 3)  # Rough estimate after consolidation
    
    if request.method == 'POST':
        try:
            total_migrated = 0
            
            for pdf in pdfs_with_focus_blocks:
                migrated_count = migrate_old_focus_blocks_for_pdf(pdf)
                total_migrated += migrated_count
            
            messages.success(request, f'✅ Successfully migrated {total_migrated} focus blocks across {total_pdfs_with_blocks} PDFs!')
            return redirect('flashcards:debug_focus_chunks')
            
        except Exception as e:
            messages.error(request, f'❌ Migration failed: {str(e)}')
            return redirect('flashcards:migrate_all_focus_blocks')
    
    context = {
        'pdfs_with_focus_blocks': pdfs_with_focus_blocks,
        'total_old_blocks': total_old_blocks,
        'total_pdfs_with_blocks': total_pdfs_with_blocks,
        'estimated_new_blocks': estimated_new_blocks
    }
    return render(request, 'flashcards/migrate_focus_blocks.html', context)

def migrate_focus_blocks(request):
    """
    Migration page for converting old format focus blocks to new interactive format
    Works directly with focus blocks, not PDFs
    """
    context = {
        'old_blocks': [],
        'grouped_blocks': [],
        'converted_blocks': [],
        'migration_stats': {},
        'migration_message': None,
        'debug_info': []  # Add debug info
    }
    
    # Debug: Let's examine the actual structure of focus blocks
    all_blocks = FocusBlock.objects.all()[:10]  # First 10 blocks for debugging
    debug_info = []
    
    for block in all_blocks:
        # Check if compact7_data has segments (new format indicator)
        has_segments = False
        has_old_compact7 = False
        is_migrated = block.title.startswith('[MIGRATED→')  # NEW: Check if already migrated
        compact7_preview = 'None'
        compact7_keys = []
        
        if block.compact7_data:
            compact7_preview = str(block.compact7_data)[:200]
            if isinstance(block.compact7_data, dict):
                has_segments = 'segments' in block.compact7_data
                compact7_keys = list(block.compact7_data.keys())
                
                # Check for old compact7 structure
                old_compact7_keys = ['template', 'revision', 'core', 'qa', 'recap', 'rescue', 'metadata']
                if block.compact7_data.get('template') == 'compact7' and any(key in compact7_keys for key in old_compact7_keys):
                    has_old_compact7 = True
        
        debug_info.append({
            'id': str(block.id)[:8],
            'title': block.title[:50],
            'has_compact7_data': block.compact7_data is not None,
            'compact7_keys': compact7_keys,
            'has_segments': has_segments,
            'has_old_compact7': has_old_compact7,
            'is_migrated': is_migrated,  # NEW: Show migration status
            'has_teacher_script': bool(block.teacher_script),
            'teacher_script_length': len(block.teacher_script) if block.teacher_script else 0,
            'has_legacy_fields': bool(block.teacher_script or block.rescue_reset or block.recap_summary),
            'pdf_name': block.pdf_document.name if block.pdf_document else 'None',
            'format_type': 'Migrated' if is_migrated else ('New (Segments)' if has_segments else ('Old (Compact7)' if has_old_compact7 else ('Legacy (Script)' if bool(block.teacher_script) else 'Empty')))
        })
    
    context['debug_info'] = debug_info
    
    # FIXED: Proper detection logic for old vs new format + exclude migrated blocks
    all_focus_blocks = FocusBlock.objects.exclude(
        title__startswith='[MIGRATED→'  # NEW: Exclude already migrated blocks
    )
    
    old_blocks_list = []
    new_blocks_count = 0
    empty_blocks_count = 0
    migrated_blocks_count = FocusBlock.objects.filter(title__startswith='[MIGRATED→').count()  # NEW: Count migrated
    
    for block in all_focus_blocks:
        is_new_format = False
        is_old_format = False
        
        if block.compact7_data and isinstance(block.compact7_data, dict):
            # New format: has segments
            if 'segments' in block.compact7_data:
                is_new_format = True
            # Old format: has compact7 template structure
            elif (block.compact7_data.get('template') == 'compact7' and 
                  any(key in block.compact7_data for key in ['revision', 'core', 'qa', 'recap', 'rescue'])):
                is_old_format = True
        
        # Legacy format: has teacher_script but no compact7_data
        if not is_new_format and not is_old_format:
            if block.teacher_script or block.rescue_reset or block.recap_summary:
                is_old_format = True
        
        if is_new_format:
            new_blocks_count += 1
        elif is_old_format:
            old_blocks_list.append(block)
        else:
            empty_blocks_count += 1
    
    # Convert back to queryset for consistency
    old_block_ids = [block.id for block in old_blocks_list]
    old_blocks = FocusBlock.objects.filter(id__in=old_block_ids).order_by('pdf_document__name', 'block_order')
    
    context['migration_stats'] = {
        'total_blocks': FocusBlock.objects.count(),  # All blocks including migrated
        'total_old_blocks': len(old_blocks_list),
        'total_new_blocks': new_blocks_count,
        'empty_blocks': empty_blocks_count,
        'migrated_blocks': migrated_blocks_count,  # NEW: Show migrated count
        'pending_migration': len(old_blocks_list)
    }
    
    context['old_blocks'] = old_blocks
    
    # Handle migration actions
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'analyze_similarities':
            try:
                grouped_blocks = analyze_block_similarities(old_blocks)
                context['grouped_blocks'] = grouped_blocks
                context['migration_message'] = f"✅ Analyzed {old_blocks.count()} blocks into {len(grouped_blocks)} groups"
            except Exception as e:
                context['migration_message'] = f"❌ Analysis failed: {str(e)}"
                
        elif action == 'convert_group':
            group_ids = request.POST.getlist('selected_blocks')
            if group_ids:
                try:
                    converted_block = convert_old_blocks_to_new(group_ids)
                    context['converted_blocks'] = [converted_block]
                    context['migration_message'] = f"✅ Converted {len(group_ids)} blocks into 1 new interactive block"
                except Exception as e:
                    context['migration_message'] = f"❌ Conversion failed: {str(e)}"
    
        # NEW: Bulk migration options
        elif action == 'bulk_migrate_individual':
            try:
                converted_blocks = bulk_migrate_individual(old_blocks)
                context['converted_blocks'] = converted_blocks
                context['migration_message'] = f"✅ Bulk migrated {len(converted_blocks)} blocks individually"
            except Exception as e:
                context['migration_message'] = f"❌ Bulk migration failed: {str(e)}"
                
        elif action == 'bulk_migrate_grouped':
            try:
                grouped_blocks = analyze_block_similarities(old_blocks)
                converted_blocks = []
                for group in grouped_blocks:
                    group_block_ids = [str(block.id) for block in group['blocks']]
                    converted_block = convert_old_blocks_to_new(group_block_ids)
                    converted_blocks.append(converted_block)
                
                context['converted_blocks'] = converted_blocks
                context['migration_message'] = f"✅ Bulk migrated {old_blocks.count()} blocks into {len(converted_blocks)} groups"
            except Exception as e:
                context['migration_message'] = f"❌ Bulk grouped migration failed: {str(e)}"
                
        elif action == 'bulk_migrate_by_pdf':
            try:
                converted_blocks = bulk_migrate_by_pdf(old_blocks)
                context['converted_blocks'] = converted_blocks
                context['migration_message'] = f"✅ Bulk migrated by PDF into {len(converted_blocks)} new blocks"
            except Exception as e:
                context['migration_message'] = f"❌ PDF-based migration failed: {str(e)}"
    
    return render(request, 'flashcards/migrate_focus_blocks.html', context)

def bulk_migrate_individual(old_blocks):
    """
    Migrate each old block individually (1:1 conversion)
    """
    converted_blocks = []
    
    for block in old_blocks:
        try:
            converted_block = convert_old_blocks_to_new([str(block.id)])
            converted_blocks.append(converted_block)
        except Exception as e:
            print(f"Failed to migrate block {block.id}: {str(e)}")
            continue
    
    return converted_blocks

def bulk_migrate_by_pdf(old_blocks):
    """
    Group blocks by PDF and migrate each PDF's blocks together
    """
    from collections import defaultdict
    
    # Group blocks by PDF
    pdf_groups = defaultdict(list)
    for block in old_blocks:
        pdf_key = block.pdf_document.name if block.pdf_document else 'No PDF'
        pdf_groups[pdf_key].append(block)
    
    converted_blocks = []
    
    for pdf_name, blocks in pdf_groups.items():
        try:
            # Convert all blocks from this PDF into one new block
            block_ids = [str(block.id) for block in blocks]
            converted_block = convert_old_blocks_to_new(block_ids)
            converted_blocks.append(converted_block)
        except Exception as e:
            print(f"Failed to migrate blocks from {pdf_name}: {str(e)}")
            continue
    
    return converted_blocks

def analyze_block_similarities(old_blocks):
    """
    Group similar old focus blocks together using AI analysis
    Returns list of block groups
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    # Prepare block data for analysis
    blocks_data = []
    for block in old_blocks:
        # Extract content from compact7_data
        content_preview = ""
        if block.compact7_data and isinstance(block.compact7_data, dict):
            # Get core content from compact7 structure
            if 'core' in block.compact7_data:
                content_preview = str(block.compact7_data['core'])[:300]
            elif 'revision' in block.compact7_data:
                content_preview = str(block.compact7_data['revision'])[:300]
        
        # Fallback to teacher_script if available
        if not content_preview and block.teacher_script:
            content_preview = block.teacher_script[:300]
        
        blocks_data.append({
            'id': str(block.id),
            'title': block.title,
            'content_preview': content_preview,
            'pdf_name': block.pdf_document.name if block.pdf_document else 'Unknown',
            'difficulty': block.difficulty_level
        })
    
    # Limit to reasonable batch size to avoid token limits
    if len(blocks_data) > 20:
        blocks_data = blocks_data[:20]
        print(f"⚠️ Limited analysis to first 20 blocks to avoid token limits")
    
    # AI prompt for grouping similar blocks
    prompt = f"""
    Analyze these {len(blocks_data)} focus blocks and group similar ones together.
    
    Focus blocks to analyze:
    {json.dumps(blocks_data, indent=2)[:3000]}  # Limit prompt size
    
    Rules for grouping:
    1. Group blocks that cover the SAME core concept (even if from different PDFs)
    2. Blocks should be similar enough to merge into ONE comprehensive block
    3. Don't group blocks that are too different - better to keep separate
    4. Consider: topic similarity, difficulty level, content overlap
    5. If blocks are very different, create individual groups (1 block per group)
    
    Return ONLY valid JSON with this exact structure:
    {{
        "groups": [
            {{
                "core_concept": "Main concept name",
                "block_ids": ["uuid1", "uuid2"],
                "similarity_reason": "Why these blocks belong together",
                "combined_title": "Suggested title for merged block"
            }}
        ]
    }}
    
    Important: Return ONLY the JSON, no other text or explanation.
    """
    
    try:
        print(f"🤖 Sending AI request for {len(blocks_data)} blocks...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system", 
                "content": "You are a helpful assistant that analyzes educational content and groups similar concepts. Always return valid JSON."
            }, {
                "role": "user", 
                "content": prompt
            }],
            max_tokens=2000,
            temperature=0.1  # Lower temperature for more consistent JSON
        )
        
        response_content = response.choices[0].message.content.strip()
        print(f"🤖 AI Response (first 200 chars): {response_content[:200]}")
        
        # Clean up response - remove any markdown formatting
        if response_content.startswith('```json'):
            response_content = response_content[7:]  # Remove ```json
        if response_content.endswith('```'):
            response_content = response_content[:-3]  # Remove ```
        
        response_content = response_content.strip()
        
        if not response_content:
            raise ValueError("Empty response from OpenAI")
        
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed. Raw response: {response_content}")
            raise ValueError(f"Invalid JSON response from AI: {str(e)}")
        
        if 'groups' not in result:
            raise ValueError("AI response missing 'groups' key")
        
        # Convert to format needed by template
        grouped_blocks = []
        for group in result.get('groups', []):
            if 'block_ids' not in group or not group['block_ids']:
                continue
                
            group_blocks = old_blocks.filter(id__in=group['block_ids'])
            if group_blocks.exists():
                grouped_blocks.append({
                    'core_concept': group.get('core_concept', 'Unknown Concept'),
                    'blocks': group_blocks,
                    'similarity_reason': group.get('similarity_reason', 'Similar content'),
                    'combined_title': group.get('combined_title', 'Combined Block'),
                    'block_count': group_blocks.count()
                })
        
        print(f"✅ Successfully created {len(grouped_blocks)} groups")
        return grouped_blocks
        
    except Exception as e:
        print(f"❌ AI analysis error: {str(e)}")
        # Fallback: create individual groups if AI fails
        print("🔄 Falling back to individual grouping...")
        fallback_groups = []
        for block in old_blocks[:10]:  # Limit fallback to 10 blocks
            fallback_groups.append({
                'core_concept': block.title[:50],
                'blocks': [block],
                'similarity_reason': 'AI analysis failed - individual migration',
                'combined_title': block.title,
                'block_count': 1
            })
        return fallback_groups

def convert_old_blocks_to_new(block_ids):
    """
    Convert a group of old format blocks into one new interactive block
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    
    client = OpenAI(api_key=api_key)
    
    # Get the old blocks
    old_blocks = FocusBlock.objects.filter(id__in=block_ids)
    if not old_blocks.exists():
        raise ValueError("No blocks found for conversion")
    
    # Helper function to clean titles
    def clean_title(title):
        """Remove migration prefixes and clean up title"""
        import re
        # Remove migration prefixes like [MIGRATED→abc12345]
        cleaned = re.sub(r'\[MIGRATED→[a-fA-F0-9]+\]\s*', '', title)
        # Remove numbered prefixes like "1. " or "2.1 "
        cleaned = re.sub(r'^\d+\.?\s*', '', cleaned).strip()
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    # Combine content from all blocks
    combined_content = []
    clean_titles = []  # Track clean titles for better AI analysis
    
    for block in old_blocks:
        # Clean the title before using it
        clean_title_text = clean_title(block.title)
        clean_titles.append(clean_title_text)
        
        block_content = {
            'title': clean_title_text,  # Use cleaned title
            'pdf_source': block.pdf_document.name if block.pdf_document else 'Unknown'
        }
        
        # Extract content from compact7_data or legacy fields
        if block.compact7_data and isinstance(block.compact7_data, dict):
            for key in ['revision', 'core', 'qa', 'recap', 'rescue']:
                if key in block.compact7_data:
                    block_content[key] = str(block.compact7_data[key])[:500]  # Limit length
        
        # Fallback to legacy fields
        if block.teacher_script:
            block_content['teacher_script'] = block.teacher_script[:500]
        if block.recap_summary:
            block_content['recap_summary'] = block.recap_summary[:500]
        if block.rescue_reset:
            block_content['rescue_reset'] = block.rescue_reset[:500]
            
        combined_content.append(block_content)
    
    # Create a better title suggestion based on clean titles
    if len(clean_titles) == 1:
        suggested_title = clean_titles[0]
    else:
        # Find common theme among titles
        common_words = []
        for title in clean_titles:
            words = title.lower().split()
            common_words.extend([w for w in words if len(w) > 3])  # Only meaningful words
        
        if common_words:
            # Use most common meaningful word
            from collections import Counter
            most_common = Counter(common_words).most_common(1)[0][0]
            suggested_title = f"Understanding {most_common.title()}"
        else:
            suggested_title = "Combined Concept Study"
    
    # AI prompt for conversion
    prompt = f"""
    Convert these old format focus blocks into ONE new interactive focus block.
    
    Old blocks to merge:
    {json.dumps(combined_content, indent=2)[:2000]}  # Limit prompt size
    
    Suggested clean title: "{suggested_title}"
    
    Create a new interactive focus block with this structure:
    {{
        "title": "Clean, descriptive title without numbers or prefixes",
        "learning_objectives": ["objective 1", "objective 2"],
        "segments": [
            {{
                "type": "recap",
                "title": "Quick Review",
                "content": "Brief recap of prerequisites",
                "duration_seconds": 60
            }},
            {{
                "type": "socratic_intro",
                "title": "Discovering the Concept",
                "content": "Questions to guide understanding",
                "duration_seconds": 120
            }},
            {{
                "type": "definition",
                "title": "Core Definition",
                "content": "Clear, precise definition",
                "duration_seconds": 90
            }},
            {{
                "type": "example",
                "title": "Concrete Example",
                "content": "Real-world example",
                "duration_seconds": 120
            }},
            {{
                "type": "practice",
                "title": "Apply It",
                "content": "Practice problem or exercise",
                "duration_seconds": 90
            }}
        ],
        "total_duration": 480,
        "difficulty_level": "intermediate"
    }}
    
    Guidelines:
    - Create a CLEAN title without numbers, prefixes, or migration markers
    - Base the title on the core concept being taught
    - Use Socratic method - ask questions to guide discovery
    - Make it interactive and engaging
    - Keep segments focused and time-bounded
    - Combine the best content from all source blocks
    - Ensure logical flow from recap → discovery → definition → example → practice
    
    Return ONLY valid JSON, no other text.
    """
    
    try:
        print(f"🤖 Converting {len(old_blocks)} blocks to new format...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system", 
                "content": "You are an expert educational content designer. Always return valid JSON."
            }, {
                "role": "user", 
                "content": prompt
            }],
            max_tokens=3000,
            temperature=0.2
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Clean up response
        if response_content.startswith('```json'):
            response_content = response_content[7:]
        if response_content.endswith('```'):
            response_content = response_content[:-3]
        
        response_content = response_content.strip()
        
        new_block_data = json.loads(response_content)
        
        # FIXED: Continue with the rest of the function logic
        # Find next available block_order for the PDF
        first_block = old_blocks.first()
        pdf_document = first_block.pdf_document
        
        if pdf_document:
            # Find the highest block_order for this PDF
            max_order = FocusBlock.objects.filter(
                pdf_document=pdf_document
            ).aggregate(
                max_order=models.Max('block_order')
            )['max_order']
            
            next_block_order = (max_order or 0) + 1
        else:
            # If no PDF, use a high number to avoid conflicts
            next_block_order = 9999
        
        # Create the new focus block with safe block_order
        new_block = FocusBlock.objects.create(
            pdf_document=pdf_document,
            main_concept_unit=first_block.main_concept_unit,
            block_order=next_block_order,  # Use next available order
            title=new_block_data.get('title', f'Migrated: {first_block.title}'),
            target_duration=new_block_data.get('total_duration', 420),
            compact7_data=new_block_data,
            difficulty_level=new_block_data.get('difficulty_level', 'intermediate'),
            learning_objectives=new_block_data.get('learning_objectives', [])
        )
        
        # Mark old blocks as migrated (don't change block_order to avoid conflicts)
        for old_block in old_blocks:
            old_block.title = f"[MIGRATED→{str(new_block.id)[:8]}] " + old_block.title
            old_block.save()
        
        print(f"✅ Successfully created new block: {new_block.title} (order: {next_block_order})")
        return new_block
        
    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        raise Exception(f"Conversion failed: {str(e)}")

def pdf_progress(request, pdf_id):
    """
    Show progress/report of PDF processing steps
    """
    try:
        pdf_document = PDFDocument.objects.get(id=pdf_id)
    except PDFDocument.DoesNotExist:
        messages.error(request, "PDF document not found")
        return redirect('flashcards:home')
    

    # Get focus blocks for this PDF
    focus_blocks = FocusBlock.objects.filter(
        pdf_document=pdf_document,
        compact7_data__has_key='segments'
    ).exclude(title__startswith='[MIGRATED→')
    
    processing_info = {
        'pdf_name': pdf_document.name,
        'pdf_size': pdf_document.pdf_file.size if pdf_document.pdf_file else 0,
        'page_count': pdf_document.page_count or 0,
        'word_count': pdf_document.word_count or 0,
        'total_duration': pdf_document.processing_duration or 0,
        'steps': [
            {
                'step': 1,
                'name': 'PDF Processing',
                'status': 'completed',
                'result': f'Extracted text from {pdf_document.page_count or "unknown"} pages'
            },
            {
                'step': 2,
                'name': 'Text Analysis',
                'status': 'completed',
                'result': f'Processed {pdf_document.word_count or "unknown"} words'
            },
            {
                'step': 3,
                'name': 'Focus Blocks Creation',
                'status': 'completed',
                'result': f'Generated {focus_blocks.count()} interactive focus blocks'
            }
        ],
        'focus_blocks_created': [
            {
                'title': block.title,
                'id': str(block.id),
                'segments_count': len(block.compact7_data.get('segments', [])),
                'difficulty': block.difficulty_level,
                'duration': block.target_duration
            }
            for block in focus_blocks
        ],
        'success': pdf_document.processed
    }

    context = {
        'pdf_document': pdf_document,
        'processing_info': processing_info,
        'total_steps': len(processing_info.get('steps')),
        'completed_steps': len([s for s in processing_info.get('steps', []) if s.get('status') == 'completed']),
        'has_errors': len(processing_info.get('errors', [])) > 0,
        'focus_blocks': processing_info.get('focus_blocks_created', []),
        'processing_duration': processing_info.get('processing_duration', 0)
    }

    return render(request, 'flashcards/pdf_progress.html', context)

def generate_knowledge_graph_from_relationships(focus_blocks, relationships):
    """
    Generate knowledge graph using persistent FocusBlockRelationship data
    """
    try:
        print(f"🕸️ Generating knowledge graph from {relationships.count() if hasattr(relationships, 'count') else len(relationships)} stored relationships...")
        
        # Step 1: Create nodes from focus blocks
        nodes = []
        for i, block in enumerate(focus_blocks):
            # Extract chunk roles from compact7_data if available
            chunk_roles = []
            if hasattr(block, 'compact7_data') and block.compact7_data:
                segments = block.compact7_data.get('segments', [])
                chunk_roles = [seg.get('type', 'unknown') for seg in segments]
            
            # Determine category based on difficulty or content analysis
            category = 'intermediate'  # default
            if block.difficulty_level == 'beginner':
                category = 'foundational'
            elif block.difficulty_level == 'advanced':
                category = 'advanced'
            
            node = {
                'id': f"block_{block.id}",
                'label': block.title,
                'title': block.title,
                'group_id': str(block.id),
                'estimated_duration': block.target_duration or 420,
                'chunk_count': len(chunk_roles) or 1,
                'chunk_roles': chunk_roles,
                'category': category,
                'size': max(20, min(60, len(chunk_roles) * 10)) if chunk_roles else 30,
                'level': i  # For hierarchical layout
            }
            nodes.append(node)
        
        # Step 2: Create edges from persistent relationships
        edges = []
        edge_id = 0
        
        for rel in relationships:
            from_id = f"block_{rel.from_block.id}"
            to_id = f"block_{rel.to_block.id}"
            
            edge = {
                'id': f"edge_{edge_id}",
                'from': from_id,
                'to': to_id,
                'label': rel.relationship_type.replace('_', ' ').title(),
                'relationship': rel.relationship_type.replace('_', ' ').title(),
                'strength': rel.edge_strength,
                'confidence': rel.confidence,
                'description': rel.description,
                'reasoning': rel.educational_reasoning,
                'width': max(1, rel.edge_strength * 5),  # Edge width based on strength
                'arrows': 'to',  # Directed graph
                'physics': rel.edge_strength > 0.3  # Only strong connections affect layout
            }
            edges.append(edge)
            edge_id += 1
        
        # Step 3: Create graph metadata
        graph_metadata = {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'relationship_types': list(set(edge['relationship'] for edge in edges)) if edges else [],
            'avg_confidence': sum(edge['confidence'] for edge in edges) / len(edges) if edges else 0,
            'strong_connections': len([e for e in edges if e['strength'] > 0.7])
        }
        
        knowledge_graph = {
            'nodes': nodes,
            'edges': edges,
            'metadata': graph_metadata
        }
        
        print(f"✅ Generated knowledge graph: {len(nodes)} nodes, {len(edges)} edges")
        return knowledge_graph
        
    except Exception as e:
        print(f"❌ Error generating knowledge graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'nodes': [], 'edges': [], 'metadata': {}}


def get_node_color(chunk_roles):
    """
    Determine node color based on chunk roles and content type
    """
    if not chunk_roles:
        return '#6c757d'  # Gray for unknown
    
    # Priority order for coloring
    if 'Definition' in chunk_roles or 'Statement' in chunk_roles:
        return '#28a745'  # Green for foundational
    elif 'Example' in chunk_roles or 'Application' in chunk_roles:
        return '#17a2b8'  # Cyan for practical
    elif 'Proof' in chunk_roles or 'Derivation' in chunk_roles:
        return '#dc3545'  # Red for advanced
    elif 'Algorithm' in chunk_roles or 'Procedure' in chunk_roles:
        return '#6f42c1'  # Purple for procedural
    else:
        return '#ffc107'  # Yellow for intermediate


def get_edge_color(relationship_type):
    """
    Get edge color based on relationship type
    """
    color_map = {
        'prerequisite': '#dc3545',      # Red
        'builds_on': '#007bff',         # Blue  
        'related': '#28a745',           # Green
        'applies_to': '#6f42c1',        # Purple
        'compares_with': '#fd7e14',     # Orange
        'specializes': '#20c997'        # Teal
    }
    return color_map.get(relationship_type, '#6c757d')  # Default gray

def generate_optimal_study_order(focus_blocks=None, user_knowledge=None):
    """
    Generate optimal study order using knowledge graph relationships
    """
    from collections import defaultdict, deque
    
    if focus_blocks is None:
        focus_blocks = FocusBlock.objects.filter(
            compact7_data__has_key='segments'
        ).exclude(title__startswith='[MIGRATED→')
    
    # Get all relationships
    relationships = FocusBlockRelationship.objects.filter(
        from_block__in=focus_blocks,
        to_block__in=focus_blocks
    ).select_related('from_block', 'to_block')
    
    # Build dependency graph
    graph = defaultdict(list)  # block_id -> [dependent_blocks]
    in_degree = defaultdict(int)  # block_id -> number of prerequisites
    blocks_map = {str(block.id): block for block in focus_blocks}
    
    # Initialize all blocks with 0 in-degree
    for block in focus_blocks:
        in_degree[str(block.id)] = 0
    
    # Build graph from prerequisites and builds_on relationships
    for rel in relationships:
        if rel.relationship_type in ['prerequisite', 'builds_on']:
            from_id = str(rel.from_block.id)
            to_id = str(rel.to_block.id)
            
            if rel.relationship_type == 'prerequisite':
                # from_block is prerequisite TO to_block
                graph[from_id].append(to_id)
                in_degree[to_id] += 1
            elif rel.relationship_type == 'builds_on':
                # to_block builds ON from_block  
                graph[from_id].append(to_id)
                in_degree[to_id] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque()
    study_order = []
    
    # Start with blocks that have no prerequisites
    for block_id, degree in in_degree.items():
        if degree == 0:
            queue.append(block_id)
    
    while queue:
        current_id = queue.popleft()
        study_order.append(blocks_map[current_id])
        
        # Remove this block and update dependencies
        for dependent_id in graph[current_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)
    
    # Handle cycles or remaining blocks
    remaining_blocks = [blocks_map[bid] for bid in blocks_map if bid not in [str(b.id) for b in study_order]]
    if remaining_blocks:
        # Sort remaining by difficulty then by creation order
        remaining_blocks.sort(key=lambda x: (
            {'beginner': 1, 'intermediate': 2, 'advanced': 3}.get(x.difficulty_level, 2),
            x.created_at
        ))
        study_order.extend(remaining_blocks)
    
    return study_order


def generate_study_schedule_view(request):
    """
    View to generate and display intelligent study schedules
    """
    study_orders = {}
    focus_blocks = FocusBlock.objects.filter(
        compact7_data__has_key='segments'
    ).exclude(title__startswith='[MIGRATED→')
    
    if focus_blocks.count() < 2:
        messages.error(request, "Need at least 2 focus blocks to generate study order")
        return redirect('flashcards:all_focus_blocks')
    
    # Generate different study orders
    study_orders['prerequisite_based'] = generate_optimal_study_order(focus_blocks)
    study_orders['difficulty_progression'] = generate_difficulty_based_order(focus_blocks)
    study_orders['clustered_concepts'] = generate_concept_clustered_order(focus_blocks)
    
    # Calculate estimated study times
    for order_type, blocks in study_orders.items():
        total_time = sum(block.target_duration or 420 for block in blocks)
        study_orders[order_type] = {
            'blocks': blocks,
            'total_time_minutes': total_time / 60,
            'estimated_days': max(1, int(total_time / 60 / 120))  # Assuming 2 hours/day
        }
    
    # Get relationships for context
    relationships = FocusBlockRelationship.objects.filter(
        from_block__in=focus_blocks,
        to_block__in=focus_blocks
    ).select_related('from_block', 'to_block')
    
    context = {
        'study_orders': study_orders,
        'total_blocks': focus_blocks.count(),
        'total_relationships': relationships.count(),
        'focus_blocks': focus_blocks
    }
    
    return render(request, 'flashcards/study_schedule.html', context)

def generate_difficulty_based_order(focus_blocks):
    """
    Order blocks by difficulty progression while respecting some prerequisites
    """
    # Group by difficulty
    beginner_blocks = [b for b in focus_blocks if b.difficulty_level == 'beginner']
    intermediate_blocks = [b for b in focus_blocks if b.difficulty_level == 'intermediate'] 
    advanced_blocks = [b for b in focus_blocks if b.difficulty_level == 'advanced']
    
    # Sort each group by creation time (proxy for document order)
    beginner_blocks.sort(key=lambda x: x.created_at)
    intermediate_blocks.sort(key=lambda x: x.created_at)
    advanced_blocks.sort(key=lambda x: x.created_at)
    
    return beginner_blocks + intermediate_blocks + advanced_blocks


def generate_concept_clustered_order(focus_blocks):
    """
    Group related concepts together for deep learning
    """
    from collections import defaultdict
    
    # Get "related" relationships
    related_rels = FocusBlockRelationship.objects.filter(
        from_block__in=focus_blocks,
        to_block__in=focus_blocks,
        relationship_type='related'
    )
    
    # Build clusters of related concepts
    clusters = defaultdict(set)
    block_to_cluster = {}
    
    for rel in related_rels:
        from_id = str(rel.from_block.id)
        to_id = str(rel.to_block.id)
        
        # Merge clusters if both blocks already have clusters
        if from_id in block_to_cluster and to_id in block_to_cluster:
            cluster1 = block_to_cluster[from_id]
            cluster2 = block_to_cluster[to_id]
            if cluster1 != cluster2:
                # Merge clusters
                clusters[cluster1].update(clusters[cluster2])
                for block_id in clusters[cluster2]:
                    block_to_cluster[block_id] = cluster1
                del clusters[cluster2]
        elif from_id in block_to_cluster:
            cluster = block_to_cluster[from_id]
            clusters[cluster].add(to_id)
            block_to_cluster[to_id] = cluster
        elif to_id in block_to_cluster:
            cluster = block_to_cluster[to_id]
            clusters[cluster].add(from_id)
            block_to_cluster[from_id] = cluster
        else:
            # Create new cluster
            cluster_id = len(clusters)
            clusters[cluster_id] = {from_id, to_id}
            block_to_cluster[from_id] = cluster_id
            block_to_cluster[to_id] = cluster_id
    
    # Add single blocks (not related to anything)
    blocks_map = {str(b.id): b for b in focus_blocks}
    for block in focus_blocks:
        if str(block.id) not in block_to_cluster:
            cluster_id = len(clusters)
            clusters[cluster_id] = {str(block.id)}
            block_to_cluster[str(block.id)] = cluster_id
    
    # Sort clusters by average difficulty and size
    ordered_blocks = []
    for cluster_blocks in clusters.values():
        cluster_focus_blocks = [blocks_map[bid] for bid in cluster_blocks]
        # Sort within cluster by difficulty
        cluster_focus_blocks.sort(key=lambda x: (
            {'beginner': 1, 'intermediate': 2, 'advanced': 3}.get(x.difficulty_level, 2),
            x.created_at
        ))
        ordered_blocks.extend(cluster_focus_blocks)
    
    return ordered_blocks


def generate_adaptive_study_path(focus_blocks, known_concepts=None, learning_style='comprehensive'):
    """
    Generate personalized study path based on user's existing knowledge
    """
    if known_concepts is None:
        known_concepts = set()
    
    # Filter out blocks user already knows
    unknown_blocks = [b for b in focus_blocks if str(b.id) not in known_concepts]
    
    if learning_style == 'just_in_time':
        # Learn concepts only when needed for something specific
        return generate_minimal_prerequisite_path(unknown_blocks)
    elif learning_style == 'depth_first':
        # Deep dive into related concepts
        return generate_concept_clustered_order(unknown_blocks)
    else:  # comprehensive
        return generate_optimal_study_order(unknown_blocks)


def calculate_learning_path_metrics(study_order):
    """
    Calculate metrics for a given study path
    """
    if not study_order:
        return {}
    
    total_duration = sum(block.target_duration or 420 for block in study_order)
    
    # Calculate prerequisite satisfaction score
    relationships = FocusBlockRelationship.objects.filter(
        from_block__in=study_order,
        to_block__in=study_order,
        relationship_type='prerequisite'
    )
    
    block_positions = {str(block.id): i for i, block in enumerate(study_order)}
    satisfied_prerequisites = 0
    total_prerequisites = relationships.count()
    
    for rel in relationships:
        from_pos = block_positions.get(str(rel.from_block.id))
        to_pos = block_positions.get(str(rel.to_block.id))
        if from_pos is not None and to_pos is not None and from_pos < to_pos:
            satisfied_prerequisites += 1
    
    prerequisite_score = (satisfied_prerequisites / total_prerequisites * 100) if total_prerequisites > 0 else 100
    
    return {
        'total_blocks': len(study_order),
        'total_duration_minutes': total_duration / 60,
        'estimated_study_days': max(1, int(total_duration / 60 / 90)),  # 1.5 hours/day
        'prerequisite_satisfaction': prerequisite_score,
        'difficulty_progression': calculate_difficulty_progression_score(study_order),
        'concept_coherence': calculate_concept_coherence_score(study_order)
    }

def start_study_path(request, path_type):
    """
    Start a study session with the selected path type
    """
    # Get focus blocks
    focus_blocks = FocusBlock.objects.filter(
        compact7_data__has_key='segments'
    ).exclude(title__startswith='[MIGRATED→')
    
    if focus_blocks.count() < 1:
        messages.error(request, "No focus blocks available for study")
        return redirect('flashcards:all_focus_blocks')
    
    # Generate the appropriate study order
    if path_type == 'prerequisite_based':
        study_order = generate_optimal_study_order(focus_blocks)
        path_name = "Prerequisite-Based Learning Path"
        path_description = "Following knowledge dependencies for systematic learning"
    elif path_type == 'difficulty_progression':
        study_order = generate_difficulty_based_order(focus_blocks)
        path_name = "Difficulty Progression Path"
        path_description = "Building confidence with gradually increasing difficulty"
    elif path_type == 'clustered_concepts':
        study_order = generate_concept_clustered_order(focus_blocks)
        path_name = "Concept Clusters Path"
        path_description = "Deep learning through related concept groups"
    else:
        messages.error(request, f"Unknown study path type: {path_type}")
        return redirect('flashcards:study_planner')
    
    if not study_order:
        messages.error(request, "Could not generate study order")
        return redirect('flashcards:study_planner')
    
    # Create or get study session
    from django.contrib.sessions.models import Session
    import uuid
    
    # Store study path in session
    session_key = f'study_path_{uuid.uuid4().hex[:8]}'
    request.session[session_key] = {
        'path_type': path_type,
        'path_name': path_name,
        'path_description': path_description,
        'block_order': [str(block.id) for block in study_order],
        'current_index': 0,
        'completed_blocks': [],
        'started_at': str(timezone.now()),
    }
    
    # Redirect to first block
    first_block = study_order[0]
    messages.success(request, f"Starting {path_name}: {first_block.title}")
    
    # Redirect to the focus block study page with session context
    return redirect('flashcards:focus_block_study_with_path', 
                   block_id=first_block.id, 
                   session_key=session_key)


def focus_block_study_with_path(request, block_id, session_key):
    """
    Study a focus block within a structured learning path
    """
    try:
        focus_block = FocusBlock.objects.get(id=block_id)
    except FocusBlock.DoesNotExist:
        messages.error(request, "Focus block not found")
        return redirect('flashcards:study_planner')
    
    # Get study path from session
    study_path = request.session.get(session_key)
    if not study_path:
        messages.warning(request, "Study session expired. Starting individual block study.")
        return redirect('flashcards:focus_block_study', block_id=block_id)
    
    # Get current position in path
    current_index = study_path['current_index']
    total_blocks = len(study_path['block_order'])
    
    # Get next/previous blocks
    next_block = None
    prev_block = None
    
    if current_index < total_blocks - 1:
        next_block_id = study_path['block_order'][current_index + 1]
        try:
            next_block = FocusBlock.objects.get(id=next_block_id)
        except FocusBlock.DoesNotExist:
            pass
    
    if current_index > 0:
        prev_block_id = study_path['block_order'][current_index - 1]
        try:
            prev_block = FocusBlock.objects.get(id=prev_block_id)
        except FocusBlock.DoesNotExist:
            pass
    
    # Calculate progress
    progress_percentage = (current_index / total_blocks) * 100 if total_blocks > 0 else 0
    
    # Get related blocks for context
    related_relationships = FocusBlockRelationship.objects.filter(
        models.Q(from_block=focus_block) | models.Q(to_block=focus_block)
    ).select_related('from_block', 'to_block')
    
    context = {
        'focus_block': focus_block,
        'study_path': study_path,
        'session_key': session_key,
        'current_index': current_index + 1,  # 1-based for display
        'total_blocks': total_blocks,
        'progress_percentage': progress_percentage,
        'next_block': next_block,
        'prev_block': prev_block,
        'related_relationships': related_relationships,
        'compact7_data': focus_block.compact7_data,
    }
    
    return render(request, 'flashcards/focus_block_study_with_path.html', context)


def complete_study_path_block(request, block_id, session_key):
    """
    Mark a block as completed in the study path and advance to next
    """
    if request.method != 'POST':
        return redirect('flashcards:focus_block_study_with_path', 
                       block_id=block_id, session_key=session_key)
    
    # Get study path from session
    study_path = request.session.get(session_key)
    if not study_path:
        messages.error(request, "Study session expired")
        return redirect('flashcards:study_planner')
    
    # Mark current block as completed
    if str(block_id) not in study_path['completed_blocks']:
        study_path['completed_blocks'].append(str(block_id))
    
    # Advance to next block
    current_index = study_path['current_index']
    total_blocks = len(study_path['block_order'])
    
    if current_index < total_blocks - 1:
        # Move to next block
        study_path['current_index'] = current_index + 1
        request.session[session_key] = study_path
        
        next_block_id = study_path['block_order'][current_index + 1]
        messages.success(request, f"Block completed! Moving to next: {current_index + 2}/{total_blocks}")
        
        return redirect('flashcards:focus_block_study_with_path', 
                       block_id=next_block_id, session_key=session_key)
    else:
        # Path completed!
        completion_time = timezone.now()
        start_time = datetime.fromisoformat(study_path['started_at'].replace('Z', '+00:00'))
        duration = completion_time - start_time
        
        messages.success(request, 
                        f"🎉 Congratulations! You completed the {study_path['path_name']} "
                        f"in {duration.total_seconds() / 3600:.1f} hours!")
        
        # Clean up session
        del request.session[session_key]
        
        return redirect('flashcards:study_planner')


# Also add missing imports and helper functions
def calculate_difficulty_progression_score(study_order):
    """Calculate how well the study order follows difficulty progression"""
    if len(study_order) < 2:
        return 100
    
    difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    progression_violations = 0
    
    for i in range(len(study_order) - 1):
        current_difficulty = difficulty_map.get(study_order[i].difficulty_level, 2)
        next_difficulty = difficulty_map.get(study_order[i + 1].difficulty_level, 2)
        
        # Count violations (going from harder to easier)
        if next_difficulty < current_difficulty:
            progression_violations += 1
    
    max_possible_violations = len(study_order) - 1
    score = ((max_possible_violations - progression_violations) / max_possible_violations) * 100
    return max(0, score)


def calculate_concept_coherence_score(study_order):
    """Calculate how well grouped related concepts are"""
    if len(study_order) < 2:
        return 100
    
    # Get related relationships
    related_rels = FocusBlockRelationship.objects.filter(
        from_block__in=study_order,
        to_block__in=study_order,
        relationship_type='related'
    )
    
    if not related_rels.exists():
        return 100  # Perfect score if no related concepts
    
    block_positions = {str(block.id): i for i, block in enumerate(study_order)}
    coherence_score = 0
    total_relationships = related_rels.count()
    
    for rel in related_rels:
        from_pos = block_positions.get(str(rel.from_block.id))
        to_pos = block_positions.get(str(rel.to_block.id))
        
        if from_pos is not None and to_pos is not None:
            # Closer together = higher coherence
            distance = abs(from_pos - to_pos)
            max_distance = len(study_order) - 1
            closeness_score = (max_distance - distance) / max_distance
            coherence_score += closeness_score
    
    return (coherence_score / total_relationships) * 100 if total_relationships > 0 else 100

def debug_focus_block_data(request):
    """Debug what's actually in focus block compact7_data"""
    from django.http import JsonResponse
    
    block = FocusBlock.objects.first()
    if not block:
        return JsonResponse({'error': 'No focus blocks found'})
    
    data = {
        'block_title': block.title,
        'compact7_data_keys': list(block.compact7_data.keys()) if block.compact7_data else [],
        'has_segments': 'segments' in (block.compact7_data or {}),
        'segments_count': len(block.compact7_data.get('segments', [])) if block.compact7_data else 0,
    }
    
    # Get structure of first segment
    if block.compact7_data and 'segments' in block.compact7_data and block.compact7_data['segments']:
        first_segment = block.compact7_data['segments'][0]
        data['first_segment'] = first_segment
        data['first_segment_keys'] = list(first_segment.keys())
    
    return JsonResponse(data, indent=2)

def advanced_focus_study(request, block_id):
    """Advanced study interface for individual focus blocks with notes and analytics"""
    focus_block = get_object_or_404(FocusBlock, id=block_id)
    
    # Create or get focus session for detailed tracking
    focus_session, created = FocusSession.objects.get_or_create(
        focus_block=focus_block,
        status__in=['active', 'paused'],
        defaults={
            'status': 'active',
            'current_segment': 0,
        }
    )
    
    if created:
        focus_session.block_start_times = {str(focus_block.id): timezone.now().isoformat()}
        focus_session.save()
        print(f"✅ Created new FocusSession: {focus_session.id}")
    
    # DEBUG: Check what data we're getting
    segments = focus_block.get_segments()
    qa_items = focus_block.get_qa_items()
    
    print(f"🔍 ADVANCED STUDY DEBUG:")
    print(f"   Block: {focus_block.title}")
    print(f"   Segments count: {len(segments)}")
    print(f"   QA items count: {len(qa_items)}")
    print(f"   Duration: {focus_block.get_estimated_duration_display()}")
    if segments:
        print(f"   First segment keys: {list(segments[0].keys()) if segments[0] else 'Empty segment'}")
    if focus_block.compact7_data:
        print(f"   compact7_data keys: {list(focus_block.compact7_data.keys())}")
        # Check for Q&A in different locations
        for key in ['qa', 'q_and_a', 'questions', 'quiz']:
            if key in focus_block.compact7_data:
                print(f"   Found {key}: {len(focus_block.compact7_data[key]) if isinstance(focus_block.compact7_data[key], list) else 'Not a list'}")
    
    context = {
        'focus_block': focus_block,
        'focus_session': focus_session,
        'current_block': focus_block,  # For template compatibility
        'segments': segments,
        'qa_items': qa_items,
        'progress_percentage': focus_session.get_completion_percentage(),
    }
    
    return render(request, 'flashcards/advanced_focus_study.html', context)


def complete_advanced_session(request, session_id):
    """Handle completion of advanced focus session with notes and analytics"""
    focus_session = get_object_or_404(FocusSession, id=session_id)
    focus_block = focus_session.focus_block
    
    if request.method == 'POST':
        # Process completion data
        proficiency_score = request.POST.get('proficiency_score')
        difficulty_rating = request.POST.get('difficulty_rating')
        learning_notes = request.POST.get('learning_notes', '')
        confusion_points = request.POST.getlist('confusion_points')
        
        # Update session
        focus_session.status = 'completed'
        focus_session.completed_at = timezone.now()
        focus_session.proficiency_score = int(proficiency_score) if proficiency_score else None
        focus_session.difficulty_rating = int(difficulty_rating) if difficulty_rating else None
        focus_session.learning_notes = learning_notes
        focus_session.confusion_points = [int(cp) for cp in confusion_points]
        
        # Calculate study time
        if focus_session.started_at:
            focus_session.total_study_time = (focus_session.completed_at - focus_session.started_at).total_seconds()
        
        focus_session.save()
        
        # Also mark as completed in the original system for compatibility
        try:
            study_session, created = StudySession.objects.get_or_create(
                pdf_document=focus_block.pdf_document,
                session_type='focus_blocks',
                defaults={'session_id': uuid.uuid4()}
            )
            study_session.completed_focus_blocks.add(focus_block)
            study_session.save()
        except Exception as e:
            print(f"Error updating StudySession: {e}")
        
        messages.success(request, f"🎉 Excellent work! You completed '{focus_block.title}' with notes and analytics saved.")
        return redirect('flashcards:all_focus_blocks')
    
    context = {
        'focus_session': focus_session,
        'focus_block': focus_block,
    }
    
    return render(request, 'flashcards/complete_session.html', context)


def update_session_progress(request, session_id):
    """AJAX endpoint to update session progress and save notes in real-time"""
    if request.method == 'POST':
        focus_session = get_object_or_404(FocusSession, id=session_id)
        
        # Update current segment
        current_segment = request.POST.get('current_segment')
        if current_segment is not None:
            focus_session.current_segment = int(current_segment)
        
        # Update segment completion
        completed_segments = request.POST.getlist('completed_segments')
        if completed_segments:
            focus_session.segments_completed = [int(seg) for seg in completed_segments]
        
        # Update real-time notes
        temp_notes = request.POST.get('temp_notes')
        if temp_notes is not None:
            # Store temporary notes in a session field or cache
            request.session[f'temp_notes_{session_id}'] = temp_notes
        
        focus_session.save()
        
        return JsonResponse({
            'success': True,
            'current_segment': focus_session.current_segment,
            'completion_percentage': focus_session.get_completion_percentage()
        })
    
    return JsonResponse({'success': False})


def session_analytics_api(request, session_id):
    """API endpoint for real-time session analytics"""
    focus_session = get_object_or_404(FocusSession, id=session_id)
    
    # Calculate current study time
    current_time = 0
    if focus_session.started_at:
        current_time = (timezone.now() - focus_session.started_at).total_seconds()
    
    return JsonResponse({
        'session_id': str(focus_session.id),
        'current_segment': focus_session.current_segment,
        'segments_completed': focus_session.segments_completed,
        'completion_percentage': focus_session.get_completion_percentage(),
        'study_time_seconds': current_time,
        'status': focus_session.status,
    })

def test_advanced_study(request):
    """Quick test to see if the advanced study view works"""
    from django.http import HttpResponse
    return HttpResponse("<h1>Advanced Study View Works!</h1><p>URL routing is working correctly.</p>")

@csrf_exempt
def mark_segment_complete(request):
    print("🔍 mark_segment_complete called")
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            segment_index = data.get('segment_index')
            
            # Get the session and mark segment complete
            session = FocusSession.objects.get(id=session_id)
            session.mark_segment_completed(segment_index, time_spent=0)
            session.save()
            
            return JsonResponse({
                'success': True,
                'progress': round(session.get_completion_percentage(), 1)
            })
            
        except FocusSession.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Session not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})
