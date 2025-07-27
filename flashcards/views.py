from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.views.decorators.http import require_http_methods
from .models import Folder, Flashcard, StudySession, PDFDocument, TextChunk, ChunkLabel, ConceptUnit, FocusBlock
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

def process_pdf_complete(pdf_document: PDFDocument) -> Tuple[bool, str, Dict]:
    """PDF processing with working deduplication"""
    try:
        print(f"🚀 Processing: {pdf_document.name}")
        
        # Step 1: Basic text extraction first
        print("📄 Step 1: Text extraction...")
        from .pdf_service import extract_pdf_text
        text, page_count, extract_success, extract_error = extract_pdf_text(pdf_document)
        
        if not extract_success:
            return False, extract_error, {}
        
        print("✅ Text extraction successful")
        
        # Step 2: Update basic PDF info
        pdf_document.extracted_text = text
        pdf_document.page_count = page_count
        pdf_document.word_count = len(text.split())
        pdf_document.save()
        
        # Step 3: ✅ CHECK FOR DUPLICATES
        print("🔍 Checking for duplicates...")
        duplicate_pdf = check_for_duplicate_content(pdf_document)
        
        if duplicate_pdf:
            print(f"📋 Duplicate detected! Copying from: {duplicate_pdf.name}")
            copy_stats = copy_pdf_data(duplicate_pdf, pdf_document)
            
            return True, f"Duplicate detected! Copied from '{duplicate_pdf.name}'", copy_stats
        
        # Step 4: Process as unique PDF
        print("🆕 Processing as unique PDF...")
        from .pdf_service import PDFTextExtractor
        extractor = PDFTextExtractor()
        
        success, error_msg, stats = extractor.extract_and_process_pdf(
            pdf_document, analyze_concepts=True
        )
        
        if not success:
            return False, error_msg, {}
        
        # Step 5: Generate focus blocks
        print("🎯 Generating focus blocks...")
        from .focus_block_service import generate_focus_blocks
        blocks_success, blocks_message = generate_focus_blocks(pdf_document)
        
        if not blocks_success:
            return False, f"Focus block generation failed: {blocks_message}", stats
        
        stats.update({
            'focus_blocks': pdf_document.focus_blocks.count(),
            'is_duplicate': False
        })
        
        return True, blocks_message, stats
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"💥 Error: {error_msg}")
        pdf_document.processing_error = error_msg
        pdf_document.save()
        return False, error_msg, {}


def check_for_duplicate_content(pdf_document: PDFDocument) -> Optional[PDFDocument]:
    """Simple duplicate detection based on content"""
    import hashlib
    from difflib import SequenceMatcher
    
    if not pdf_document.extracted_text:
        return None
    
    # Create content hash
    content_hash = hashlib.sha256(pdf_document.extracted_text.encode('utf-8')).hexdigest()
    
    # Check for exact match first
    exact_match = PDFDocument.objects.filter(
        processed=True,
        focus_blocks__isnull=False  # Only match PDFs that have focus blocks
    ).exclude(id=pdf_document.id).first()
    
    if exact_match and exact_match.extracted_text:
        exact_hash = hashlib.sha256(exact_match.extracted_text.encode('utf-8')).hexdigest()
        if content_hash == exact_hash:
            print(f"✅ Found exact hash match: {exact_match.name}")
            return exact_match
    
    # Check for similarity (80%+ match)
    threshold = 0.8
    existing_pdfs = PDFDocument.objects.filter(
        processed=True,
        focus_blocks__isnull=False
    ).exclude(id=pdf_document.id)
    
    for existing_pdf in existing_pdfs:
        if existing_pdf.extracted_text:
            similarity = SequenceMatcher(
                None, 
                pdf_document.extracted_text[:2000],  # First 2000 chars
                existing_pdf.extracted_text[:2000]
            ).ratio()
            
            if similarity >= threshold:
                print(f"📊 Found similar PDF: {existing_pdf.name} ({similarity:.1%} similar)")
                return existing_pdf
    
    return None


def copy_pdf_data(source_pdf: PDFDocument, target_pdf: PDFDocument) -> Dict:
    """Copy all data from source PDF to target PDF"""
    from .models import TextChunk, ConceptUnit, ChunkLabel, FocusBlock
    
    # Copy basic info
    target_pdf.cleaned_text = source_pdf.cleaned_text
    target_pdf.processed = True
    target_pdf.concepts_analyzed = True
    target_pdf.is_duplicate = True
    target_pdf.duplicate_of = source_pdf
    target_pdf.save()
    
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
    
    # Copy concept units and labels
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
    
    # Copy focus blocks
    blocks_copied = 0
    target_concepts = list(target_pdf.concept_units.all().order_by('concept_order'))
    
    for i, block in enumerate(source_pdf.focus_blocks.all().order_by('block_order')):
        main_concept = target_concepts[i] if i < len(target_concepts) else target_concepts[0]
        revision_concept = target_concepts[i-1] if i > 0 and i-1 < len(target_concepts) else None
        
        FocusBlock.objects.create(
            pdf_document=target_pdf,
            main_concept_unit=main_concept,
            revision_concept_unit=revision_concept,
            block_order=block.block_order,
            title=block.title,
            target_duration=block.target_duration,
            compact7_data=block.compact7_data,  # Copy the complete Compact7 JSON
            difficulty_level=block.difficulty_level,
            learning_objectives=block.learning_objectives
        )
        blocks_copied += 1
    
    print(f"📋 Copied: {chunks_copied} chunks, {concepts_copied} concepts, {blocks_copied} focus blocks")
    
    return {
        'is_duplicate': True,
        'chunks': chunks_copied,
        'concepts': concepts_copied,
        'focus_blocks': blocks_copied
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
        'concepts_analyzed_flag': pdf_document.concepts_analyzed,
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
    session_id = request.session.get(f'focus_session_{pdf_id}')
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
        request.session[f'focus_session_{pdf_id}'] = str(study_session.session_id)
    
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
    """Display all focus blocks from all PDFs in a unified teaching interface"""
    
    # Get all focus blocks ordered by creation and block order
    all_blocks = FocusBlock.objects.select_related(
        'pdf_document', 'main_concept_unit'
    ).order_by('pdf_document__created_at', 'block_order')
    
    if not all_blocks.exists():
        context = {'no_blocks': True}
        return render(request, 'flashcards/all_focus_blocks.html', context)
    
    # Prepare blocks for display
    formatted_blocks = []
    for block in all_blocks:
        # Get all content for this block
        segments = block.get_segments()
        qa_items = block.get_qa_items()
        revision_data = block.get_revision_data()
        recap_data = block.get_recap_data()
        rescue_data = block.get_rescue_data()
        
        formatted_blocks.append({
            'block': block,
            'segments': segments,
            'qa_items': qa_items,
            'revision_data': revision_data,
            'recap_data': recap_data,
            'rescue_data': rescue_data,
            'source_pdf': block.pdf_document.name,
            'total_segments': len(segments),
            'total_qa': len(qa_items),
        })
    
    # Calculate total study time
    total_time = sum(block.target_duration for block in all_blocks) / 60  # Convert to minutes
    
    context = {
        'formatted_blocks': formatted_blocks,
        'total_blocks': all_blocks.count(),
        'total_study_time': total_time,
        'unique_pdfs': all_blocks.values('pdf_document__name').distinct().count(),
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
        pdfs = pdfs.filter(processed=True, concepts_analyzed=True)
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
    ready_pdfs = PDFDocument.objects.filter(processed=True, concepts_analyzed=True).count()
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
    
    context = {
        'all_focus_blocks': all_focus_blocks,
        'total_blocks': all_focus_blocks.count(),
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
            
            # Get proficiency data
            proficiency_score = request.POST.get('proficiency_score')
            difficulty_rating = request.POST.get('difficulty_rating')
            learning_notes = request.POST.get('learning_notes', '')
            confusion_points = request.POST.getlist('confusion_points')
            
            # Update session
            session.proficiency_score = int(proficiency_score) if proficiency_score else None
            session.difficulty_rating = int(difficulty_rating) if difficulty_rating else None
            session.learning_notes = learning_notes
            session.confusion_points = confusion_points
            session.status = 'completed'
            session.completed_at = timezone.now()
            session.save()
            
            messages.success(request, f"Session completed! Proficiency score: {proficiency_score}/5")
            return redirect('flashcards:all_focus_blocks')
            
        except Exception as e:
            messages.error(request, f"Error saving session: {e}")
            return redirect('flashcards:focus_mode', session_id=session_id)
    
    # GET request - show completion form
    try:
        session = FocusSession.objects.get(id=session_id)
        context = {
            'session': session,
            'focus_block': session.focus_block,
        }
        return render(request, 'flashcards/complete_session.html', context)
    except FocusSession.DoesNotExist:
        messages.error(request, "Session not found")
        return redirect('flashcards:all_focus_blocks')
