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
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
from datetime import timedelta

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

def process_pdf_complete(pdf_document):
    """Check duplicates FIRST with lightweight text extraction"""
    import hashlib
    
    try:
        print(f"🚀 Processing: {pdf_document.name}")
        
        # ✅ STEP 1: LIGHTWEIGHT text extraction (no concept analysis!)
        print("📄 Step 1: Extracting text for duplicate check...")
        
        from .pdf_service import PDFTextExtractor
        extractor = PDFTextExtractor()
        
        # ✅ JUST extract text, no processing
        pdf_path = pdf_document.pdf_file.path
        text, page_count = extractor.extract_text_pdfplumber(pdf_path)
        
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
        
        # ✅ NOW do the heavy processing (only if unique)
        success, error_msg, stats = extractor.extract_and_process_pdf(pdf_document, analyze_concepts=True)
        if not success:
            return False, error_msg, {}
        
        # Generate focus blocks
        from .focus_block_service import generate_focus_blocks
        blocks_success, blocks_message = generate_focus_blocks(pdf_document)
        if not blocks_success:
            return False, f"Focus blocks failed: {blocks_message}", stats
        
        return True, f"✅ Processed! Generated {pdf_document.focus_blocks.count()} focus blocks.", stats
        
    except Exception as e:
        return False, f"Error: {str(e)}", {}


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
    """Unified view to display all focus blocks"""
    
    # Get all focus blocks
    all_blocks = FocusBlock.objects.select_related('pdf_document').order_by(
        'pdf_document__created_at', 'block_order'
    )
    
    if not all_blocks.exists():
        return render(request, 'flashcards/all_focus_blocks.html', {'no_blocks': True})
    
    # Format blocks for display (simple version)
    formatted_blocks = []
    for block in all_blocks:
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
    total_time = sum(block.target_duration for block in all_blocks) / 60
    
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
    
    # ✅ GET COMPLETION DATA FROM DATABASE
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
            
            # Create a new FocusSession record
            session = FocusSession.objects.create(
                focus_block=focus_block,
                proficiency_score=int(proficiency_score),
                total_study_time=float(completion_time),
                status='completed',
                completed_at=timezone.now()
            )
            
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
