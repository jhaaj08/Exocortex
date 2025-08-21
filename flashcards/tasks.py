# flashcards/tasks.py

from celery import shared_task
from celery.exceptions import Retry
import hashlib
import time
import logging
from django.utils import timezone

# Add these imports at the top of flashcards/tasks.py (after existing imports)
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from collections import defaultdict

# Add these imports at the top
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from django.db import transaction

# Add these tasks after your existing ones

@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def generate_focus_block_embeddings_task(self, focus_block_ids=None, batch_size=10):
    """
    Generate embeddings for FocusBlocks to enable semantic deduplication
    
    Args:
        focus_block_ids: List of specific block IDs to process (None = all without embeddings)
        batch_size: Number of blocks to process in one API call
        
    Returns:
        dict: Processing results
    """
    start_time = time.time()
    logger.info(f"🔢 Starting embedding generation for FocusBlocks")
    
    try:
        from .models import FocusBlock
        from openai import OpenAI
        from django.conf import settings
        
        # Get blocks to process
        if focus_block_ids:
            blocks = FocusBlock.objects.filter(id__in=focus_block_ids)
        else:
            # Get blocks without embeddings
            blocks = FocusBlock.objects.filter(content_embedding=[])
        
        if not blocks.exists():
            return {
                'success': True,
                'message': 'No blocks need embedding generation',
                'data': {'blocks_processed': 0, 'processing_time': 0}
            }
        
        # Initialize OpenAI client
        api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not api_key:
            error_msg = "OpenAI API key not configured"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg, 'data': {}}
        
        client = OpenAI(api_key=api_key)
        
        blocks_processed = 0
        
        # Process blocks in batches
        for i in range(0, len(blocks), batch_size):
            batch = list(blocks[i:i + batch_size])
            
            try:
                # Prepare content for embedding
                texts_to_embed = []
                for block in batch:
                    content_text = extract_block_content_for_embedding(block)
                    texts_to_embed.append(content_text)
                
                # Generate embeddings
                response = client.embeddings.create(
                    model="text-embedding-3-small",  # 1536 dimensions, cost-effective
                    input=texts_to_embed
                )
                
                # Save embeddings and content hashes
                for idx, block in enumerate(batch):
                    embedding = response.data[idx].embedding
                    content_hash = generate_content_hash(extract_block_content_for_embedding(block))
                    
                    block.content_embedding = embedding
                    block.content_hash = content_hash
                    block.save()
                    
                    blocks_processed += 1
                    logger.info(f"   ✅ Generated embedding for: {block.title}")
                
            except Exception as batch_error:
                logger.error(f"❌ Failed to process batch {i//batch_size + 1}: {str(batch_error)}")
                continue
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Embedding generation completed: {blocks_processed} blocks in {processing_time:.2f}s")
        
        return {
            'success': True,
            'message': f"Generated embeddings for {blocks_processed} focus blocks in {processing_time:.2f}s",
            'data': {
                'blocks_processed': blocks_processed,
                'processing_time': processing_time
            }
        }
        
    except Exception as e:
        error_msg = f"Embedding generation failed: {str(e)}"
        logger.error(error_msg)
        
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying in {self.default_retry_delay}s")
            raise self.retry(countdown=self.default_retry_delay, exc=e)
        
        return {'success': False, 'message': error_msg, 'data': {}}


@shared_task(bind=True, max_retries=2, default_retry_delay=90)
def deduplicate_focus_blocks_task(self, pdf_document_id=None, similarity_threshold=0.85, dry_run=False):
    """
    Find and merge duplicate FocusBlocks based on semantic similarity
    
    Args:
        pdf_document_id: Process only blocks from this PDF (None = all blocks)
        similarity_threshold: Minimum similarity to consider duplicates (0.85 = 85%)
        dry_run: If True, only identify duplicates without merging
        
    Returns:
        dict: Deduplication results
    """
    start_time = time.time()
    logger.info(f"🔍 Starting FocusBlock deduplication (threshold: {similarity_threshold})")
    
    try:
        from .models import FocusBlock, StudySession
        
        # Get blocks to check
        if pdf_document_id:
            blocks = FocusBlock.objects.filter(
                pdf_document_id=pdf_document_id,
                content_embedding__isnull=False
            ).exclude(content_embedding=[])
        else:
            blocks = FocusBlock.objects.filter(
                content_embedding__isnull=False
            ).exclude(content_embedding=[])
        
        if blocks.count() < 2:
            return {
                'success': True,
                'message': 'Not enough blocks with embeddings for deduplication',
                'data': {'duplicates_found': 0, 'blocks_merged': 0}
            }
        
        logger.info(f"📊 Analyzing {blocks.count()} blocks for duplicates")
        
        # Find duplicate pairs
        duplicate_pairs = find_duplicate_focus_blocks(blocks, similarity_threshold)
        
        if not duplicate_pairs:
            return {
                'success': True,
                'message': f'No duplicates found above {similarity_threshold} threshold',
                'data': {'duplicates_found': 0, 'blocks_merged': 0}
            }
        
        logger.info(f"🎯 Found {len(duplicate_pairs)} duplicate pairs")
        
        if dry_run:
            # Just report what would be merged
            duplicate_info = []
            for (block1, block2, similarity) in duplicate_pairs:
                duplicate_info.append({
                    'block1_id': str(block1.id),
                    'block1_title': block1.title,
                    'block2_id': str(block2.id),
                    'block2_title': block2.title,
                    'similarity': similarity
                })
            
            return {
                'success': True,
                'message': f'DRY RUN: Found {len(duplicate_pairs)} duplicate pairs',
                'data': {
                    'duplicates_found': len(duplicate_pairs),
                    'blocks_merged': 0,
                    'duplicate_pairs': duplicate_info
                }
            }
        
        # Perform actual merging
        blocks_merged = 0
        sessions_affected = 0
        
        for (block1, block2, similarity) in duplicate_pairs:
            try:
                merge_result = merge_duplicate_focus_blocks(block1, block2)
                if merge_result['success']:
                    blocks_merged += 1
                    sessions_affected += merge_result['sessions_affected']
                    logger.info(f"   ✅ Merged: {block2.title} → {block1.title} (similarity: {similarity:.3f})")
            except Exception as merge_error:
                logger.error(f"   ❌ Failed to merge blocks: {str(merge_error)}")
                continue
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Deduplication completed: {blocks_merged} merges, {sessions_affected} sessions affected")
        
        return {
            'success': True,
            'message': f"Merged {blocks_merged} duplicate blocks in {processing_time:.2f}s",
            'data': {
                'duplicates_found': len(duplicate_pairs),
                'blocks_merged': blocks_merged,
                'sessions_affected': sessions_affected,
                'processing_time': processing_time
            }
        }
        
    except Exception as e:
        error_msg = f"Deduplication failed: {str(e)}"
        logger.error(error_msg)
        
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying in {self.default_retry_delay}s")
            raise self.retry(countdown=self.default_retry_delay, exc=e)
        
        return {'success': False, 'message': error_msg, 'data': {}}


def extract_block_content_for_embedding(focus_block):
    """
    Extract meaningful content from a focus block for embedding generation
    """
    content_parts = []
    
    # Add title
    content_parts.append(f"Title: {focus_block.title}")
    
    # Add learning objectives
    if focus_block.learning_objectives:
        objectives_text = " ".join(focus_block.learning_objectives)
        content_parts.append(f"Objectives: {objectives_text}")
    
    # Extract content from segments
    if focus_block.compact7_data and 'segments' in focus_block.compact7_data:
        for segment in focus_block.compact7_data['segments']:
            segment_type = segment.get('type', 'content')
            segment_content = segment.get('content', '')
            
            if segment_type == 'knowledge_check':
                # For Q&A, include both question and answer
                question = segment.get('question', '')
                answer = segment.get('answer', '')
                if question and answer:
                    content_parts.append(f"Q: {question} A: {answer}")
            else:
                # For regular segments, include content
                if segment_content:
                    # Strip HTML tags for cleaner text
                    import re
                    clean_content = re.sub(r'<[^>]+>', '', segment_content)
                    content_parts.append(clean_content)
    
    # Combine all parts
    full_content = " ".join(content_parts)
    
    # Truncate if too long (embedding models have token limits)
    if len(full_content) > 8000:  # ~2000 tokens
        full_content = full_content[:8000] + "..."
    
    return full_content


def generate_content_hash(content_text):
    """Generate SHA-256 hash of content for quick duplicate detection"""
    return hashlib.sha256(content_text.encode('utf-8')).hexdigest()


def find_duplicate_focus_blocks(blocks, similarity_threshold=0.85):
    """
    Find pairs of duplicate focus blocks based on embedding similarity
    
    Returns: List of (block1, block2, similarity) tuples
    """
    blocks_list = list(blocks)
    duplicate_pairs = []
    
    # Quick hash-based filtering first
    hash_groups = {}
    for block in blocks_list:
        content_hash = block.content_hash
        if content_hash:
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append(block)
    
    # Check exact hash matches
    for hash_value, hash_blocks in hash_groups.items():
        if len(hash_blocks) > 1:
            # Exact duplicates - merge all to first one
            for i in range(1, len(hash_blocks)):
                duplicate_pairs.append((hash_blocks[0], hash_blocks[i], 1.0))
    
    # Check semantic similarity for different hashes
    embeddings = []
    blocks_with_embeddings = []
    
    for block in blocks_list:
        if block.content_embedding and len(block.content_embedding) > 0:
            embeddings.append(block.content_embedding)
            blocks_with_embeddings.append(block)
    
    if len(embeddings) < 2:
        return duplicate_pairs
    
    # Calculate similarity matrix
    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Find similar pairs
    for i in range(len(blocks_with_embeddings)):
        for j in range(i + 1, len(blocks_with_embeddings)):
            similarity = similarity_matrix[i][j]
            
            if similarity >= similarity_threshold:
                block1 = blocks_with_embeddings[i]
                block2 = blocks_with_embeddings[j]
                
                # Avoid hash duplicates we already found
                if block1.content_hash != block2.content_hash:
                    duplicate_pairs.append((block1, block2, similarity))
    
    return duplicate_pairs


def merge_duplicate_focus_blocks(keep_block, merge_block):
    """
    Merge two duplicate focus blocks, keeping the older one
    
    Args:
        keep_block: The block to keep (usually older)
        merge_block: The block to merge and delete
        
    Returns:
        dict: Merge results
    """
    from .models import StudySession
    
    try:
        with transaction.atomic():
            # Determine which block to keep (prefer older one)
            if merge_block.created_at < keep_block.created_at:
                keep_block, merge_block = merge_block, keep_block
            
            logger.info(f"🔀 Merging '{merge_block.title}' into '{keep_block.title}'")
            
            # ✅ 1. Combine compact7_data segments
            keep_segments = keep_block.compact7_data.get('segments', [])
            merge_segments = merge_block.compact7_data.get('segments', [])
            
            # Remove duplicate segments (basic deduplication by content)
            seen_content = set()
            combined_segments = []
            
            for segment in keep_segments + merge_segments:
                segment_key = f"{segment.get('type', '')}-{segment.get('content', '')[:100]}"
                if segment_key not in seen_content:
                    combined_segments.append(segment)
                    seen_content.add(segment_key)
            
            # ✅ 2. Update compact7_data
            combined_data = keep_block.compact7_data.copy()
            combined_data['segments'] = combined_segments
            
            # Update total duration (sum of unique segments)
            total_duration = sum(s.get('duration_seconds', 90) for s in combined_segments)
            combined_data['total_duration'] = total_duration
            
            # Combine learning objectives
            keep_objectives = set(keep_block.learning_objectives or [])
            merge_objectives = set(merge_block.learning_objectives or [])
            combined_objectives = list(keep_objectives | merge_objectives)
            
            # ✅ 3. Update the keep_block
            keep_block.compact7_data = combined_data
            keep_block.target_duration = total_duration
            keep_block.learning_objectives = combined_objectives
            keep_block.is_merged = True
            
            # Track merged blocks
            merged_from = keep_block.merged_from_blocks or []
            merged_from.append(str(merge_block.id))
            keep_block.merged_from_blocks = merged_from
            
            keep_block.save()
            
            # ✅ 4. Update StudySessions pointing to merge_block
            sessions_affected = 0
            
            # Update current_focus_block references
            current_sessions = StudySession.objects.filter(current_focus_block=merge_block)
            for session in current_sessions:
                session.current_focus_block = keep_block
                session.current_segment = 0  # Reset to beginning
                session.save()
                sessions_affected += 1
            
            # Update completed_focus_blocks references
            completed_sessions = StudySession.objects.filter(completed_focus_blocks=merge_block)
            for session in completed_sessions:
                session.completed_focus_blocks.remove(merge_block)
                session.completed_focus_blocks.add(keep_block)
                sessions_affected += 1
            
            # Update segment_progress (remove old block progress)
            progress_sessions = StudySession.objects.filter(
                segment_progress__has_key=str(merge_block.id)
            )
            for session in progress_sessions:
                if str(merge_block.id) in session.segment_progress:
                    # Transfer progress to keep_block if it doesn't exist
                    if str(keep_block.id) not in session.segment_progress:
                        session.segment_progress[str(keep_block.id)] = session.segment_progress[str(merge_block.id)]
                    del session.segment_progress[str(merge_block.id)]
                    session.save()
                    sessions_affected += 1
            
            # ✅ 5. Delete the duplicate block
            merge_block.delete()
            
            logger.info(f"   ✅ Merged successfully: {len(combined_segments)} total segments, {sessions_affected} sessions updated")
            
            return {
                'success': True,
                'keep_block_id': str(keep_block.id),
                'merged_block_id': str(merge_block.id),
                'total_segments': len(combined_segments),
                'sessions_affected': sessions_affected
            }
            
    except Exception as e:
        logger.error(f"❌ Merge failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'sessions_affected': 0
        }


# ✅ UPDATED COMPLETE PIPELINE: Extract → Chunk → Generate Focus Blocks
@shared_task
def complete_pdf_processing_pipeline(pdf_document_id, similarity_threshold=0.60, comprehensive_coverage=True):
    """
    Complete pipeline: Extract text → Advanced chunking → Generate focus blocks
    
    This runs all three tasks sequentially to fully process a PDF.
    """
    import time
    start_time = time.time()
    
    logger.info(f"🚀 Starting complete PDF processing pipeline for ID: {pdf_document_id}")
    
    try:
        # Step 1: Extract text
        logger.info("📄 Step 1: Extracting text...")
        extraction_result = extract_pdf_text_task(pdf_document_id)
        
        if not extraction_result['success']:
            error_msg = f"Text extraction failed: {extraction_result['message']}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'pipeline_step': 'extraction',
                'results': {'extraction': extraction_result}
            }
        
        logger.info(f"✅ Text extraction completed: {extraction_result['message']}")
        
        # Step 2: Advanced chunking
        logger.info("🧠 Step 2: Advanced chunking with communities...")
        chunking_result = advanced_chunk_processing_task(pdf_document_id, similarity_threshold)
        
        if not chunking_result['success']:
            error_msg = f"Chunking failed: {chunking_result['message']}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'pipeline_step': 'chunking',
                'results': {
                    'extraction': extraction_result,
                    'chunking': chunking_result
                }
            }
        
        logger.info(f"✅ Advanced chunking completed: {chunking_result['message']}")
        
        # Step 3: Generate focus blocks
        logger.info("🎯 Step 3: Generating comprehensive focus blocks...")
        focus_blocks_result = generate_focus_blocks_from_communities_task(pdf_document_id, comprehensive_coverage)
        
        if not focus_blocks_result['success']:
            error_msg = f"Focus block generation failed: {focus_blocks_result['message']}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'pipeline_step': 'focus_blocks',
                'results': {
                    'extraction': extraction_result,
                    'chunking': chunking_result,
                    'focus_blocks': focus_blocks_result
                }
            }
        
        # Mark PDF as processed
        from .models import PDFDocument
        pdf_document = PDFDocument.objects.get(id=pdf_document_id)
        pdf_document.processed = True
        pdf_document.save()
        
        total_time = time.time() - start_time
        
        logger.info(f"🎉 Complete pipeline finished successfully in {total_time:.2f}s")
        logger.info(f"   📊 {extraction_result['data']['word_count']:,} words processed")
        logger.info(f"   🏘️ {chunking_result['data']['communities_count']} communities detected")
        logger.info(f"   📚 {focus_blocks_result['data']['focus_blocks_created']} focus blocks created")
        
        return {
            'success': True,
            'message': f"Complete pipeline finished in {total_time:.2f}s - {focus_blocks_result['data']['focus_blocks_created']} focus blocks created",
            'pipeline_step': 'complete',
            'total_time': total_time,
            'results': {
                'extraction': extraction_result,
                'chunking': chunking_result,
                'focus_blocks': focus_blocks_result
            }
        }
        
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'message': error_msg,
            'pipeline_step': 'pipeline_error',
            'results': {}
        }


# ✅ ENHANCED COMPLETE PIPELINE: Extract → Chunk → Generate → Embed → Deduplicate
@shared_task
def complete_pdf_processing_with_deduplication(pdf_document_id, similarity_threshold=0.60, dedup_threshold=0.85):
    """
    Complete pipeline with deduplication: Extract → Chunk → Generate → Embed → Deduplicate
    """
    import time
    start_time = time.time()
    
    logger.info(f"🚀 Starting complete PDF processing with deduplication for ID: {pdf_document_id}")
    
    try:
        # Step 1-3: Run the existing pipeline
        logger.info("📄 Steps 1-3: Running extraction, chunking, and focus block generation...")
        pipeline_result = complete_pdf_processing_pipeline(pdf_document_id, similarity_threshold)
        
        if not pipeline_result['success']:
            return pipeline_result
        
        # Step 4: Generate embeddings for new blocks
        logger.info("🔢 Step 4: Generating embeddings for new focus blocks...")
        from .models import FocusBlock
        new_blocks = FocusBlock.objects.filter(
            pdf_document_id=pdf_document_id,
            content_embedding=[]
        )
        
        if new_blocks.exists():
            embedding_result = generate_focus_block_embeddings_task([str(b.id) for b in new_blocks])
            if not embedding_result['success']:
                logger.warning(f"⚠️ Embedding generation failed: {embedding_result['message']}")
        
        # Step 5: Run deduplication
        logger.info("🔍 Step 5: Running deduplication...")
        dedup_result = deduplicate_focus_blocks_task(
            pdf_document_id=pdf_document_id,
            similarity_threshold=dedup_threshold,
            dry_run=False
        )
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'message': f"Complete pipeline with deduplication finished in {total_time:.2f}s",
            'total_time': total_time,
            'results': {
                'pipeline': pipeline_result,
                'embeddings': embedding_result if 'embedding_result' in locals() else None,
                'deduplication': dedup_result
            }
        }
        
    except Exception as e:
        error_msg = f"Enhanced pipeline failed: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'message': error_msg, 'results': {}}


# Set up logging
logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def extract_pdf_text_task(self, pdf_document_id):
    """
    Self-contained Celery task for PDF text extraction
    
    Args:
        pdf_document_id: Primary key of PDFDocument to process
        
    Returns:
        dict: {
            'success': bool,
            'message': str,
            'data': {
                'text_length': int,
                'content_hash': str,
                'page_count': int,
                'word_count': int,
                'extraction_method': str,
                'processing_duration': float
            } or None,
            'duplicate_info': {
                'is_duplicate': bool,
                'duplicate_of_id': int or None,
                'existing_name': str or None
            } or None
        }
    """
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports in Celery
        from .models import PDFDocument
        from .pdf_service import PDFTextExtractor
        
        logger.info(f"📄 Starting PDF text extraction task for ID: {pdf_document_id}")
        
        # ✅ STEP 1: Get PDF document
        try:
            pdf_document = PDFDocument.objects.get(id=pdf_document_id)
            logger.info(f"📂 Found PDF: {pdf_document.name}")
        except PDFDocument.DoesNotExist:
            error_msg = f"PDF document with ID {pdf_document_id} not found"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'data': None,
                'duplicate_info': None
            }
        
        # ✅ STEP 2: Check if already processed
        if pdf_document.extracted_text and pdf_document.content_hash:
            logger.info(f"📋 PDF already processed, skipping extraction")
            return {
                'success': True,
                'message': f"PDF already processed: {len(pdf_document.extracted_text)} characters",
                'data': {
                    'text_length': len(pdf_document.extracted_text),
                    'content_hash': pdf_document.content_hash,
                    'page_count': pdf_document.page_count,
                    'word_count': pdf_document.word_count,
                    'extraction_method': 'cached',
                    'processing_duration': 0.0
                },
                'duplicate_info': {
                    'is_duplicate': pdf_document.is_duplicate,
                    'duplicate_of_id': pdf_document.duplicate_of_id if pdf_document.duplicate_of else None,
                    'existing_name': pdf_document.duplicate_of.name if pdf_document.duplicate_of else None
                }
            }
        
        # ✅ STEP 3: Extract text using file content method (Railway compatible)
        try:
            extractor = PDFTextExtractor()
            
            # Use the new file content method instead of file path
            text, page_count, extraction_method = extractor.extract_text_from_file_content(pdf_document)
            
            # Validate extraction
            if not text or len(text.strip()) < 50:
                error_msg = f"Could not extract meaningful text from PDF: {pdf_document.name}"
                logger.error(error_msg)
                
                # Retry on extraction failure
                if self.request.retries < self.max_retries:
                    retry_delay = 60 * (2 ** self.request.retries)  # Exponential backoff
                    logger.warning(f"🔄 Retrying extraction in {retry_delay}s (attempt {self.request.retries + 1}/{self.max_retries})")
                    raise self.retry(countdown=retry_delay)
                
                return {
                    'success': False,
                    'message': error_msg,
                    'data': None,
                    'duplicate_info': None
                }
            
            logger.info(f"✅ Text extracted successfully using {extraction_method}: {len(text)} characters, {page_count} pages")
            
        except Exception as extraction_error:
            error_msg = f"Text extraction failed: {str(extraction_error)}"
            logger.error(error_msg)
            
            # Retry on extraction errors
            if self.request.retries < self.max_retries:
                retry_delay = 60 * (2 ** self.request.retries)
                logger.warning(f"🔄 Retrying due to extraction error in {retry_delay}s")
                raise self.retry(countdown=retry_delay, exc=extraction_error)
            
            return {
                'success': False,
                'message': error_msg,
                'data': None,
                'duplicate_info': None
            }
        
        # ✅ STEP 4: Calculate content metrics
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        word_count = len(text.split())
        file_size = pdf_document.pdf_file.size if pdf_document.pdf_file else 0
        
        logger.info(f"📊 Content metrics - Hash: {content_hash[:12]}..., Words: {word_count:,}")
        
        # ✅ STEP 5: Check for duplicates
        duplicate_info = {
            'is_duplicate': False,
            'duplicate_of_id': None,
            'existing_name': None
        }
        
        try:
            existing_pdf = PDFDocument.objects.filter(
                content_hash=content_hash,
                processed=True,
                focus_blocks__isnull=False
            ).exclude(id=pdf_document.id).first()
            
            if existing_pdf:
                duplicate_info = {
                    'is_duplicate': True,
                    'duplicate_of_id': existing_pdf.id,
                    'existing_name': existing_pdf.name
                }
                logger.info(f"🛑 Duplicate detected: {existing_pdf.name} (ID: {existing_pdf.id})")
            else:
                logger.info(f"✅ Unique content confirmed")
                
        except Exception as db_error:
            logger.warning(f"⚠️ Duplicate check failed (continuing): {db_error}")
        
        # ✅ STEP 6: Update PDFDocument model
        try:
            pdf_document.extracted_text = text
            pdf_document.content_hash = content_hash
            pdf_document.page_count = page_count
            pdf_document.word_count = word_count
            pdf_document.file_size = file_size
            
            # Update duplicate info if found
            if duplicate_info['is_duplicate']:
                pdf_document.is_duplicate = True
                pdf_document.duplicate_of_id = duplicate_info['duplicate_of_id']
            
            # Don't mark as processed yet - that's for the main pipeline
            pdf_document.save()
            
            logger.info(f"💾 PDFDocument updated successfully")
            
        except Exception as save_error:
            error_msg = f"Failed to save PDFDocument: {str(save_error)}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'data': None,
                'duplicate_info': None
            }
        
        # ✅ STEP 7: Return success result
        processing_duration = round(time.time() - start_time, 2)
        
        result_data = {
            'text_length': len(text),
            'content_hash': content_hash,
            'page_count': page_count,
            'word_count': word_count,
            'extraction_method': extraction_method,
            'processing_duration': processing_duration
        }
        
        success_msg = f"Text extraction completed: {word_count:,} words from {page_count} pages in {processing_duration}s"
        if duplicate_info['is_duplicate']:
            success_msg += f" (duplicate of '{duplicate_info['existing_name']}')"
        
        logger.info(f"🎉 Task completed successfully: {success_msg}")
        
        return {
            'success': True,
            'message': success_msg,
            'data': result_data,
            'duplicate_info': duplicate_info
        }
        
    except Retry:
        # Re-raise Retry exceptions to let Celery handle them
        raise
        
    except Exception as e:
        error_duration = round(time.time() - start_time, 2)
        error_msg = f"Unexpected error in PDF extraction task after {error_duration}s: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            'success': False,
            'message': error_msg,
            'data': None,
            'duplicate_info': None
        }


@shared_task
def cleanup_failed_extractions():
    """
    Periodic task to clean up PDFDocuments that failed extraction
    Run this daily via Celery Beat
    """
    from .models import PDFDocument
    from django.utils import timezone
    from datetime import timedelta
    
    cutoff_time = timezone.now() - timedelta(days=1)
    
    failed_docs = PDFDocument.objects.filter(
        created_at__lt=cutoff_time,
        extracted_text='',
        processed=False
    )
    
    count = failed_docs.count()
    if count > 0:
        logger.info(f"🧹 Cleaning up {count} failed extraction documents")
        failed_docs.delete()
    
    return f"Cleaned up {count} failed extraction documents"


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def advanced_chunk_processing_task(self, pdf_document_id, similarity_threshold=0.65, min_chunk_size=100, max_chunk_size=800):
    """
    Advanced chunk processing using embeddings and community detection
    
    This task:
    1. Creates semantic chunks from extracted text
    2. Generates embeddings using sentence-transformers (free)
    3. Builds similarity graph based on cosine similarity
    4. Runs community detection (Louvain algorithm)
    5. Aggregates communities into coherent chunks
    6. Saves processed chunks to PDFDocument
    
    Args:
        pdf_document_id: Primary key of PDFDocument to process
        similarity_threshold: Minimum similarity for graph edges (default: 0.65)
        min_chunk_size: Minimum words per initial chunk (default: 100)
        max_chunk_size: Maximum words per initial chunk (default: 800)
        
    Returns:
        dict: {
            'success': bool,
            'message': str, 
            'data': {
                'communities_count': int,
                'total_chunks': int,
                'processing_time': float,
                'avg_community_size': float
            }
        }
    """
    start_time = time.time()
    logger.info(f"🧠 Starting advanced chunk processing for PDF ID: {pdf_document_id}")
    
    try:
        # ✅ STEP 1: Get PDF document and validate
        from .models import PDFDocument
        
        try:
            pdf_document = PDFDocument.objects.get(id=pdf_document_id)
            if not pdf_document.extracted_text:
                error_msg = f"No extracted text found for PDF ID: {pdf_document_id}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'data': {}
                }
                
            logger.info(f"📄 Processing: {pdf_document.name}")
            logger.info(f"📊 Text length: {len(pdf_document.extracted_text):,} characters")
            
        except PDFDocument.DoesNotExist:
            error_msg = f"PDFDocument with ID {pdf_document_id} does not exist"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'data': {}
            }
        
        # ✅ STEP 2: Create initial semantic chunks
        logger.info("🔪 Creating initial semantic chunks...")
        initial_chunks = create_semantic_chunks(
            pdf_document.extracted_text, 
            min_size=min_chunk_size, 
            max_size=max_chunk_size
        )
        logger.info(f"📝 Created {len(initial_chunks)} initial chunks")
        
        if len(initial_chunks) < 2:
            # If too few chunks, return as-is
            result_data = {
                'communities_count': 1,
                'total_chunks': len(initial_chunks),
                'processing_time': time.time() - start_time,
                'avg_community_size': len(initial_chunks)
            }
            
            # Save simple chunks
            pdf_document.advanced_chunks = json.dumps({
                'communities': [{'chunks': initial_chunks, 'community_id': 0}],
                'metadata': result_data
            })
            pdf_document.save()
            
            return {
                'success': True,
                'message': f"Processed {len(initial_chunks)} chunks (too few for community detection)",
                'data': result_data
            }
        
        # ✅ STEP 3: Generate embeddings using TF-IDF (fallback to sentence-transformers if available)
        logger.info("🔢 Generating embeddings...")
        try:
            embeddings = generate_embeddings_tfidf(initial_chunks)
            embedding_method = "TF-IDF"
        except Exception as embed_error:
            logger.warning(f"TF-IDF embedding failed: {embed_error}, using simple similarity")
            embeddings = generate_simple_embeddings(initial_chunks)
            embedding_method = "Simple"
            
        logger.info(f"✅ Generated {len(embeddings)} embeddings using {embedding_method}")
        
        # ✅ STEP 4: Build similarity graph
        logger.info("🕸️ Building similarity graph...")
        similarity_graph = build_similarity_graph(
            embeddings, 
            threshold=similarity_threshold,
            chunk_texts=initial_chunks
        )
        
        edges_count = similarity_graph.number_of_edges()
        nodes_count = similarity_graph.number_of_nodes()
        logger.info(f"📊 Graph: {nodes_count} nodes, {edges_count} edges")
        
        # ✅ STEP 5: Run community detection
        logger.info("🏘️ Running community detection...")
        communities = detect_communities_louvain(similarity_graph)
        logger.info(f"🎯 Detected {len(communities)} communities")
        
        # ✅ STEP 6: Aggregate communities into coherent chunks
        logger.info("📚 Aggregating communities into chunks...")
        aggregated_chunks = aggregate_communities(communities, initial_chunks)
        
        # ✅ STEP 7: Save results to database
        processing_time = time.time() - start_time
        
        result_data = {
            'communities_count': len(communities),
            'total_chunks': len(aggregated_chunks),
            'processing_time': processing_time,
            'avg_community_size': np.mean([len(community) for community in communities]) if communities else 0,
            'embedding_method': embedding_method,
            'similarity_threshold': similarity_threshold,
            'edges_count': edges_count
        }
        
        # Save to PDFDocument
        pdf_document.advanced_chunks = json.dumps({
            'communities': aggregated_chunks,
            'metadata': result_data
        })
        pdf_document.save()
        
        logger.info(f"💾 Saved {len(aggregated_chunks)} community-based chunks")
        logger.info(f"⏱️ Total processing time: {processing_time:.2f}s")
        
        return {
            'success': True,
            'message': f"Successfully processed {len(aggregated_chunks)} community-based chunks in {processing_time:.2f}s",
            'data': result_data
        }
        
    except Exception as e:
        error_msg = f"Advanced chunk processing failed: {str(e)}"
        logger.error(error_msg)
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying in {self.default_retry_delay}s (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(countdown=self.default_retry_delay, exc=e)
        
        return {
            'success': False,
            'message': error_msg,
            'data': {}
        }


def create_semantic_chunks(text, min_size=100, max_size=800):
    """Create semantic chunks based on paragraphs and sentences"""
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    for paragraph in paragraphs:
        para_words = len(paragraph.split())
        
        # If adding this paragraph exceeds max_size, finalize current chunk
        if current_word_count + para_words > max_size and current_word_count >= min_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_word_count = para_words
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_word_count += para_words
    
    # Add final chunk
    if current_chunk and current_word_count >= min_size:
        chunks.append(current_chunk.strip())
    
    return chunks


def generate_embeddings_tfidf(chunks, max_features=1000):
    """Generate TF-IDF embeddings for chunks"""
    
    # Clean chunks
    cleaned_chunks = []
    for chunk in chunks:
        # Basic cleaning
        cleaned = re.sub(r'[^\w\s]', ' ', chunk.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned_chunks.append(cleaned)
    
    # Generate TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    
    tfidf_matrix = vectorizer.fit_transform(cleaned_chunks)
    return tfidf_matrix.toarray()


def generate_simple_embeddings(chunks):
    """Fallback: Generate simple word-frequency based embeddings"""
    
    # Get all unique words
    all_words = set()
    cleaned_chunks = []
    
    for chunk in chunks:
        words = re.findall(r'\w+', chunk.lower())
        cleaned_chunks.append(words)
        all_words.update(words)
    
    word_to_idx = {word: idx for idx, word in enumerate(list(all_words))}
    
    # Create simple frequency vectors
    embeddings = []
    for words in cleaned_chunks:
        vector = np.zeros(len(word_to_idx))
        word_counts = defaultdict(int)
        
        for word in words:
            word_counts[word] += 1
        
        for word, count in word_counts.items():
            if word in word_to_idx:
                vector[word_to_idx[word]] = count / len(words)  # Normalize
        
        embeddings.append(vector)
    
    return np.array(embeddings)


def build_similarity_graph(embeddings, threshold=0.65, chunk_texts=None):
    """Build similarity graph using cosine similarity"""
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(embeddings)):
        G.add_node(i, text=chunk_texts[i] if chunk_texts else f"Chunk {i}")
    
    # Add edges based on similarity threshold
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] >= threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])
    
    return G


def detect_communities_louvain(graph):
    """Detect communities using Louvain algorithm"""
    
    try:
        # Try networkx community detection
        import networkx.algorithms.community as nx_community
        communities = list(nx_community.greedy_modularity_communities(graph))
        return [list(community) for community in communities]
        
    except ImportError:
        # Fallback: Simple connected components
        logger.warning("NetworkX community detection not available, using connected components")
        return [list(component) for component in nx.connected_components(graph)]


def aggregate_communities(communities, original_chunks):
    """Aggregate chunks within each community"""
    
    aggregated = []
    
    for community_id, community_indices in enumerate(communities):
        # Combine chunks in this community
        community_chunks = [original_chunks[i] for i in community_indices]
        
        # Create aggregated chunk
        aggregated_chunk = {
            'community_id': community_id,
            'chunk_count': len(community_chunks),
            'chunks': community_chunks,
            'combined_text': '\n\n'.join(community_chunks),
            'word_count': sum(len(chunk.split()) for chunk in community_chunks),
            'indices': community_indices
        }
        
        aggregated.append(aggregated_chunk)
    
    # Sort by community size (largest first)
    aggregated.sort(key=lambda x: x['word_count'], reverse=True)
    
    return aggregated


# ✅ FIXED CHAIN TASK: Combine extraction + advanced processing
@shared_task
def process_pdf_with_advanced_chunking(pdf_document_id, similarity_threshold=0.65):
    """
    Chained task: Extract text → Advanced chunking
    
    This creates a proper Celery chain that runs extraction first,
    then advanced chunking only if extraction succeeds.
    """
    from celery import chain
    
    # Create the chain - second task ignores first task's result
    workflow = chain(
        extract_pdf_text_task.s(pdf_document_id),
        advanced_chunk_processing_task.si(pdf_document_id, similarity_threshold)  # .si() = immutable signature
    )
    
    # Execute the chain NON-BLOCKING
    result = workflow.apply_async()
    return f"Chain started for PDF {pdf_document_id} - Task ID: {result.id}"


# ✅ BETTER APPROACH: Callback-style chaining
@shared_task
def process_pdf_sequential(pdf_document_id, similarity_threshold=0.65):
    """
    Sequential processing: Extract → Check success → Advanced chunking
    """
    import time
    start_time = time.time()
    
    logger.info(f"🔗 Starting sequential PDF processing for ID: {pdf_document_id}")
    
    # Step 1: Extract text
    logger.info("📄 Step 1: Extracting text...")
    extraction_result = extract_pdf_text_task(pdf_document_id)
    
    if not extraction_result['success']:
        error_msg = f"Text extraction failed: {extraction_result['message']}"
        logger.error(error_msg)
        return {
            'success': False,
            'message': error_msg,
            'extraction_result': extraction_result,
            'chunking_result': None
        }
    
    logger.info(f"✅ Text extraction completed: {extraction_result['message']}")
    
    # Step 2: Advanced chunking (only if extraction succeeded)
    logger.info("🧠 Step 2: Advanced chunking...")
    chunking_result = advanced_chunk_processing_task(pdf_document_id, similarity_threshold)
    
    total_time = time.time() - start_time
    
    if chunking_result['success']:
        logger.info(f"✅ Sequential processing completed in {total_time:.2f}s")
        return {
            'success': True,
            'message': f"Full pipeline completed in {total_time:.2f}s",
            'extraction_result': extraction_result,
            'chunking_result': chunking_result,
            'total_time': total_time
        }
    else:
        logger.error(f"❌ Chunking failed: {chunking_result['message']}")
        return {
            'success': False,
            'message': f"Chunking failed: {chunking_result['message']}",
            'extraction_result': extraction_result,
            'chunking_result': chunking_result,
            'total_time': total_time
        }

# Add these imports at the top of flashcards/tasks.py (after existing imports)
import json
from django.db import models

# Add this new task after the existing tasks in flashcards/tasks.py

@shared_task(bind=True, max_retries=2, default_retry_delay=180)
def generate_focus_blocks_from_communities_task(self, pdf_document_id, comprehensive_coverage=True, max_tokens_per_community=3000):
    """
    Generate focus blocks from communities using LLM with comprehensive content coverage
    
    This task:
    1. Retrieves communities from PDFDocument.advanced_chunks
    2. Uses enhanced AI prompts to ensure NO content is missed
    3. Creates comprehensive focus blocks with multiple segments per community
    4. Handles large communities by chunking them appropriately
    5. Saves focus blocks to database
    
    Args:
        pdf_document_id: Primary key of PDFDocument
        comprehensive_coverage: If True, ensures all content is covered (default: True)
        max_tokens_per_community: Max tokens per community to avoid AI limits (default: 3000)
        
    Returns:
        dict: {
            'success': bool,
            'message': str,
            'data': {
                'focus_blocks_created': int,
                'communities_processed': int,
                'total_segments': int,
                'total_qa_items': int,
                'processing_time': float
            }
        }
    """
    start_time = time.time()
    logger.info(f"🎯 Starting focus block generation from communities for PDF ID: {pdf_document_id}")
    
    try:
        # ✅ STEP 1: Get PDF document and validate communities
        from .models import PDFDocument, ConceptUnit, FocusBlock
        from openai import OpenAI
        from django.conf import settings
        
        try:
            pdf_document = PDFDocument.objects.get(id=pdf_document_id)
            
            if not pdf_document.advanced_chunks:
                error_msg = f"No advanced chunks found for PDF ID: {pdf_document_id}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'data': {}
                }
                
            communities_data = json.loads(pdf_document.advanced_chunks) if isinstance(pdf_document.advanced_chunks, str) else pdf_document.advanced_chunks
            communities = communities_data.get('communities', [])
            
            if not communities:
                error_msg = f"No communities found in advanced_chunks for PDF ID: {pdf_document_id}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'data': {}
                }
                
            logger.info(f"📄 Processing: {pdf_document.name}")
            logger.info(f"🏘️ Found {len(communities)} communities to process")
            
        except PDFDocument.DoesNotExist:
            error_msg = f"PDFDocument with ID {pdf_document_id} does not exist"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'data': {}
            }
        
        # ✅ STEP 2: Initialize OpenAI client
        api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not api_key:
            error_msg = "OpenAI API key not configured"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'data': {}
            }
        
        client = OpenAI(api_key=api_key)
        
        # ✅ STEP 3: Process each community into focus blocks
        focus_blocks_created = []
        total_segments = 0
        total_qa_items = 0
        
        for community_idx, community in enumerate(communities):
            logger.info(f"🏘️ Processing community {community_idx + 1}/{len(communities)}")
            
            try:
                # Handle large communities by splitting if necessary
                focus_blocks_for_community = process_community_comprehensive(
                    client=client,
                    community=community,
                    pdf_document=pdf_document,
                    community_idx=community_idx,
                    comprehensive_coverage=comprehensive_coverage,
                    max_tokens=max_tokens_per_community
                )
                
                focus_blocks_created.extend(focus_blocks_for_community)
                
                # Count segments and Q&A
                for block in focus_blocks_for_community:
                    block_segments = block.compact7_data.get('segments', [])
                    total_segments += len([s for s in block_segments if s.get('type') != 'knowledge_check'])
                    total_qa_items += len([s for s in block_segments if s.get('type') == 'knowledge_check'])
                    
            except Exception as community_error:
                logger.error(f"❌ Failed to process community {community_idx}: {str(community_error)}")
                continue
        
        # ✅ STEP 4: Final statistics and return
        processing_time = time.time() - start_time
        
        result_data = {
            'focus_blocks_created': len(focus_blocks_created),
            'communities_processed': len(communities),
            'total_segments': total_segments,
            'total_qa_items': total_qa_items,
            'processing_time': processing_time
        }
        
        logger.info(f"✅ Focus block generation completed:")
        logger.info(f"   📚 Created {len(focus_blocks_created)} focus blocks")
        logger.info(f"   📝 Total segments: {total_segments}")
        logger.info(f"   ❓ Total Q&A items: {total_qa_items}")
        logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
        
        return {
            'success': True,
            'message': f"Successfully created {len(focus_blocks_created)} focus blocks from {len(communities)} communities in {processing_time:.2f}s",
            'data': result_data
        }
        
    except Exception as e:
        error_msg = f"Focus block generation failed: {str(e)}"
        logger.error(error_msg)
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying in {self.default_retry_delay}s (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(countdown=self.default_retry_delay, exc=e)
        
        return {
            'success': False,
            'message': error_msg,
            'data': {}
        }


def process_community_comprehensive(client, community, pdf_document, community_idx, comprehensive_coverage=True, max_tokens=3000):
    """
    Process a single community into one or more focus blocks with comprehensive coverage
    """
    from .models import ConceptUnit, FocusBlock
    import json
    
    community_text = community.get('combined_text', '')
    community_chunks = community.get('chunks', [])
    word_count = community.get('word_count', 0)
    
    logger.info(f"   📊 Community {community_idx}: {word_count} words, {len(community_chunks)} chunks")
    
    # ✅ Split large communities to ensure comprehensive coverage
    if word_count > 1500 or len(community_chunks) > 6:  # Large community
        logger.info(f"   📚 Large community detected - splitting for comprehensive coverage")
        return process_large_community_split(client, community, pdf_document, community_idx, max_tokens)
    else:
        logger.info(f"   📝 Processing as single focus block")
        return process_single_community(client, community, pdf_document, community_idx, max_tokens)


def process_large_community_split(client, community, pdf_document, community_idx, max_tokens):
    """
    Split large communities into multiple focused blocks to ensure nothing is missed
    """
    from .models import ConceptUnit, FocusBlock
    import json
    
    community_chunks = community.get('chunks', [])
    
    # Group chunks into sub-communities (2-3 chunks per block for detailed coverage)
    sub_communities = []
    current_sub = []
    current_word_count = 0
    
    for chunk in community_chunks:
        chunk_words = len(chunk.split()) if isinstance(chunk, str) else len(str(chunk).split())
        
        if current_word_count + chunk_words > 800 and current_sub:  # Max ~800 words per sub-community
            sub_communities.append({
                'chunks': current_sub,
                'combined_text': '\n\n'.join([str(c) for c in current_sub]),
                'word_count': current_word_count
            })
            current_sub = [chunk]
            current_word_count = chunk_words
        else:
            current_sub.append(chunk)
            current_word_count += chunk_words
    
    # Add final sub-community
    if current_sub:
        sub_communities.append({
            'chunks': current_sub,
            'combined_text': '\n\n'.join([str(c) for c in current_sub]),
            'word_count': current_word_count
        })
    
    logger.info(f"   🔀 Split into {len(sub_communities)} focused sub-blocks")
    
    # Process each sub-community
    focus_blocks = []
    for sub_idx, sub_community in enumerate(sub_communities):
        try:
            block = create_comprehensive_focus_block(
                client=client,
                community_data=sub_community,
                pdf_document=pdf_document,
                block_order=f"{community_idx}_{sub_idx}",
                title_suffix=f"Part {sub_idx + 1}" if len(sub_communities) > 1 else "",
                max_tokens=max_tokens
            )
            focus_blocks.append(block)
        except Exception as sub_error:
            logger.error(f"   ❌ Failed to create sub-block {sub_idx}: {str(sub_error)}")
            continue
    
    return focus_blocks


def process_single_community(client, community, pdf_document, community_idx, max_tokens):
    """
    Process a single community into one comprehensive focus block
    """
    try:
        block = create_comprehensive_focus_block(
            client=client,
            community_data=community,
            pdf_document=pdf_document,
            block_order=community_idx,
            title_suffix="",
            max_tokens=max_tokens
        )
        return [block]
    except Exception as error:
        logger.error(f"   ❌ Failed to create focus block: {str(error)}")
        return []


def create_comprehensive_focus_block(client, community_data, pdf_document, block_order, title_suffix="", max_tokens=3000):
    """
    Create a comprehensive focus block with enhanced AI prompt for full coverage
    """
    from .models import ConceptUnit, FocusBlock
    import json
    
    community_text = community_data.get('combined_text', '')
    chunks = community_data.get('chunks', [])
    
    # ✅ ENHANCED PROMPT for comprehensive coverage
    prompt = f"""
    Based on the following content from a PDF document, create a comprehensive focus block that covers ALL important information without missing anything:

    CONTENT TO COVER:
    {community_text}

    REQUIREMENT: COMPREHENSIVE COVERAGE
    - Include ALL key concepts, definitions, examples, and details
    - Create multiple segments to cover different aspects thoroughly
    - Ensure nothing important is overlooked or summarized too briefly
    - If the content covers multiple sub-topics, create separate segments for each

    Create a focus block with this ENHANCED structure:
    {{
        "title": "Concise title (2-7 words, no academic prefixes)",
        "learning_objectives": ["specific objective 1", "specific objective 2", "specific objective 3", "specific objective 4"],
        "segments": [
            {{
                "type": "context",
                "title": "Background & Context",
                "content": "Essential background information and context",
                "duration_seconds": 90
            }},
            {{
                "type": "core_concept_1",
                "title": "Primary Concept",
                "content": "Detailed explanation of the main concept",
                "duration_seconds": 180
            }},
            {{
                "type": "core_concept_2", 
                "title": "Secondary Concept",
                "content": "Detailed explanation of secondary concepts (if applicable)",
                "duration_seconds": 180
            }},
            {{
                "type": "examples",
                "title": "Detailed Examples",
                "content": "Comprehensive examples with explanations",
                "duration_seconds": 150
            }},
            {{
                "type": "applications",
                "title": "Applications & Use Cases",
                "content": "Real-world applications and use cases",
                "duration_seconds": 120
            }},
            {{
                "type": "technical_details",
                "title": "Technical Details",
                "content": "Important technical details and specifications",
                "duration_seconds": 150
            }},
            {{
                "type": "summary",
                "title": "Key Takeaways",
                "content": "Comprehensive summary of all key points",
                "duration_seconds": 90
            }}
        ],
        "qa_items": [
            {{
                "question": "What are the fundamental concepts covered in this section?",
                "answer": "Comprehensive answer covering all fundamental concepts",
                "difficulty": "basic"
            }},
            {{
                "question": "How do these concepts apply in real-world scenarios?",
                "answer": "Detailed answer with practical applications and examples",
                "difficulty": "intermediate"
            }},
            {{
                "question": "What are the advanced implications and connections to other topics?",
                "answer": "Advanced answer showing deeper understanding and connections",
                "difficulty": "advanced"
            }},
            {{
                "question": "What technical details or edge cases are important to understand?",
                "answer": "Detailed technical answer covering important specifics",
                "difficulty": "advanced"
            }}
        ],
        "total_duration": 960,
        "difficulty_level": "intermediate"
    }}

    CRITICAL GUIDELINES:
    - COMPREHENSIVE: Cover ALL content provided - don't summarize or skip details
    - ADAPTIVE: Adjust segment types based on actual content (remove unused segments)
    - DETAILED: Each segment should have substantial, detailed content
    - SPECIFIC: Use specific information from the provided text
    - PRACTICAL: Include real examples and applications when available
    - Return ONLY valid JSON, no additional text
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are an expert educational content designer specialized in comprehensive learning material creation. Ensure NO content is overlooked or oversimplified."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=max_tokens,
            temperature=0.2  # Lower temperature for more consistent, comprehensive output
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Clean response
        if response_content.startswith('```json'):
            response_content = response_content[7:]
        if response_content.endswith('```'):
            response_content = response_content[:-3]
        
        block_data = json.loads(response_content.strip())
        
        # ✅ Combine segments and Q&A (same as template function)
        original_segments = block_data.get('segments', [])
        qa_items = block_data.get('qa_items', [])
        
        # Filter out empty or placeholder segments
        filtered_segments = []
        for segment in original_segments:
            content = segment.get('content', '').strip()
            if content and len(content) > 50:  # Only include substantial content
                filtered_segments.append(segment)
        
        all_segments = list(filtered_segments)
        
        # Add Q&A as additional segments
        for i, qa in enumerate(qa_items):
            question_text = qa.get('question', '')
            answer_text = qa.get('answer', '')
            
            qa_segment = {
                'type': 'knowledge_check',
                'title': f'Knowledge Check {i+1}',
                'question': question_text,
                'answer': answer_text,
                'difficulty': qa.get('difficulty', ''),
                'content': f"<strong>Q: {question_text}</strong><br><br><em>A: {answer_text}</em>",
                'duration_seconds': 90
            }
            all_segments.append(qa_segment)
        
        # ✅ Create comprehensive block data
        title = block_data.get('title', f'Concept Block {block_order}')
        if title_suffix:
            title += f" - {title_suffix}"
            
        combined_block_data = {
            'title': title,
            'learning_objectives': block_data.get('learning_objectives', []),
            'segments': all_segments,
            'qa_items': [],  # Empty since Q&A is now in segments
            'total_duration': sum(s.get('duration_seconds', 90) for s in all_segments),
            'difficulty_level': block_data.get('difficulty_level', 'intermediate')
        }
        
        # ✅ Create concept unit
        max_order = ConceptUnit.objects.filter(pdf_document=pdf_document).aggregate(
            max_order=models.Max('concept_order')
        )['max_order']
        next_order = (max_order or 0) + 1
        
        concept_unit = ConceptUnit.objects.create(
            pdf_document=pdf_document,
            title=title,
            concept_order=next_order,
            complexity_score=0.75
        )
        
        # ✅ Create focus block
        focus_block = FocusBlock.objects.create(
            pdf_document=pdf_document,
            main_concept_unit=concept_unit,
            block_order=next_order,
            title=title,
            target_duration=combined_block_data.get('total_duration', 420),
            compact7_data=combined_block_data,
            difficulty_level=combined_block_data.get('difficulty_level', 'intermediate'),
            learning_objectives=combined_block_data.get('learning_objectives', [])
        )
        
        logger.info(f"   ✅ Created focus block: '{title}' with {len(all_segments)} segments")
        return focus_block
        
    except Exception as e:
        raise Exception(f"Failed to create comprehensive focus block: {str(e)}")


# ✅ ENHANCED COMPLETE PIPELINE: Extract → Chunk → Generate → Embed → Deduplicate → Knowledge Graph
@shared_task
def complete_pdf_processing_with_knowledge_graph(pdf_document_id, similarity_threshold=0.60, dedup_threshold=0.85, kg_threshold=0.75):
    """
    Complete pipeline with knowledge graph: Extract → Chunk → Generate → Embed → Deduplicate → Build Knowledge Graph
    """
    import time
    start_time = time.time()
    
    logger.info(f"🚀 Starting complete PDF processing with knowledge graph for ID: {pdf_document_id}")
    
    try:
        # Steps 1-5: Run the existing pipeline with deduplication
        logger.info("📄 Steps 1-5: Running complete pipeline with deduplication...")
        pipeline_result = complete_pdf_processing_with_deduplication(
            pdf_document_id, similarity_threshold, dedup_threshold
        )
        
        if not pipeline_result['success']:
            return pipeline_result
        
        # Step 6: Generate knowledge graph
        logger.info("🕸️ Step 6: Generating knowledge graph...")
        kg_result = generate_knowledge_graph_task(
            pdf_document_id=pdf_document_id,
            relationship_threshold=kg_threshold,
            rebuild=False
        )
        
        # Step 7: Generate optimal study paths
        logger.info("🎯 Step 7: Generating optimal study paths...")
        paths_result = generate_optimal_study_paths_task(
            pdf_document_id=pdf_document_id,
            path_strategy='prerequisite_first'
        )
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'message': f"Complete pipeline with knowledge graph finished in {total_time:.2f}s",
            'total_time': total_time,
            'results': {
                'pipeline': pipeline_result,
                'knowledge_graph': kg_result,
                'study_paths': paths_result
            }
        }
        
    except Exception as e:
        error_msg = f"Enhanced pipeline with knowledge graph failed: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'message': error_msg, 'results': {}}

# Add these imports at the top of flashcards/tasks.py (if not already there)
import networkx as nx
from collections import defaultdict, deque

# ✅ ADD: Missing Knowledge Graph Tasks

@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def generate_knowledge_graph_task(self, pdf_document_id=None, relationship_threshold=0.75, rebuild=False):
    """
    Generate comprehensive knowledge graph for focus blocks
    
    This task:
    1. Analyzes all focus blocks (or specific PDF) for relationships
    2. Uses embeddings + LLM analysis to identify prerequisites and dependencies  
    3. Creates FocusBlockRelationship records
    4. Builds NetworkX graph for analysis
    5. Generates optimal study sequences
    
    Args:
        pdf_document_id: Process only blocks from this PDF (None = all blocks)
        relationship_threshold: Minimum confidence for creating relationships (0.75)
        rebuild: If True, delete existing relationships and rebuild
        
    Returns:
        dict: Knowledge graph generation results
    """
    start_time = time.time()
    logger.info(f"🕸️ Starting knowledge graph generation for PDF ID: {pdf_document_id or 'ALL'}")
    
    try:
        from .models import FocusBlock, FocusBlockRelationship
        
        # Get focus blocks to analyze
        if pdf_document_id:
            blocks = FocusBlock.objects.filter(
                pdf_document_id=pdf_document_id,
                content_embedding__isnull=False
            ).exclude(content_embedding=[])
        else:
            blocks = FocusBlock.objects.filter(
                content_embedding__isnull=False
            ).exclude(content_embedding=[])
        
        if blocks.count() < 2:
            return {
                'success': True,
                'message': 'Not enough blocks with embeddings for relationship analysis',
                'data': {'relationships_created': 0, 'blocks_analyzed': blocks.count()}
            }
        
        logger.info(f"📊 Analyzing {blocks.count()} blocks for relationships")
        
        # Clean up existing relationships if rebuilding
        if rebuild:
            if pdf_document_id:
                FocusBlockRelationship.objects.filter(
                    from_block__pdf_document_id=pdf_document_id
                ).delete()
            else:
                FocusBlockRelationship.objects.all().delete()
            logger.info(f"🧹 Cleared existing relationships for rebuild")
        
        # Step 1: Find semantic similarities using embeddings
        logger.info("🔢 Step 1: Analyzing semantic similarities...")
        similarity_relationships = analyze_semantic_similarities(blocks, relationship_threshold)
        
        # Step 2: Use LLM to identify prerequisites and dependencies
        logger.info("🧠 Step 2: LLM analysis for prerequisites and dependencies...")
        prerequisite_relationships = analyze_prerequisites_with_llm(blocks, relationship_threshold)
        
        # Step 3: Combine and create relationship records
        logger.info("💾 Step 3: Creating relationship records...")
        all_relationships = similarity_relationships + prerequisite_relationships
        relationships_created = create_relationship_records(all_relationships)
        
        # Step 4: Build and analyze the knowledge graph
        logger.info("🕸️ Step 4: Building knowledge graph...")
        graph_analysis = build_and_analyze_knowledge_graph(blocks)
        
        processing_time = time.time() - start_time
        
        result_data = {
            'relationships_created': relationships_created,
            'blocks_analyzed': blocks.count(),
            'processing_time': processing_time,
            'graph_metrics': graph_analysis,
            'similarity_relationships': len(similarity_relationships),
            'prerequisite_relationships': len(prerequisite_relationships)
        }
        
        logger.info(f"✅ Knowledge graph generation completed:")
        logger.info(f"   📊 {relationships_created} relationships created")
        logger.info(f"   🔗 {graph_analysis['edges']} total edges in graph")
        logger.info(f"   🎯 {graph_analysis['strongly_connected_components']} strongly connected components")
        logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
        
        return {
            'success': True,
            'message': f"Generated knowledge graph with {relationships_created} relationships in {processing_time:.2f}s",
            'data': result_data
        }
        
    except Exception as e:
        error_msg = f"Knowledge graph generation failed: {str(e)}"
        logger.error(error_msg)
        
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying in {self.default_retry_delay}s")
            raise self.retry(countdown=self.default_retry_delay, exc=e)
        
        return {'success': False, 'message': error_msg, 'data': {}}


def analyze_semantic_similarities(blocks, threshold=0.75):
    """
    Analyze semantic similarities between focus blocks using embeddings
    """
    relationships = []
    blocks_list = list(blocks)
    
    if len(blocks_list) < 2:
        return relationships
    
    # Get embeddings
    embeddings = []
    valid_blocks = []
    
    for block in blocks_list:
        if block.content_embedding and len(block.content_embedding) > 0:
            embeddings.append(block.content_embedding)
            valid_blocks.append(block)
    
    if len(embeddings) < 2:
        return relationships
    
    # Calculate similarity matrix
    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Find related blocks
    for i in range(len(valid_blocks)):
        for j in range(i + 1, len(valid_blocks)):
            similarity = similarity_matrix[i][j]
            
            if similarity >= threshold:
                block1 = valid_blocks[i]
                block2 = valid_blocks[j]
                
                # Determine relationship type based on similarity level
                if similarity >= 0.90:
                    rel_type = 'related'  # Very similar content
                    confidence = min(similarity, 0.95)
                elif similarity >= 0.80:
                    rel_type = 'builds_on'  # One might build on the other
                    confidence = similarity * 0.9
                else:
                    rel_type = 'related'  # Moderately related
                    confidence = similarity * 0.8
                
                relationships.append({
                    'from_block': block1,
                    'to_block': block2,
                    'relationship_type': rel_type,
                    'confidence': confidence,
                    'similarity_score': similarity,
                    'source': 'embedding_similarity'
                })
    
    logger.info(f"   📊 Found {len(relationships)} semantic similarity relationships")
    return relationships


def analyze_prerequisites_with_llm(blocks, threshold=0.75):
    """
    Use LLM to analyze focus blocks and identify prerequisite relationships
    """
    from openai import OpenAI
    from django.conf import settings
    import json
    
    relationships = []
    
    # Initialize OpenAI client
    api_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not api_key:
        logger.warning("⚠️ OpenAI API key not configured - skipping LLM prerequisite analysis")
        return relationships
    
    client = OpenAI(api_key=api_key)
    
    blocks_list = list(blocks)
    
    # Process blocks in groups for comprehensive analysis
    batch_size = 5  # Analyze 5 blocks at a time to avoid token limits
    
    for i in range(0, len(blocks_list), batch_size):
        batch = blocks_list[i:i + batch_size]
        
        try:
            batch_relationships = analyze_batch_prerequisites(client, batch, threshold)
            relationships.extend(batch_relationships)
        except Exception as batch_error:
            logger.error(f"❌ Failed to analyze batch {i//batch_size + 1}: {str(batch_error)}")
            continue
    
    logger.info(f"   🧠 Found {len(relationships)} LLM-identified prerequisite relationships")
    return relationships


def analyze_batch_prerequisites(client, blocks_batch, threshold):
    """
    Analyze a batch of blocks for prerequisite relationships using LLM
    """
    import json
    
    # Prepare block summaries for AI analysis
    block_summaries = []
    for idx, block in enumerate(blocks_batch):
        summary = {
            'id': str(block.id),
            'index': idx,
            'title': block.title,
            'learning_objectives': block.learning_objectives or [],
            'difficulty_level': block.difficulty_level,
            'key_concepts': extract_key_concepts_from_block(block)
        }
        block_summaries.append(summary)
    
    # Enhanced prompt for prerequisite analysis
    prompt = f"""
    Analyze the following focus blocks and identify prerequisite relationships between them.
    
    FOCUS BLOCKS:
    {json.dumps(block_summaries, indent=2)}
    
    For each pair of blocks, determine if one is a prerequisite for another by considering:
    1. CONCEPTUAL PREREQUISITES: Does Block A introduce concepts needed for Block B?
    2. SKILL DEPENDENCIES: Does Block A teach skills required for Block B?
    3. DIFFICULTY PROGRESSION: Should Block A naturally come before Block B in learning?
    4. LOGICAL SEQUENCE: Is there a logical learning order?
    
    Return a JSON array of relationships in this format:
    [
        {{
            "from_block_index": 0,
            "to_block_index": 1,
            "relationship_type": "prerequisite|builds_on|applies_to|specializes",
            "confidence": 0.85,
            "description": "Detailed explanation of why this relationship exists",
            "educational_reasoning": "How this relationship helps learning progression"
        }}
    ]
    
    RELATIONSHIP TYPES:
    - "prerequisite": Block A must be learned before Block B (strong dependency)
    - "builds_on": Block B extends concepts from Block A (medium dependency)  
    - "applies_to": Block A provides theory, Block B shows applications
    - "specializes": Block B is a specific case of Block A
    
    GUIDELINES:
    - Only include relationships with confidence >= 0.7
    - Focus on clear prerequisite chains and logical progressions
    - Avoid weak or unclear relationships
    - Consider difficulty levels (basic → intermediate → advanced)
    - Return empty array if no strong relationships exist
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are an expert educational analyst who identifies prerequisite relationships between learning materials. Focus on creating optimal learning sequences."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=2000,
            temperature=0.2
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Clean and parse response
        if response_content.startswith('```json'):
            response_content = response_content[7:]
        if response_content.endswith('```'):
            response_content = response_content[:-3]
        
        relationship_data = json.loads(response_content.strip())
        
        # Convert to relationship objects
        relationships = []
        for rel in relationship_data:
            if rel.get('confidence', 0) >= threshold:
                from_idx = rel.get('from_block_index')
                to_idx = rel.get('to_block_index')
                
                if (from_idx is not None and to_idx is not None and 
                    0 <= from_idx < len(blocks_batch) and 0 <= to_idx < len(blocks_batch)):
                    
                    relationships.append({
                        'from_block': blocks_batch[from_idx],
                        'to_block': blocks_batch[to_idx],
                        'relationship_type': rel.get('relationship_type', 'related'),
                        'confidence': rel.get('confidence', 0.75),
                        'similarity_score': 0.0,  # Not from embedding similarity
                        'description': rel.get('description', ''),
                        'educational_reasoning': rel.get('educational_reasoning', ''),
                        'source': 'llm_analysis'
                    })
        
        return relationships
        
    except Exception as e:
        logger.error(f"❌ LLM analysis failed: {str(e)}")
        return []


def extract_key_concepts_from_block(focus_block):
    """
    Extract key concepts from a focus block for analysis
    """
    concepts = []
    
    # Add concepts from learning objectives
    if focus_block.learning_objectives:
        concepts.extend(focus_block.learning_objectives)
    
    # Extract from segment titles and content
    if focus_block.compact7_data and 'segments' in focus_block.compact7_data:
        for segment in focus_block.compact7_data['segments']:
            # Add segment titles
            title = segment.get('title', '')
            if title:
                concepts.append(title)
            
            # Extract key terms from content (basic extraction)
            content = segment.get('content', '')
            if content:
                # Simple keyword extraction (could be enhanced)
                import re
                clean_content = re.sub(r'<[^>]+>', '', content)
                # Look for capitalized terms that might be concepts
                key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_content)
                concepts.extend(key_terms[:3])  # Limit to avoid noise
    
    # Return unique concepts, limited to avoid prompt bloat
    unique_concepts = list(set(concepts))[:10]
    return unique_concepts


def create_relationship_records(relationships):
    """
    Create FocusBlockRelationship database records from analysis results
    """
    from .models import FocusBlockRelationship
    from django.db import transaction
    
    created_count = 0
    
    with transaction.atomic():
        for rel_data in relationships:
            try:
                # Check if relationship already exists
                existing = FocusBlockRelationship.objects.filter(
                    from_block=rel_data['from_block'],
                    to_block=rel_data['to_block'],
                    relationship_type=rel_data['relationship_type']
                ).first()
                
                if not existing:
                    # Calculate edge strength (combination of confidence and similarity)
                    edge_strength = (
                        rel_data['confidence'] * 0.7 + 
                        rel_data['similarity_score'] * 0.3
                    )
                    
                    FocusBlockRelationship.objects.create(
                        from_block=rel_data['from_block'],
                        to_block=rel_data['to_block'],
                        relationship_type=rel_data['relationship_type'],
                        confidence=rel_data['confidence'],
                        similarity_score=rel_data['similarity_score'],
                        edge_strength=edge_strength,
                        description=rel_data.get('description', f"Auto-generated {rel_data['relationship_type']} relationship"),
                        educational_reasoning=rel_data.get('educational_reasoning', f"Learning {rel_data['from_block'].title} helps with {rel_data['to_block'].title}")
                    )
                    created_count += 1
                    
                    # Create bidirectional relationship for symmetric types
                    if rel_data['relationship_type'] in ['related', 'compares_with']:
                        reverse_exists = FocusBlockRelationship.objects.filter(
                            from_block=rel_data['to_block'],
                            to_block=rel_data['from_block'],
                            relationship_type=rel_data['relationship_type']
                        ).exists()
                        
                        if not reverse_exists:
                            FocusBlockRelationship.objects.create(
                                from_block=rel_data['to_block'],
                                to_block=rel_data['from_block'],
                                relationship_type=rel_data['relationship_type'],
                                confidence=rel_data['confidence'],
                                similarity_score=rel_data['similarity_score'],
                                edge_strength=edge_strength,
                                description=rel_data.get('description', f"Auto-generated {rel_data['relationship_type']} relationship"),
                                educational_reasoning=rel_data.get('educational_reasoning', f"Learning {rel_data['to_block'].title} helps with {rel_data['from_block'].title}")
                            )
                            created_count += 1
                
            except Exception as create_error:
                logger.error(f"❌ Failed to create relationship: {str(create_error)}")
                continue
    
    return created_count


def build_and_analyze_knowledge_graph(blocks):
    """
    Build NetworkX graph and analyze knowledge structure
    """
    from .models import FocusBlockRelationship
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for block in blocks:
        G.add_node(str(block.id), 
                  title=block.title,
                  difficulty=block.difficulty_level,
                  duration=block.target_duration)
    
    # Add edges from relationships
    relationships = FocusBlockRelationship.objects.filter(
        from_block__in=blocks,
        to_block__in=blocks
    )
    
    for rel in relationships:
        G.add_edge(str(rel.from_block.id), str(rel.to_block.id),
                  relationship=rel,
                  type=rel.relationship_type,
                  confidence=rel.confidence,
                  strength=rel.edge_strength)
    
    # Analyze graph structure
    analysis = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'strongly_connected_components': nx.number_strongly_connected_components(G),
        'weakly_connected_components': nx.number_weakly_connected_components(G),
        'is_dag': nx.is_directed_acyclic_graph(G),
        'average_clustering': nx.average_clustering(G.to_undirected()),
    }
    
    # Find optimal study sequences
    if analysis['is_dag']:
        try:
            topological_order = list(nx.topological_sort(G))
            analysis['optimal_study_sequence'] = topological_order
        except:
            analysis['optimal_study_sequence'] = []
    else:
        analysis['optimal_study_sequence'] = []
    
    return analysis


@shared_task
def generate_optimal_study_paths_task(pdf_document_id=None, path_strategy='prerequisite_first'):
    """
    Generate optimal study paths based on the knowledge graph
    
    Args:
        pdf_document_id: Generate paths for specific PDF (None = all blocks)
        path_strategy: 'prerequisite_first', 'difficulty_progression', 'concept_clusters'
        
    Returns:
        dict: Generated study paths
    """
    logger.info(f"🎯 Generating optimal study paths using strategy: {path_strategy}")
    
    try:
        from .models import FocusBlock, FocusBlockRelationship
        
        # Get focus blocks
        if pdf_document_id:
            blocks = FocusBlock.objects.filter(pdf_document_id=pdf_document_id)
        else:
            blocks = FocusBlock.objects.all()
        
        if blocks.count() < 2:
            return {
                'success': True,
                'message': 'Not enough blocks for path generation',
                'data': {'paths': []}
            }
        
        # Build knowledge graph
        G = build_knowledge_graph_networkx(blocks)
        
        # Generate paths based on strategy
        if path_strategy == 'prerequisite_first':
            paths = generate_prerequisite_based_paths(G, blocks)
        elif path_strategy == 'difficulty_progression':
            paths = generate_difficulty_based_paths(G, blocks)
        elif path_strategy == 'concept_clusters':
            paths = generate_cluster_based_paths(G, blocks)
        else:
            paths = generate_prerequisite_based_paths(G, blocks)  # Default
        
        return {
            'success': True,
            'message': f"Generated {len(paths)} optimal study paths",
            'data': {
                'paths': paths,
                'strategy': path_strategy,
                'total_blocks': blocks.count()
            }
        }
        
    except Exception as e:
        error_msg = f"Study path generation failed: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'message': error_msg, 'data': {}}


def build_knowledge_graph_networkx(blocks):
    """
    Build NetworkX graph from focus blocks and relationships
    """
    from .models import FocusBlockRelationship
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for block in blocks:
        G.add_node(str(block.id), 
                  block=block,
                  title=block.title,
                  difficulty=block.difficulty_level,
                  duration=block.target_duration,
                  order=block.block_order)
    
    # Add edges from relationships
    relationships = FocusBlockRelationship.objects.filter(
        from_block__in=blocks,
        to_block__in=blocks
    )
    
    for rel in relationships:
        G.add_edge(str(rel.from_block.id), str(rel.to_block.id),
                  relationship=rel,
                  type=rel.relationship_type,
                  confidence=rel.confidence,
                  strength=rel.edge_strength)
    
    return G


def generate_prerequisite_based_paths(graph, blocks):
    """
    Generate study paths based on prerequisite relationships
    """
    paths = []
    
    # Find strongly connected components and topologically sort
    if nx.is_directed_acyclic_graph(graph):
        # Perfect - we can do topological sort
        topo_order = list(nx.topological_sort(graph))
        
        main_path = []
        for node_id in topo_order:
            block = next((b for b in blocks if str(b.id) == node_id), None)
            if block:
                main_path.append({
                    'block_id': str(block.id),
                    'title': block.title,
                    'difficulty': block.difficulty_level,
                    'duration': block.target_duration,
                    'prerequisites_satisfied': True
                })
        
        if main_path:
            paths.append({
                'path_id': 'prerequisite_optimal',
                'name': 'Optimal Prerequisite Path',
                'description': 'Follows prerequisite relationships for optimal learning',
                'blocks': main_path,
                'total_duration': sum(b['duration'] for b in main_path),
                'estimated_completion_hours': sum(b['duration'] for b in main_path) / 3600
            })
    
    else:
        # Graph has cycles - create multiple paths for connected components
        components = list(nx.strongly_connected_components(graph))
        
        for i, component in enumerate(components):
            component_blocks = []
            for node_id in component:
                block = next((b for b in blocks if str(b.id) == node_id), None)
                if block:
                    component_blocks.append(block)
            
            if component_blocks:
                component_path = []
                
                # Sort by difficulty within component
                sorted_blocks = sorted(component_blocks, 
                                     key=lambda x: (x.difficulty_level, x.block_order))
                
                for block in sorted_blocks:
                    component_path.append({
                        'block_id': str(block.id),
                        'title': block.title,
                        'difficulty': block.difficulty_level,
                        'duration': block.target_duration,
                        'component': i
                    })
                
                paths.append({
                    'path_id': f'component_{i}',
                    'name': f'Learning Component {i + 1}',
                    'description': f'Interconnected concepts that can be learned together',
                    'blocks': component_path,
                    'total_duration': sum(b['duration'] for b in component_path),
                    'estimated_completion_hours': sum(b['duration'] for b in component_path) / 3600
                })
    
    return paths


def generate_difficulty_based_paths(graph, blocks):
    """
    Generate study paths based on difficulty progression
    """
    difficulty_order = ['beginner', 'intermediate', 'advanced']
    
    # Group blocks by difficulty
    difficulty_groups = {level: [] for level in difficulty_order}
    
    for block in blocks:
        level = block.difficulty_level
        if level in difficulty_groups:
            difficulty_groups[level].append(block)
    
    # Create paths for each difficulty level
    paths = []
    
    for level in difficulty_order:
        level_blocks = difficulty_groups[level]
        if not level_blocks:
            continue
        
        # Sort by original block order within difficulty level
        sorted_blocks = sorted(level_blocks, key=lambda x: x.block_order)
        
        path_blocks = []
        for block in sorted_blocks:
            path_blocks.append({
                'block_id': str(block.id),
                'title': block.title,
                'difficulty': block.difficulty_level,
                'duration': block.target_duration,
                'order_in_level': len(path_blocks) + 1
            })
        
        if path_blocks:
            paths.append({
                'path_id': f'difficulty_{level}',
                'name': f'{level.title()} Level Path',
                'description': f'All {level} level concepts in logical order',
                'blocks': path_blocks,
                'total_duration': sum(b['duration'] for b in path_blocks),
                'estimated_completion_hours': sum(b['duration'] for b in path_blocks) / 3600,
                'difficulty_level': level
            })
    
    return paths


def generate_cluster_based_paths(graph, blocks):
    """
    Generate study paths based on concept clusters (communities in the graph)
    """
    try:
        # Convert to undirected for community detection
        undirected_graph = graph.to_undirected()
        
        # Use simple connected components as fallback
        components = list(nx.connected_components(undirected_graph))
        paths = []
        
        for i, component in enumerate(components):
            component_blocks = []
            for node_id in component:
                block = next((b for b in blocks if str(b.id) == node_id), None)
                if block:
                    component_blocks.append(block)
            
            if component_blocks:
                # Sort blocks within community by difficulty and order
                sorted_blocks = sorted(component_blocks, 
                                     key=lambda x: (x.difficulty_level, x.block_order))
                
                path_blocks = []
                for block in sorted_blocks:
                    path_blocks.append({
                        'block_id': str(block.id),
                        'title': block.title,
                        'difficulty': block.difficulty_level,
                        'duration': block.target_duration,
                        'cluster': i
                    })
                
                paths.append({
                    'path_id': f'cluster_{i}',
                    'name': f'Concept Cluster {i + 1}',
                    'description': f'Related concepts that form a coherent learning unit',
                    'blocks': path_blocks,
                    'total_duration': sum(b['duration'] for b in path_blocks),
                    'estimated_completion_hours': sum(b['duration'] for b in path_blocks) / 3600,
                    'cluster_id': i
                })
        
        return paths
        
    except Exception as e:
        logger.warning(f"⚠️ Cluster-based path generation failed: {e}, using simple ordering")
        
        # Fallback: simple ordering by difficulty
        return generate_difficulty_based_paths(graph, blocks)


