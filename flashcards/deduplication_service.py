import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from .models import PDFDocument, TextChunk, ConceptUnit, FocusBlock

logger = logging.getLogger(__name__)

class PDFDeduplicationService:
    """Service to detect and handle duplicate PDFs"""
    
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.min_content_length = 100  # Minimum content length to process
    
    def check_for_duplicates(self, pdf_document: PDFDocument) -> Tuple[bool, Optional[PDFDocument], float]:
        """
        Check if PDF is duplicate of existing content
        Returns: (is_duplicate, original_pdf, similarity_score)
        """
        cleaned_text = pdf_document.get_cleaned_text()
        if not cleaned_text or len(cleaned_text) < self.min_content_length:
            return False, None, 0.0
        
        # Generate content identifiers
        content_hash = pdf_document.calculate_content_hash()
        content_fingerprint = pdf_document.create_content_fingerprint()
        
        # Update PDF with identifiers
        pdf_document.content_hash = content_hash
        pdf_document.content_fingerprint = content_fingerprint
        pdf_document.save()
        
        print(f"🔍 DEBUG: Checking duplicates for '{pdf_document.name}'")
        print(f"🔍 DEBUG: Content hash: {content_hash[:16]}...")
        print(f"🔍 DEBUG: Fingerprint length: {len(content_fingerprint)} chars")
        
        # Step 1: Check for exact hash match
        exact_match = PDFDocument.objects.filter(
            content_hash=content_hash,
            processed=True,
            is_duplicate=False
        ).exclude(id=pdf_document.id).first()
        
        if exact_match:
            print(f"✅ DEBUG: Found exact hash match with '{exact_match.name}'")
            return True, exact_match, 1.0
        
        # Step 2: Check for fuzzy similarity
        similar_pdfs = pdf_document.find_similar_pdfs(self.similarity_threshold)
        
        if similar_pdfs:
            best_match = similar_pdfs[0]
            similarity = best_match['similarity']
            original_pdf = best_match['pdf']
            
            print(f"📊 DEBUG: Found similar PDF '{original_pdf.name}' with {similarity:.2%} similarity")
            return True, original_pdf, similarity
        
        print(f"🆕 DEBUG: No duplicates found. PDF is unique.")
        return False, None, 0.0
    
    def handle_duplicate(self, duplicate_pdf: PDFDocument, original_pdf: PDFDocument, similarity: float):
        """Handle a detected duplicate by copying data from original"""
        print(f"🔄 DEBUG: Handling duplicate: copying data from '{original_pdf.name}' to '{duplicate_pdf.name}'")
        
        # Mark as duplicate
        duplicate_pdf.is_duplicate = True
        duplicate_pdf.duplicate_of = original_pdf
        duplicate_pdf.similarity_score = similarity
        duplicate_pdf.processed = True
        duplicate_pdf.concepts_analyzed = True
        duplicate_pdf.save()
        
        # Copy processed data structures
        copied_chunks = self.copy_text_chunks(original_pdf, duplicate_pdf)
        copied_concepts = self.copy_concept_units(original_pdf, duplicate_pdf)
        copied_blocks = self.copy_focus_blocks(original_pdf, duplicate_pdf)
        
        print(f"✅ DEBUG: Copied {copied_chunks} chunks, {copied_concepts} concepts, {copied_blocks} focus blocks")
        
        return {
            'chunks': copied_chunks,
            'concepts': copied_concepts,
            'focus_blocks': copied_blocks,
            'similarity': similarity
        }
    
    def copy_text_chunks(self, source_pdf: PDFDocument, target_pdf: PDFDocument) -> int:
        """Copy text chunks from source to target PDF"""
        source_chunks = source_pdf.text_chunks.all().order_by('chunk_order')
        copied_count = 0
        
        for chunk in source_chunks:
            TextChunk.objects.create(
                pdf_document=target_pdf,
                chunk_text=chunk.chunk_text,
                chunk_order=chunk.chunk_order,
                word_start=chunk.word_start,
                word_end=chunk.word_end,
                word_count=chunk.word_count
            )
            copied_count += 1
        
        return copied_count
    
    def copy_concept_units(self, source_pdf: PDFDocument, target_pdf: PDFDocument) -> int:
        """Copy concept units and their chunk labels"""
        from .models import ChunkLabel
        
        source_concepts = source_pdf.concept_units.all().order_by('concept_order')
        target_chunks = target_pdf.text_chunks.all().order_by('chunk_order')
        copied_count = 0
        
        # Create chunk mapping (same order)
        chunk_mapping = {}
        source_chunks = source_pdf.text_chunks.all().order_by('chunk_order')
        for i, (source_chunk, target_chunk) in enumerate(zip(source_chunks, target_chunks)):
            chunk_mapping[source_chunk.id] = target_chunk
        
        for concept in source_concepts:
            # Create new concept unit
            new_concept = ConceptUnit.objects.create(
                pdf_document=target_pdf,
                title=concept.title,
                description=concept.description,
                concept_order=concept.concept_order,
                primary_labels=concept.primary_labels,
                estimated_reading_time=concept.estimated_reading_time,
                cognitive_load=concept.cognitive_load,
                word_count=concept.word_count
            )
            
            # Copy chunk labels
            source_labels = concept.chunk_labels.all()
            for label in source_labels:
                if label.text_chunk.id in chunk_mapping:
                    target_chunk = chunk_mapping[label.text_chunk.id]
                    ChunkLabel.objects.create(
                        text_chunk=target_chunk,
                        concept_unit=new_concept,
                        label_type=label.label_type,
                        confidence_score=label.confidence_score,
                        concept_keywords=label.concept_keywords
                    )
            
            copied_count += 1
        
        return copied_count
    
    def copy_focus_blocks(self, source_pdf: PDFDocument, target_pdf: PDFDocument) -> int:
        """Copy focus blocks with updated concept unit references"""
        source_blocks = source_pdf.focus_blocks.all().order_by('block_order')
        target_concepts = list(target_pdf.concept_units.all().order_by('concept_order'))
        copied_count = 0
        
        for i, block in enumerate(source_blocks):
            # Map to corresponding concept units
            main_concept = target_concepts[i] if i < len(target_concepts) else target_concepts[0]
            revision_concept = target_concepts[i-1] if i > 0 and i-1 < len(target_concepts) else None
            
            FocusBlock.objects.create(
                pdf_document=target_pdf,
                main_concept_unit=main_concept,
                revision_concept_unit=revision_concept,
                block_order=block.block_order,
                title=block.title,
                target_duration=block.target_duration,
                compact7_data=block.compact7_data,  # Copy the complete JSON
                difficulty_level=block.difficulty_level,
                learning_objectives=block.learning_objectives,
                prerequisite_concepts=block.prerequisite_concepts
            )
            copied_count += 1
        
        return copied_count
    
    def get_deduplication_stats(self) -> Dict:
        """Get statistics about duplicates in the system"""
        total_pdfs = PDFDocument.objects.count()
        unique_pdfs = PDFDocument.objects.filter(is_duplicate=False).count()
        duplicate_pdfs = PDFDocument.objects.filter(is_duplicate=True).count()
        
        # Group by original
        originals_with_dupes = PDFDocument.objects.filter(
            duplicates__isnull=False
        ).distinct().count()
        
        return {
            'total_pdfs': total_pdfs,
            'unique_pdfs': unique_pdfs,
            'duplicate_pdfs': duplicate_pdfs,
            'originals_with_duplicates': originals_with_dupes,
            'deduplication_ratio': duplicate_pdfs / total_pdfs if total_pdfs > 0 else 0
        }

# Convenience function
def check_pdf_duplicates(pdf_document: PDFDocument, similarity_threshold=0.8):
    """Check and handle PDF duplicates"""
    service = PDFDeduplicationService(similarity_threshold)
    return service.check_for_duplicates(pdf_document) 