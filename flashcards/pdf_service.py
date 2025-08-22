import PyPDF2
import pdfplumber
import logging
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .text_processor import process_text
from .models import TextChunk
from .concept_service import analyze_pdf_concepts
import os
import tempfile
import io

logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """Service class to extract and process text from PDF files"""
    
    def __init__(self):
        self.max_pages = 100  # Limit to prevent processing huge PDFs
    
    def extract_text_pypdf2(self, pdf_path):
        """Extract text using PyPDF2 (faster but less accurate)"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                page_count = len(reader.pages)
                
                pages_to_process = min(page_count, self.max_pages)
                
                for page_num in range(pages_to_process):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                return text.strip(), page_count
        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return None, 0
    
    def extract_text_pypdf2_from_content(self, file_content):
        """Extract text using PyPDF2 from file content (for Railway compatibility)"""
        try:
            # Create BytesIO object from file content
            file_stream = io.BytesIO(file_content)
            reader = PyPDF2.PdfReader(file_stream)
            text = ""
            page_count = len(reader.pages)
            
            pages_to_process = min(page_count, self.max_pages)
            
            for page_num in range(pages_to_process):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            return text.strip(), page_count
        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return None, 0
    
    def extract_text_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (more accurate but slower)"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                page_count = len(pdf.pages)
                
                pages_to_process = min(page_count, self.max_pages)
                
                for page_num in range(pages_to_process):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                return text.strip(), page_count
        
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return None, 0
    
    def extract_text_pdfplumber_from_content(self, file_content):
        """Extract text using pdfplumber from file content (for Railway compatibility)"""
        try:
            # Create BytesIO object from file content
            file_stream = io.BytesIO(file_content)
            with pdfplumber.open(file_stream) as pdf:
                text = ""
                page_count = len(pdf.pages)
                
                pages_to_process = min(page_count, self.max_pages)
                
                for page_num in range(pages_to_process):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                return text.strip(), page_count
        
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return None, 0
    
    def extract_text_from_file_content(self, pdf_document):
        """Extract text from PDF using file content instead of file path (Railway compatible)"""
        try:
            logger.info(f"📂 Found PDF: {pdf_document.name}")
            
            # Read file content using Django's file API
            with pdf_document.pdf_file.open('rb') as file:
                file_content = file.read()
            
            pdf_document.file_size = len(file_content)
            
            # Try pdfplumber first
            text, page_count = self.extract_text_pdfplumber_from_content(file_content)
            extraction_method = "pdfplumber"
            
            # Fallback to PyPDF2 if needed
            if not text or len(text.strip()) < 50:
                logger.info("📄 pdfplumber failed, trying PyPDF2 fallback...")
                text, page_count = self.extract_text_pypdf2_from_content(file_content)
                extraction_method = "pypdf2"
            
            # Validate extraction
            if not text or len(text.strip()) < 50:
                error_msg = f"Could not extract meaningful text from PDF: {pdf_document.name}"
                logger.error(error_msg)
                return None, 0, None
            
            logger.info(f"✅ Text extracted successfully using {extraction_method}: {len(text)} characters, {page_count} pages")
            return text, page_count, extraction_method
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return None, 0, None
    
    def extract_and_process_pdf(self, pdf_document, analyze_concepts=True):
        """
        Main method to extract, clean, chunk, and analyze PDF text
        Returns tuple: (success, error_message, processing_stats)
        """
        try:
            # Step 1: Extract raw text using file content (Railway compatible)
            raw_text, page_count, extraction_method = self.extract_text_from_file_content(pdf_document)
            
            if not raw_text:
                return False, "Could not extract text from PDF. The file might be scanned images or corrupted.", {}
            
            if len(raw_text.strip()) < 50:
                return False, "Very little text found in PDF. It might be image-based or encrypted.", {}
            
            # Step 2: Process and clean text
            cleaned_text, chunks, stats = process_text(raw_text)
            
            # Step 3: Update PDF document
            pdf_document.extracted_text = raw_text
            # Note: cleaned_text is now generated on-demand via get_cleaned_text()
            pdf_document.page_count = page_count
            pdf_document.word_count = stats['words']
            pdf_document.processed = True
            pdf_document.save()
            
            # Step 4: Create text chunks
            self.create_text_chunks(pdf_document, chunks)
            
            processing_stats = {
                'page_count': page_count,
                'raw_text_length': len(raw_text),
                'cleaned_text_length': len(cleaned_text),
                'word_count': stats['words'],
                'chunk_count': len(chunks),
                'paragraphs': stats['paragraphs'],
                'concept_units': 0,
                'concept_analysis_success': False,
                'extraction_method': extraction_method
            }
            
            # Step 5: Analyze concepts with LLM (optional)
            print(f"🔍 DEBUG: analyze_concepts flag is: {analyze_concepts}")  # ✅ Debug print
            
            if analyze_concepts:
                try:
                    print(f"🧠 DEBUG: Starting concept analysis...")  # ✅ Debug print
                    
                    from .concept_service import analyze_pdf_concepts
                    concept_success, concept_message = analyze_pdf_concepts(pdf_document)
                    
                    print(f"📊 DEBUG: Concept analysis result: success={concept_success}, message='{concept_message}'")  # ✅ Debug print
                    
                    if concept_success:
                        concept_count = pdf_document.concept_units.count()
                        processing_stats['concept_units'] = concept_count
                        processing_stats['concept_analysis_success'] = True
                        processing_stats['concept_message'] = concept_message
                        print(f"✅ DEBUG: Concept analysis successful: {concept_count} units created")  # ✅ Debug print
                    else:
                        processing_stats['concept_message'] = f"Concept analysis failed: {concept_message}"
                        print(f"❌ DEBUG: Concept analysis failed: {concept_message}")  # ✅ Debug print
                        logger.warning(f"Concept analysis failed for PDF {pdf_document.id}: {concept_message}")
                except Exception as e:
                    error_msg = f"Concept analysis error: {str(e)}"
                    processing_stats['concept_message'] = error_msg
                    print(f"💥 DEBUG: Concept analysis exception: {error_msg}")  # ✅ Debug print
                    logger.error(f"Concept analysis error for PDF {pdf_document.id}: {str(e)}")
            else:
                print(f"⏭️ DEBUG: Skipping concept analysis (analyze_concepts=False)")  # ✅ Debug print
            
            return True, "", processing_stats
        
        except Exception as e:
            error_msg = f"Unexpected error during PDF processing: {str(e)}"
            logger.error(error_msg)
            pdf_document.processing_error = error_msg
            pdf_document.save()
            return False, error_msg, {}
    
    def create_text_chunks(self, pdf_document, chunks):
        """Create TextChunk objects from processed chunks"""
        # Delete existing chunks
        pdf_document.text_chunks.all().delete()
        
        # Create new chunks
        chunk_objects = []
        for order, (chunk_text, start_word, end_word) in enumerate(chunks):
            chunk_obj = TextChunk(
                pdf_document=pdf_document,
                chunk_text=chunk_text,
                chunk_order=order + 1,
                word_start=start_word,
                word_end=end_word
            )
            chunk_objects.append(chunk_obj)
        
        # Bulk create for efficiency
        TextChunk.objects.bulk_create(chunk_objects)

# Convenience function for easy import
def extract_pdf_text(pdf_document, analyze_concepts=True):
    """Extract and process text from a PDF document model instance"""
    extractor = PDFTextExtractor()
    success, error_msg, stats = extractor.extract_and_process_pdf(pdf_document, analyze_concepts)
    
    if success:
        return pdf_document.get_cleaned_text(), pdf_document.page_count, True, ""
    else:
        return "", 0, False, error_msg