from django.db import models
from django.contrib.auth.models import User
import uuid
import os
import hashlib
from difflib import SequenceMatcher


class Folder(models.Model):
    """Model to store information about uploaded folders"""
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=500, help_text="Path to the folder containing markdown files")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    total_files = models.IntegerField(default=0)
    processed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.total_files} files)"

class PDFDocument(models.Model):
    """Model to store uploaded PDF documents"""
    name = models.CharField(max_length=255, blank=True)
    pdf_file = models.FileField(upload_to='pdfs/')
    
    # Basic fields
    content_hash = models.CharField(max_length=64, db_index=True, blank=True)
    content_fingerprint = models.TextField(blank=True, help_text="Content fingerprint for fuzzy matching")
    file_size = models.IntegerField(default=0)
    page_count = models.IntegerField(default=0)
    word_count = models.IntegerField(default=0)
    processed = models.BooleanField(default=False)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    processing_error = models.TextField(blank=True)
    
    # Text fields (for now)
    extracted_text = models.TextField(blank=True)
    # Note: cleaned_text removed - now generated on-demand via get_cleaned_text()
    processing_duration = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    
    # ✅ ADD: Advanced chunking data
    advanced_chunks = models.JSONField(
        default=dict, 
        blank=True, 
        help_text="Community-detected chunks with embeddings and metadata"
    )
    
    # Deduplication fields
    is_duplicate = models.BooleanField(default=False)
    duplicate_of = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='duplicates')
    similarity_score = models.FloatField(default=0.0)
    
    def get_file_size_display(self):
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def __str__(self):
        return self.name or "Untitled PDF"
    
    def get_extracted_text(self):
        """Read text from file when needed"""
        if self.extracted_text_file:
            return self.extracted_text_file.read().decode('utf-8')
        return ""
    
    def get_cleaned_text(self):
        """Generate cleaned text on-demand from extracted_text"""
        if not self.extracted_text:
            return ""
        
        # Import here to avoid circular imports
        from .text_processor import TextProcessor
        processor = TextProcessor()
        return processor.clean_text(self.extracted_text)
    
    def calculate_content_hash(self):
        """Calculate SHA-256 hash of extracted text for deduplication"""
        if not self.extracted_text:
            return ""
        
        import hashlib
        return hashlib.sha256(self.extracted_text.encode('utf-8')).hexdigest()
    
    def create_content_fingerprint(self, length=1000):
        """Create shortened fingerprint for fuzzy matching"""
        cleaned_text = self.get_cleaned_text()
        if not cleaned_text:
            return ""
        
        # Create fingerprint from key sentences
        sentences = cleaned_text.split('.')[:50]  # First 50 sentences
        fingerprint = '. '.join(sentences)[:length]
        return fingerprint
    
    def calculate_similarity(self, other_pdf):
        """Calculate similarity with another PDF using fingerprints"""
        if not self.content_fingerprint or not other_pdf.content_fingerprint:
            return 0.0
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, 
                                   self.content_fingerprint.lower(), 
                                   other_pdf.content_fingerprint.lower()).ratio()
        return similarity
    
    def find_similar_pdfs(self, threshold=0.8):
        """Find PDFs with similarity above threshold"""
        if not self.content_fingerprint:
            return []
        
        similar_pdfs = []
        existing_pdfs = PDFDocument.objects.filter(
            processed=True, 
            is_duplicate=False
        ).exclude(id=self.id)
        
        for pdf in existing_pdfs:
            similarity = self.calculate_similarity(pdf)
            if similarity >= threshold:
                similar_pdfs.append({
                    'pdf': pdf,
                    'similarity': similarity
                })
        
        # Sort by similarity (highest first)
        similar_pdfs.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_pdfs
    
    def delete(self, *args, **kwargs):
        """Delete the actual file when model is deleted"""
        if self.pdf_file:
            if os.path.isfile(self.pdf_file.path):
                os.remove(self.pdf_file.path)
        super().delete(*args, **kwargs)

class TextChunk(models.Model):
    """Model to store processed text chunks"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='text_chunks')
    chunk_text = models.TextField(help_text="The actual chunk content")
    chunk_order = models.IntegerField(help_text="Order of this chunk in the document")
    word_start = models.IntegerField(help_text="Starting word position in original text")
    word_end = models.IntegerField(help_text="Ending word position in original text")
    word_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['pdf_document', 'chunk_order']
        unique_together = ['pdf_document', 'chunk_order']
    
    def __str__(self):
        preview = self.chunk_text[:50] + "..." if len(self.chunk_text) > 50 else self.chunk_text
        return f"Chunk {self.chunk_order}: {preview}"
    
    def save(self, *args, **kwargs):
        # Auto-calculate word count
        if self.chunk_text:
            self.word_count = len(self.chunk_text.split())
        super().save(*args, **kwargs)

class ConceptUnit(models.Model):
    """Model to store concept units formed by grouping related chunks"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='concept_units')
    title = models.CharField(max_length=255, help_text="Auto-generated concept title")
    description = models.TextField(blank=True, help_text="Brief description of the concept")
    primary_labels = models.JSONField(default=list, help_text="Main concept labels for this unit")
    concept_order = models.IntegerField(help_text="Order of this concept in the document")
    word_count = models.IntegerField(default=0)
    
    # ✅ New reading time fields
    estimated_reading_time = models.FloatField(default=0.0, help_text="Estimated reading time in minutes (LLM calculated)")
    complexity_score = models.FloatField(default=0.5, help_text="Content complexity score (0-1)")
    cognitive_load = models.CharField(
        max_length=20,
        choices=[
            ('low', 'Low'),
            ('medium', 'Medium'), 
            ('high', 'High'),
            ('very_high', 'Very High'),
        ],
        default='medium',
        help_text="Cognitive load required"
    )
    time_optimized = models.BooleanField(default=False, help_text="Whether this unit has been time-optimized")
    original_order = models.IntegerField(null=True, blank=True, help_text="Original order before optimization")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['pdf_document', 'concept_order']
        unique_together = ['pdf_document', 'concept_order']
    
    def __str__(self):
        return f"Concept {self.concept_order}: {self.title} ({self.estimated_reading_time:.1f}min)"
    
    def get_combined_text(self):
        """Get the combined text from all chunks in this concept unit"""
        chunks = self.chunk_labels.select_related('text_chunk').order_by('text_chunk__chunk_order')
        return '\n\n'.join(chunk.text_chunk.chunk_text for chunk in chunks)
    
    def is_optimal_length(self):
        """Check if this concept unit is within optimal time range (5-10 minutes)"""
        return 5.0 <= self.estimated_reading_time <= 10.0
    
    def needs_splitting(self):
        """Check if this concept unit should be split (> 10 minutes)"""
        return self.estimated_reading_time > 10.0
    
    def can_be_combined(self, other_unit):
        """Check if this unit can be combined with another (total < 10 minutes)"""
        if not other_unit:
            return False
        total_time = self.estimated_reading_time + other_unit.estimated_reading_time
        return total_time < 10.0

class ChunkLabel(models.Model):
    """Model to store LLM-assigned labels for text chunks"""
    CONCEPT_LABELS = [
        ('Definition', 'Definition'),
        ('Intuition', 'Intuition'),
        ('Example', 'Example'),
        ('Procedure', 'Procedure'),
        ('Algorithm', 'Algorithm'),
        ('Derivation', 'Derivation'),
        ('Proof', 'Proof'),
        ('Statement', 'Statement'),
        ('Theorem', 'Theorem'),
        ('Lemma', 'Lemma'),
        ('Assumption', 'Assumption'),
        ('Property', 'Property'),
        ('Discussion', 'Discussion'),
        ('Contrast', 'Contrast'),
        ('Warning', 'Warning'),
        ('NewTopic', 'NewTopic'),
        ('Recap', 'Recap'),
        ('WrapUp', 'WrapUp'),
        ('DataSpec', 'DataSpec'),
        ('Code', 'Code'),
        ('Pseudocode', 'Pseudocode'),
        ('FigureCaption', 'FigureCaption'),
        ('Exercise', 'Exercise'),
        ('ProblemStatement', 'ProblemStatement'),
        ('Introduction', 'Introduction'),
        ('Conclusion', 'Conclusion'),
        ('Summary', 'Summary'),
        ('Background', 'Background'),
        ('Methodology', 'Methodology'),
        ('Results', 'Results'),
        ('Analysis', 'Analysis'),
        ('Interpretation', 'Interpretation'),
        ('Application', 'Application'),
        ('Limitation', 'Limitation'),
        ('FutureWork', 'FutureWork'),
        ('Related', 'Related'),
        ('Comparison', 'Comparison'),
        ('Classification', 'Classification'),
        ('Explanation', 'Explanation'),
        ('Elaboration', 'Elaboration'),
        ('Other', 'Other'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    text_chunk = models.OneToOneField(TextChunk, on_delete=models.CASCADE, related_name='chunk_label')
    concept_unit = models.ForeignKey(ConceptUnit, on_delete=models.CASCADE, related_name='chunk_labels', null=True, blank=True)
    primary_label = models.CharField(max_length=50, choices=CONCEPT_LABELS, help_text="Primary concept label")
    secondary_labels = models.JSONField(default=list, help_text="Additional relevant labels")
    confidence_score = models.FloatField(default=0.0, help_text="LLM confidence in labeling (0-1)")
    reasoning = models.TextField(blank=True, help_text="LLM explanation for the labeling")
    concept_keywords = models.JSONField(default=list, help_text="Key concepts extracted from this chunk")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['text_chunk__chunk_order']
    
    def __str__(self):
        return f"{self.text_chunk} → {self.primary_label}"

class Flashcard(models.Model):
    """Model to store individual flashcards"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    text_chunk = models.ForeignKey(TextChunk, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    concept_unit = models.ForeignKey(ConceptUnit, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    question = models.TextField()
    answer = models.TextField()
    difficulty_level = models.CharField(
        max_length=20,
        choices=[
            ('beginner', 'Beginner'),
            ('intermediate', 'Intermediate'),
            ('advanced', 'Advanced'),
        ],
        default='intermediate'
    )
    source_file = models.CharField(max_length=255, blank=True, help_text="Original source file name")
    tags = models.JSONField(default=list, help_text="List of relevant tags")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        source = self.folder.name if self.folder else (self.pdf_document.name if self.pdf_document else "Unknown")
        return f"Q: {self.question[:50]}... (from {source})"

class FocusBlock(models.Model):
    """Model for 6-7 minute focused study sessions using Compact7 template"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='focus_blocks')
    main_concept_unit = models.ForeignKey(ConceptUnit, on_delete=models.CASCADE, related_name='main_focus_blocks')
    revision_concept_unit = models.ForeignKey(ConceptUnit, on_delete=models.SET_NULL, null=True, blank=True, related_name='revision_focus_blocks')
    
    # Focus block metadata
    block_order = models.IntegerField(help_text="Order of this focus block in the study sequence")
    title = models.CharField(max_length=500, help_text="Focus block title")  # ✅ Increased from 255 to 500
    target_duration = models.FloatField(default=420, help_text="Target duration in seconds (7 min = 420 sec)")
    
    # ✅ NEW: Store complete Compact7 JSON structure
    compact7_data = models.JSONField(help_text="Complete Compact7 template JSON structure")
    
    # Legacy fields for backward compatibility
    teacher_script = models.TextField(blank=True, help_text="Legacy teaching script")
    rescue_reset = models.TextField(blank=True, help_text="Legacy rescue content")
    recap_summary = models.TextField(blank=True, help_text="Legacy recap")
    
    # Metadata
    difficulty_level = models.CharField(
        max_length=20,
        choices=[
            ('beginner', 'Beginner'),
            ('intermediate', 'Intermediate'),
            ('advanced', 'Advanced'),
        ],
        default='intermediate'
    )
    learning_objectives = models.JSONField(default=list, help_text="List of learning objectives")
    prerequisite_concepts = models.JSONField(default=list, help_text="Prerequisite concepts")
    
    # ✅ ADD: Embedding for deduplication
    content_embedding = models.JSONField(
        default=list, 
        blank=True, 
        help_text="Semantic embedding vector for deduplication (1536 dimensions)"
    )
    content_hash = models.CharField(
        max_length=64, 
        blank=True, 
        db_index=True, 
        help_text="Hash of content for quick duplicate detection"
    )
    is_merged = models.BooleanField(
        default=False, 
        help_text="True if this block was created by merging duplicates"
    )
    merged_from_blocks = models.JSONField(
        default=list, 
        help_text="List of block IDs that were merged into this one"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['pdf_document', 'block_order']
        unique_together = ['pdf_document', 'block_order']
    
    def __str__(self):
        return f"Focus Block {self.block_order}: {self.title}"
    
    def get_estimated_duration_display(self):
        """Get human-readable duration"""
        if hasattr(self, 'compact7_data') and self.compact7_data:
            # Calculate total time from segments (updated for new structure)
            total_seconds = 0
            
            # Try direct access first (new format)
            if 'segments' in self.compact7_data:
                segments = self.compact7_data['segments']
            # Fallback to nested core format (old format)  
            elif 'core' in self.compact7_data:
                segments = self.compact7_data['core'].get('segments', [])
            else:
                segments = []
                
            for segment in segments:
                total_seconds += segment.get('time_sec', 0)
            
            # Also check for total_duration field
            if total_seconds == 0 and 'total_duration' in self.compact7_data:
                total_seconds = self.compact7_data['total_duration']
            
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            # Fallback to target_duration
            minutes = int(self.target_duration // 60)
            seconds = int(self.target_duration % 60)
            return f"{minutes}m {seconds}s"
    
    def get_core_goal(self):
        """Get the core learning goal"""
        if self.compact7_data and 'core' in self.compact7_data:
            return self.compact7_data['core'].get('goal', '')
        return ''
    
    def get_segments(self):
        """Get teaching segments"""
        if self.compact7_data:
            # Try direct access first (new format)
            if 'segments' in self.compact7_data:
                return self.compact7_data['segments']
            # Fallback to nested core format (old format)
            elif 'core' in self.compact7_data:
                return self.compact7_data['core'].get('segments', [])
        return []
    
    def get_qa_items(self):
        """Get Q&A items"""
        if self.compact7_data:
            # Try both field names (qa_items is the new format)
            if 'qa_items' in self.compact7_data:
                return self.compact7_data['qa_items']
            elif 'qa' in self.compact7_data:
                return self.compact7_data['qa']
            elif 'core' in self.compact7_data:
                return self.compact7_data['core'].get('qa', [])
        return []
    
    def get_revision_data(self):
        """Get revision data"""
        if self.compact7_data:
            return self.compact7_data.get('revision', {})
        return {}
    
    def get_recap_data(self):
        """Get recap data"""
        if self.compact7_data:
            return self.compact7_data.get('recap', {})
        return {}
    
    def get_rescue_data(self):
        """Get rescue/reset data"""
        if self.compact7_data:
            return self.compact7_data.get('rescue', {})
        return {}

class StudySession(models.Model):
    """Enhanced model to track study sessions with focus blocks"""
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, related_name='study_sessions', null=True, blank=True)
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='study_sessions', null=True, blank=True)
    
    # Focus block tracking
    current_focus_block = models.ForeignKey(FocusBlock, on_delete=models.SET_NULL, null=True, blank=True, related_name='current_sessions')
    completed_focus_blocks = models.ManyToManyField(FocusBlock, blank=True, related_name='completed_sessions')

    # ✅ ADD THESE NEW FIELDS:
    current_segment = models.IntegerField(default=0, help_text="Current segment index in current block")
    segment_progress = models.JSONField(default=dict, help_text="Completed segments per block: {block_id: [0,1,2,...]}")
    
    # Block completion tracking
    block_completion_data = models.JSONField(
        default=dict,
        help_text="Block completion data: {block_id: {rating: 1-5, time_spent: seconds, notes: 'text', timestamp: 'ISO'}}"
    )
    
    # Legacy flashcard tracking
    cards_shown = models.ManyToManyField(Flashcard, blank=True)
    current_page = models.IntegerField(default=1)
    cards_per_page = models.IntegerField(default=10)
    
    # Session metadata
    session_type = models.CharField(
        max_length=20,
        choices=[
            ('flashcards', 'Flashcards'),
            ('focus_blocks', 'Focus Blocks'),
            ('mixed', 'Mixed'),
        ],
        default='focus_blocks'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    
    # Phase 1: session lifecycle + planned playlist
    status = models.CharField(
        max_length=20,
        choices=[('active','Active'), ('completed','Completed'), ('abandoned','Abandoned')],
        default='active',
    )
    ended_at = models.DateTimeField(null=True, blank=True)
    
    planned_blocks = models.JSONField(default=list, help_text="Ordered list of planned FocusBlock ids as strings")
    plan_position = models.IntegerField(default=0, help_text="Index into planned_blocks for current position")
    
    target_duration_min = models.IntegerField(null=True, blank=True)
    mix_ratio_review = models.FloatField(default=0.3)
    
    def __str__(self):
        source = self.folder.name if self.folder else (self.pdf_document.name if self.pdf_document else "Unknown")
        return f"Study session for {source} ({self.session_type})"
    
    def get_progress_percentage(self):
        """Calculate progress through focus blocks"""
        if not self.pdf_document:
            return 0
        
        total_blocks = self.pdf_document.focus_blocks.count()
        completed_blocks = self.completed_focus_blocks.count()
        
        if total_blocks == 0:
            return 0
        
        return round((completed_blocks / total_blocks) * 100, 1)

    def get_current_block_progress(self):
        """Get progress within current block"""
        if not self.current_focus_block:
            return 0
        
        block_id = str(self.current_focus_block.id)
        completed_segments = self.segment_progress.get(block_id, [])
        total_segments = len(self.current_focus_block.get_segments())
        
        if total_segments == 0:
            return 0
        return (len(completed_segments) / total_segments) * 100

    def mark_segment_completed(self, segment_index):
        """Mark a segment as completed in current block"""
        if not self.current_focus_block:
            return False
        
        block_id = str(self.current_focus_block.id)
        
        # Initialize segment progress for this block if needed
        if block_id not in self.segment_progress:
            self.segment_progress[block_id] = []
        
        # Add segment if not already completed
        if segment_index not in self.segment_progress[block_id]:
            self.segment_progress[block_id].append(segment_index)
            self.segment_progress[block_id].sort()  # Keep sorted
        
        # Update current segment
        self.current_segment = segment_index + 1
        
        # Check if block is complete
        total_segments = len(self.current_focus_block.get_segments())
        completed_segments = len(self.segment_progress[block_id])
        
        print(f"🔍 Segment {segment_index} completed. Progress: {completed_segments}/{total_segments}")
        
        if completed_segments >= total_segments:
            # Block completed - add to completed blocks
            self.completed_focus_blocks.add(self.current_focus_block)
            print(f"🎉 Block '{self.current_focus_block.title}' completed!")
            
            # Advance to next block
            return self.advance_to_next_block()
        
        self.save()
        return True

    def advance_to_next_block(self):
        """Move to next uncompleted block (prefer planned playlist if present)"""
        from django.utils import timezone
        # Use planned playlist if present
        if self.planned_blocks:
            return self.advance_by_plan()

        all_blocks = FocusBlock.objects.all().order_by('created_at', 'block_order')
        completed_ids = self.completed_focus_blocks.values_list('id', flat=True)
        uncompleted_blocks = all_blocks.exclude(id__in=completed_ids)

        if uncompleted_blocks.exists():
            next_block = uncompleted_blocks.first()
            self.current_focus_block = next_block
            self.current_segment = 0
            self.save()
            print(f"🔄 Advanced to next block: {next_block.title}")
            return True
        else:
            # All blocks completed
            self.current_focus_block = None
            self.status = 'completed'
            self.ended_at = timezone.now()
            self.save()
            print("🎉 All blocks completed!")
            return False
    
    def save_block_completion(self, block_id, proficiency_rating, time_spent_seconds, notes=""):
        """Save block completion data including rating, time, and notes"""
        from datetime import datetime
        
        block_id = str(block_id)
        
        # Initialize block_completion_data if needed
        if not isinstance(self.block_completion_data, dict):
            self.block_completion_data = {}
        
        # Save completion data
        self.block_completion_data[block_id] = {
            'proficiency_rating': proficiency_rating,
            'time_spent_seconds': time_spent_seconds,
            'notes': notes,
            'timestamp': datetime.now().isoformat(),
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.save()
        
        print(f"✅ Saved block completion: Block {block_id}, Rating: {proficiency_rating}/5, Time: {time_spent_seconds}s")
        return True
    
    # Phase 1: planning helpers
    def set_plan(self, block_ids):
        """Set the planned playlist and position; ensure current block is first uncompleted in plan."""
        self.planned_blocks = [str(bid) for bid in block_ids]
        self.plan_position = 0
        # set current block to first planned that's not completed
        for bid in self.planned_blocks:
            try:
                fb = FocusBlock.objects.get(id=bid)
            except FocusBlock.DoesNotExist:
                continue
            if fb not in self.completed_focus_blocks.all():
                self.current_focus_block = fb
                break
        self.status = 'active'
        self.save()

    def advance_by_plan(self):
        """Advance using planned playlist; complete session when plan is exhausted."""
        from django.utils import timezone
        if not self.planned_blocks:
            return self.advance_to_next_block()

        n = len(self.planned_blocks)
        pos = self.plan_position + 1
        while pos < n:
            try:
                fb = FocusBlock.objects.get(id=self.planned_blocks[pos])
            except FocusBlock.DoesNotExist:
                pos += 1
                continue
            if fb not in self.completed_focus_blocks.all():
                self.plan_position = pos
                self.current_focus_block = fb
                self.current_segment = 0
                self.save()
                print(f"🔄 Advanced by plan to: {fb.title}")
                return True
            pos += 1

        # Plan exhausted
        self.current_focus_block = None
        self.status = 'completed'
        self.ended_at = timezone.now()
        self.save()
        print("🎉 Planned playlist completed!")
        return False
    
    @classmethod
    def create_intelligent_plan(cls, user, target_duration_min=60, review_ratio=0.3):
        """Create an intelligent study plan following the master sequence"""
        from django.contrib.auth.models import User
        
        # For now, use first admin user if user is None
        if user is None:
            user = User.objects.filter(is_superuser=True).first()
            if not user:
                print("⚠️ No admin user found, using simple plan")
                return cls._create_simple_plan(target_duration_min)
        
        print(f"🧠 Creating intelligent plan for {user.username}")
        print(f"📊 Target: {target_duration_min}min, Review ratio: {review_ratio:.0%}")
        
        # Get the MASTER SEQUENCE (same as Focus Blocks page)
        all_blocks = FocusBlock.objects.select_related('pdf_document').order_by(
            'pdf_document__created_at', 'block_order'
        )
        master_sequence = cls._order_by_prerequisites(list(all_blocks))
        
        # Get completion status
        completed_block_ids = set()
        if user:
            # Get completed blocks from UserBlockState
            user_completed = UserBlockState.objects.filter(
                user=user,
                status='completed'
            ).values_list('block_id', flat=True)
            completed_block_ids.update(str(bid) for bid in user_completed)
            
            # Also get completed blocks from StudySessions
            session_completed = StudySession.objects.values_list('completed_focus_blocks', flat=True).distinct()
            completed_block_ids.update(str(bid) for bid in session_completed if bid)
        
        # Calculate target counts
        avg_block_duration = 7  # minutes
        total_blocks = max(1, target_duration_min // avg_block_duration)
        
        # Select blocks sequentially from master sequence
        selected_blocks = []
        review_count = 0
        new_count = 0
        target_review_count = int(total_blocks * review_ratio)
        target_new_count = total_blocks - target_review_count
        
        print(f"📋 Master sequence has {len(master_sequence)} blocks")
        print(f"🎯 Target: {target_review_count} review + {target_new_count} new = {total_blocks} total")
        
        for block in master_sequence:
            if len(selected_blocks) >= total_blocks:
                break
                
            block_id = str(block.id)
            is_completed = block_id in completed_block_ids
            
            if is_completed and review_count < target_review_count:
                # This is a review block
                selected_blocks.append(block)
                review_count += 1
                print(f"📖 Added review block {len(selected_blocks)}: {block.title[:50]}")
            elif not is_completed and new_count < target_new_count:
                # This is a new block
                selected_blocks.append(block)
                new_count += 1
                print(f"🆕 Added new block {len(selected_blocks)}: {block.title[:50]}")
        
        # If we don't have enough blocks, fill with available blocks from sequence
        if len(selected_blocks) < total_blocks:
            remaining_needed = total_blocks - len(selected_blocks)
            print(f"🔄 Need {remaining_needed} more blocks, adding from sequence...")
            
            for block in master_sequence:
                if len(selected_blocks) >= total_blocks:
                    break
                if block not in selected_blocks:
                    selected_blocks.append(block)
                    print(f"➕ Added additional block {len(selected_blocks)}: {block.title[:50]}")
        
        print(f"✅ Plan created: {review_count} review + {new_count} new = {len(selected_blocks)} total")
        print(f"📋 Sequential order maintained from master sequence")
        
        return [str(block.id) for block in selected_blocks]
    
    @classmethod
    def _create_simple_plan(cls, target_duration_min=60):
        """Fallback: simple plan following master sequence"""
        print("📝 Creating simple plan (following master sequence)")
        
        # Get the same master sequence as other plans
        all_blocks = FocusBlock.objects.select_related('pdf_document').order_by(
            'pdf_document__created_at', 'block_order'
        )
        master_sequence = cls._order_by_prerequisites(list(all_blocks))
        
        # Get completed blocks
        all_completed_ids = StudySession.objects.values_list('completed_focus_blocks', flat=True).distinct()
        completed_ids = set(str(bid) for bid in all_completed_ids if bid)
        
        # Select uncompleted blocks from master sequence
        avg_block_duration = 7  # minutes
        max_blocks = target_duration_min // avg_block_duration
        
        selected_blocks = []
        for block in master_sequence:
            if len(selected_blocks) >= max_blocks:
                break
            if str(block.id) not in completed_ids:
                selected_blocks.append(block)
                print(f"📋 Added block {len(selected_blocks)}: {block.title[:50]}")
        
        print(f"📊 Simple plan: {len(selected_blocks)} blocks (max {max_blocks}) following master sequence")
        
        return [str(block.id) for block in selected_blocks]
    
    @classmethod
    def _order_by_prerequisites(cls, blocks):
        """Order blocks based on prerequisite relationships from knowledge graph"""
        try:
            # Try to use knowledge graph for ordering
            from django.db.models import Q
            
            # Get all prerequisite relationships
            prereq_relationships = FocusBlockRelationship.objects.filter(
                relationship_type='prerequisite',
                from_block__in=blocks,
                to_block__in=blocks
            ).select_related('from_block', 'to_block')
            
            if not prereq_relationships.exists():
                # No relationships found, return original order
                print("📋 No prerequisites found, using default order")
                return blocks
            
            # Build dependency graph
            dependencies = {}
            for block in blocks:
                dependencies[block.id] = []
            
            for rel in prereq_relationships:
                # to_block depends on from_block
                if rel.to_block.id in dependencies:
                    dependencies[rel.to_block.id].append(rel.from_block.id)
            
            # Topological sort
            ordered_ids = cls._topological_sort(dependencies)
            
            # Convert back to blocks in sorted order
            block_dict = {block.id: block for block in blocks}
            ordered_blocks = []
            
            for block_id in ordered_ids:
                if block_id in block_dict:
                    ordered_blocks.append(block_dict[block_id])
            
            # Add any remaining blocks that weren't in the dependency graph
            remaining_blocks = [b for b in blocks if b not in ordered_blocks]
            ordered_blocks.extend(remaining_blocks)
            
            print(f"🔗 Applied prerequisite ordering: {len(ordered_blocks)} blocks")
            return ordered_blocks
            
        except Exception as e:
            print(f"⚠️ Error ordering by prerequisites: {e}")
            return blocks
    
    @classmethod
    def _topological_sort(cls, dependencies):
        """Simple topological sort for prerequisite ordering"""
        # Kahn's algorithm
        in_degree = {}
        for node in dependencies:
            in_degree[node] = len(dependencies[node])
        
        # Find nodes with no dependencies
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Find nodes that depend on this node
            for other_node, deps in dependencies.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        return result

class FocusSession(models.Model):
    """Track individual focus block study sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    focus_block = models.ForeignKey(FocusBlock, on_delete=models.CASCADE, related_name='study_sessions')
    
    # Session tracking
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Progress tracking
    current_segment = models.IntegerField(default=0, help_text="Current segment index")
    segments_completed = models.JSONField(default=list, help_text="List of completed segment indices")
    
    # Timing data
    total_study_time = models.FloatField(null=True, blank=True, help_text="Actual time spent in seconds")
    segment_times = models.JSONField(default=dict, help_text="Time spent on each segment")
    
    # Learning assessment
    proficiency_score = models.IntegerField(
        null=True, blank=True,
        choices=[(i, f"{i}/5") for i in range(1, 6)],
        help_text="Self-assessed proficiency score (1-5)"
    )
    difficulty_rating = models.IntegerField(
        null=True, blank=True,
        choices=[(i, f"{i}/5") for i in range(1, 6)],
        help_text="How difficult was this content? (1=easy, 5=hard)"
    )
    
    # Session status
    status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('paused', 'Paused'),
            ('completed', 'Completed'),
            ('abandoned', 'Abandoned')
        ],
        default='active'
    )
    
    # Notes and feedback
    learning_notes = models.TextField(blank=True, help_text="Student's notes during session")
    confusion_points = models.JSONField(default=list, help_text="Segments where student got confused")
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Session: {self.focus_block.title} ({self.status})"
    
    def get_completion_percentage(self):
        """Get completion percentage"""
        if not self.focus_block:
            return 0
        total_segments = len(self.focus_block.get_segments())
        if total_segments == 0:
            return 0
        return (len(self.segments_completed) / total_segments) * 100
    
    def mark_segment_completed(self, segment_index, time_spent):
        """Mark a segment as completed"""
        if segment_index not in self.segments_completed:
            self.segments_completed.append(segment_index)
        self.segment_times[str(segment_index)] = time_spent
        self.current_segment = segment_index + 1
        self.save()


class FocusBlockRelationship(models.Model):
    """Model to store relationships between focus blocks for knowledge graph"""
    RELATIONSHIP_TYPES = [
        ('prerequisite', 'Prerequisite'),        # A must be learned before B
        ('builds_on', 'Builds On'),             # B extends concepts from A
        ('related', 'Related'),                 # Related but independent concepts
        ('applies_to', 'Applies To'),           # A provides theory, B shows applications
        ('compares_with', 'Compares With'),     # Blocks compare/contrast approaches
        ('specializes', 'Specializes'),         # B is a specific case of A
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_block = models.ForeignKey(FocusBlock, on_delete=models.CASCADE, related_name='outgoing_relationships')
    to_block = models.ForeignKey(FocusBlock, on_delete=models.CASCADE, related_name='incoming_relationships')
    
    relationship_type = models.CharField(max_length=20, choices=RELATIONSHIP_TYPES)
    confidence = models.FloatField(help_text="AI confidence in this relationship (0-1)")
    similarity_score = models.FloatField(help_text="Semantic similarity score (0-1)")
    edge_strength = models.FloatField(help_text="Combined relationship strength (0-1)")
    
    description = models.TextField(help_text="AI-generated description of the relationship")
    educational_reasoning = models.TextField(help_text="Why this relationship helps learning")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['from_block', 'to_block', 'relationship_type']
        ordering = ['-edge_strength', '-confidence']
    
    def __str__(self):
        return f"{self.from_block.title} → {self.to_block.title} ({self.relationship_type})"
    
    def is_bidirectional(self):
        """Check if this is a bidirectional relationship (like 'related')"""
        return self.relationship_type in ['related', 'compares_with']
    
    def get_reverse_relationship(self):
        """Get the reverse relationship if it exists"""
        return FocusBlockRelationship.objects.filter(
            from_block=self.to_block,
            to_block=self.from_block,
            relationship_type=self.relationship_type
        ).first() 


class UserBlockState(models.Model):
    """Phase 2: Track per-user scheduling state for each focus block (spaced repetition)"""
    
    STATUS_CHOICES = [
        ('new', 'New'),
        ('learning', 'Learning'),
        ('review', 'Review'),
        ('lapsed', 'Lapsed'),
        ('suspended', 'Suspended'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='block_states')
    block = models.ForeignKey(FocusBlock, on_delete=models.CASCADE, related_name='user_states')
    
    # Scheduling state
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    last_reviewed_at = models.DateTimeField(null=True, blank=True)
    next_due_at = models.DateTimeField(null=True, blank=True)
    
    # Spaced repetition parameters
    review_count = models.IntegerField(default=0, help_text="Number of times reviewed")
    lapses = models.IntegerField(default=0, help_text="Number of times marked as difficult/forgotten")
    ease_factor = models.FloatField(default=2.5, help_text="SM-2 ease factor (1.3-4.0)")
    stability = models.FloatField(default=1.0, help_text="Memory stability in days")
    
    # Performance tracking
    avg_rating = models.FloatField(null=True, blank=True, help_text="Average proficiency rating (1-5)")
    recent_qa_score = models.FloatField(null=True, blank=True, help_text="Recent Q&A accuracy (0-1)")
    total_time_spent = models.IntegerField(default=0, help_text="Total study time in seconds")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'block']
        ordering = ['next_due_at', '-updated_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['next_due_at']),
            models.Index(fields=['status', 'next_due_at']),
        ]
    
    def __str__(self):
        return f"{self.user.username} → {self.block.title} ({self.status})"
    
    def is_due(self):
        """Check if this block is due for review"""
        if not self.next_due_at:
            return self.status == 'new'
        from django.utils import timezone
        return timezone.now() >= self.next_due_at
    
    def days_until_due(self):
        """Get days until due (negative if overdue)"""
        if not self.next_due_at:
            return 0
        from django.utils import timezone
        delta = self.next_due_at - timezone.now()
        return delta.days
    
    def update_from_rating(self, rating, time_spent_seconds=0, qa_correct=0, qa_total=0):
        """Update scheduling parameters based on new rating (1-5)"""
        from django.utils import timezone
        from datetime import timedelta
        
        self.last_reviewed_at = timezone.now()
        self.review_count += 1
        self.total_time_spent += time_spent_seconds
        
        # Update average rating
        if self.avg_rating is None:
            self.avg_rating = rating
        else:
            # Weighted average: recent ratings have more influence
            self.avg_rating = (self.avg_rating * 0.7) + (rating * 0.3)
        
        # Update Q&A score if provided
        if qa_total > 0:
            qa_score = qa_correct / qa_total
            if self.recent_qa_score is None:
                self.recent_qa_score = qa_score
            else:
                self.recent_qa_score = (self.recent_qa_score * 0.6) + (qa_score * 0.4)
        
        # SM-2 algorithm adaptation
        if rating >= 3:
            # Successful recall
            if self.status == 'new':
                self.status = 'learning'
                self.next_due_at = timezone.now() + timedelta(days=1)
            elif self.status == 'learning':
                self.status = 'review'
                self.next_due_at = timezone.now() + timedelta(days=6)
            else:
                # Review mode: use ease factor
                interval_days = max(1, int(self.stability * self.ease_factor))
                self.next_due_at = timezone.now() + timedelta(days=interval_days)
                self.stability = interval_days
            
            # Adjust ease factor based on rating
            if rating == 5:
                self.ease_factor = min(4.0, self.ease_factor + 0.15)
            elif rating == 4:
                self.ease_factor = min(4.0, self.ease_factor + 0.1)
            # rating == 3: no change
        else:
            # Poor recall (rating 1-2): lapse
            self.lapses += 1
            self.status = 'lapsed'
            self.ease_factor = max(1.3, self.ease_factor - 0.2)
            self.stability = max(1, self.stability * 0.8)
            # Short retry interval
            self.next_due_at = timezone.now() + timedelta(hours=6)
        
        self.save()
        print(f"📅 Updated {self.user.username} → {self.block.title}: {self.status}, due in {self.days_until_due()} days")
    
    @classmethod
    def get_due_blocks_for_user(cls, user, include_overdue=True):
        """Get blocks that are due for review for a specific user"""
        from django.utils import timezone
        
        queryset = cls.objects.filter(user=user)
        
        if include_overdue:
            # Include new blocks (no due date) and overdue blocks
            queryset = queryset.filter(
                models.Q(next_due_at__isnull=True, status='new') |
                models.Q(next_due_at__lte=timezone.now())
            )
        else:
            # Only blocks due today (not overdue)
            today = timezone.now().date()
            queryset = queryset.filter(
                models.Q(next_due_at__isnull=True, status='new') |
                models.Q(next_due_at__date=today)
            )
        
        return queryset.select_related('block').order_by('next_due_at', 'status')
    
    @classmethod
    def get_new_blocks_for_user(cls, user, exclude_completed=True):
        """Get new blocks that user hasn't studied yet"""
        studied_block_ids = cls.objects.filter(user=user).values_list('block_id', flat=True)
        
        queryset = FocusBlock.objects.exclude(id__in=studied_block_ids)
        
        if exclude_completed:
            # Also exclude blocks completed in any study session
            from django.db.models import Q
            completed_in_sessions = StudySession.objects.filter(
                completed_focus_blocks__isnull=False
            ).values_list('completed_focus_blocks', flat=True)
            queryset = queryset.exclude(id__in=completed_in_sessions)
        
        return queryset.order_by('created_at', 'block_order')
    
    @classmethod
    def get_struggling_blocks_for_user(cls, user, threshold_rating=2.5):
        """Get blocks where user is struggling (low avg rating or recent lapses)"""
        return cls.objects.filter(
            user=user
        ).filter(
            models.Q(avg_rating__lt=threshold_rating) |
            models.Q(status='lapsed') |
            models.Q(lapses__gte=2)
        ).select_related('block').order_by('avg_rating', '-lapses') 

