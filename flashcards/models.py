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
    file_size = models.IntegerField(default=0)
    page_count = models.IntegerField(default=0)
    word_count = models.IntegerField(default=0)
    processed = models.BooleanField(default=False)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    processing_error = models.TextField(blank=True)
    
    # Text fields (for now)
    extracted_text = models.TextField(blank=True)
    cleaned_text = models.TextField(blank=True)  # ✅ ADD BACK
    processing_duration = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    # ✅ ADD: Deduplication fields
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
    
    def create_content_fingerprint(self, length=1000):
        """Create shortened fingerprint for fuzzy matching"""
        if not self.cleaned_text:
            return ""
        
        # Create fingerprint from key sentences
        sentences = self.cleaned_text.split('.')[:50]  # First 50 sentences
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
        """Move to next uncompleted block"""
        from django.db.models import Q
        
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
            self.save()
            print("🎉 All blocks completed!")
            return False

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

