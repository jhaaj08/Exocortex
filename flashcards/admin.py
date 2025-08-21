from django.contrib import admin
from .models import Folder, Flashcard, StudySession, PDFDocument, FocusBlock, FocusSession, FocusBlockRelationship, UserBlockState

@admin.register(Folder)
class FolderAdmin(admin.ModelAdmin):
    list_display = ['name', 'path', 'total_files', 'processed', 'created_at']
    list_filter = ['processed', 'created_at']
    search_fields = ['name', 'path']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(PDFDocument)
class PDFDocumentAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_size_display', 'page_count', 'processed']  # Remove created_at
    list_filter = ['processed']  # Remove created_at
    search_fields = ['name']
    readonly_fields = ['file_size', 'page_count', 'content_hash']  # Only existing fields
    
    def file_size_display(self, obj):
        return obj.get_file_size_display()
    file_size_display.short_description = 'File Size'

@admin.register(Flashcard)
class FlashcardAdmin(admin.ModelAdmin):
    list_display = ['question_preview', 'source_type', 'source_name', 'difficulty_level', 'created_at']
    list_filter = ['difficulty_level', 'created_at']
    search_fields = ['question', 'answer', 'tags']
    readonly_fields = ['id', 'created_at', 'updated_at']
    filter_horizontal = []
    
    def question_preview(self, obj):
        return obj.question[:100] + "..." if len(obj.question) > 100 else obj.question
    question_preview.short_description = 'Question'
    
    def source_type(self, obj):
        if obj.folder:
            return "Folder"
        elif obj.pdf_document:
            return "PDF"
        return "Unknown"
    source_type.short_description = 'Source Type'
    
    def source_name(self, obj):
        if obj.folder:
            return obj.folder.name
        elif obj.pdf_document:
            return obj.pdf_document.name
        return "N/A"
    source_name.short_description = 'Source'

@admin.register(StudySession)
class StudySessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'source_name', 'current_page', 'session_type', 'last_accessed']
    list_filter = ['session_type', 'created_at', 'last_accessed']
    readonly_fields = ['session_id', 'created_at', 'last_accessed']
    filter_horizontal = ['cards_shown', 'completed_focus_blocks']
    search_fields = ['folder__name', 'pdf_document__name']
    
    def source_name(self, obj):
        if obj.folder:
            return f"Folder: {obj.folder.name}"
        elif obj.pdf_document:
            return f"PDF: {obj.pdf_document.name}"
        return "Unknown"
    source_name.short_description = 'Source'

@admin.register(FocusBlock)
class FocusBlockAdmin(admin.ModelAdmin):
    list_display = ['title', 'pdf_document', 'block_order', 'target_duration_minutes']
    list_filter = ['pdf_document', 'difficulty_level']
    search_fields = ['title', 'pdf_document__name']
    readonly_fields = ['id', 'block_order']
    ordering = ['pdf_document__created_at', 'block_order']
    
    def target_duration_minutes(self, obj):
        return f"{obj.target_duration / 60:.1f} min"
    target_duration_minutes.short_description = 'Duration'

@admin.register(FocusSession)
class FocusSessionAdmin(admin.ModelAdmin):
    list_display = ['focus_block_title', 'proficiency_score', 'study_time_minutes', 'status', 'completed_at']
    list_filter = ['status', 'proficiency_score', 'completed_at']
    search_fields = ['focus_block__title', 'focus_block__pdf_document__name']
    readonly_fields = ['id', 'started_at', 'focus_block']
    ordering = ['-completed_at']
    
    def focus_block_title(self, obj):
        return obj.focus_block.title
    focus_block_title.short_description = 'Focus Block'
    
    def study_time_minutes(self, obj):
        if obj.total_study_time:
            return f"{obj.total_study_time / 60:.1f} min"
        return "N/A"
    study_time_minutes.short_description = 'Study Time'

@admin.register(FocusBlockRelationship)
class FocusBlockRelationshipAdmin(admin.ModelAdmin):
    list_display = ['from_block_title', 'relationship_type', 'to_block_title', 'confidence', 'edge_strength']
    list_filter = ['relationship_type', 'confidence', 'created_at']
    search_fields = ['from_block__title', 'to_block__title', 'description']
    ordering = ['-edge_strength', '-confidence']
    
    def from_block_title(self, obj):
        return obj.from_block.title
    from_block_title.short_description = 'From Block'
    
    def to_block_title(self, obj):
        return obj.to_block.title
    to_block_title.short_description = 'To Block'

@admin.register(UserBlockState)
class UserBlockStateAdmin(admin.ModelAdmin):
    list_display = ['user', 'block_title', 'status', 'avg_rating', 'review_count', 'next_due_display', 'days_until_due']
    list_filter = ['status', 'user', 'created_at', 'next_due_at']
    search_fields = ['user__username', 'block__title', 'block__pdf_document__name']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['next_due_at', '-updated_at']
    
    def block_title(self, obj):
        return obj.block.title
    block_title.short_description = 'Block'
    
    def next_due_display(self, obj):
        if obj.next_due_at:
            return obj.next_due_at.strftime('%Y-%m-%d %H:%M')
        return 'Not scheduled'
    next_due_display.short_description = 'Next Due'
    
    def days_until_due(self, obj):
        days = obj.days_until_due()
        if days > 0:
            return f"In {days} days"
        elif days < 0:
            return f"Overdue by {abs(days)} days"
        else:
            return "Due today"
    days_until_due.short_description = 'Due Status'
    
    # Bulk actions for scheduling
    def mark_as_suspended(self, request, queryset):
        queryset.update(status='suspended')
        self.message_user(request, f"Marked {queryset.count()} blocks as suspended.")
    mark_as_suspended.short_description = "Mark selected blocks as suspended"
    
    def reset_to_new(self, request, queryset):
        queryset.update(status='new', review_count=0, lapses=0, ease_factor=2.5, next_due_at=None)
        self.message_user(request, f"Reset {queryset.count()} blocks to new status.")
    reset_to_new.short_description = "Reset selected blocks to new"
    
    actions = [mark_as_suspended, reset_to_new]
