from django.contrib import admin
from .models import Folder, Flashcard, StudySession, PDFDocument

@admin.register(Folder)
class FolderAdmin(admin.ModelAdmin):
    list_display = ['name', 'path', 'total_files', 'processed', 'created_at']
    list_filter = ['processed', 'created_at']
    search_fields = ['name', 'path']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(PDFDocument)
class PDFDocumentAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_size_display', 'page_count', 'processed', 'created_at']
    list_filter = ['processed', 'created_at']
    search_fields = ['name']
    readonly_fields = ['file_size', 'page_count', 'extracted_text', 'processing_error', 'created_at']
    
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
    list_display = ['session_id', 'source_name', 'current_page', 'cards_per_page', 'last_accessed']
    list_filter = ['created_at', 'last_accessed']
    readonly_fields = ['session_id', 'created_at', 'last_accessed']
    filter_horizontal = ['cards_shown']
    
    def source_name(self, obj):
        if obj.folder:
            return f"Folder: {obj.folder.name}"
        elif obj.pdf_document:
            return f"PDF: {obj.pdf_document.name}"
        return "Unknown"
    source_name.short_description = 'Source'
