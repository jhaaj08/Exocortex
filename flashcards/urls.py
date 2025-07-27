from django.urls import path
from . import views

app_name = 'flashcards'

urlpatterns = [
    path('', views.home, name='home'),
    path('add/', views.add_folder, name='add_folder'),
    path('folders/', views.folder_list, name='folder_list'),
    path('folder/<int:folder_id>/', views.folder_detail, name='folder_detail'),
    path('folder/<int:folder_id>/process/', views.process_folder, name='process_folder'),
    path('folder/<int:folder_id>/process/ajax/', views.process_folder_ajax, name='process_folder_ajax'),
    path('folder/<int:folder_id>/study/', views.study_flashcards, name='study_flashcards'),
    
    # Delete URLs
    path('folder/<int:folder_id>/delete/', views.delete_folder, name='delete_folder'),
    path('flashcard/<uuid:flashcard_id>/delete/', views.delete_flashcard, name='delete_flashcard'),
    path('folder/<int:folder_id>/bulk-delete/', views.bulk_delete_flashcards, name='bulk_delete_flashcards'),
    
    # PDF processing detail URLs
    path('pdf/<int:pdf_id>/details/', views.pdf_processing_details, name='pdf_processing_details'),
    path('chunk/<uuid:chunk_id>/', views.chunk_detail, name='chunk_detail'),
    
    # Focus block URLs
    path('pdf/<int:pdf_id>/focus-blocks/', views.pdf_focus_blocks, name='pdf_focus_blocks'),
    path('pdf/<int:pdf_id>/focus-blocks/generate/', views.generate_focus_blocks_view, name='generate_focus_blocks'),
    path('pdf/<int:pdf_id>/focus-blocks/study/', views.study_focus_blocks, name='study_focus_blocks'),
    path('pdf/<int:pdf_id>/focus-blocks/debug/', views.focus_block_debug, name='focus_block_debug'),
    path('focus-block/<uuid:block_id>/', views.focus_block_detail, name='focus_block_detail'),
    path('study-session/<uuid:session_id>/complete/<uuid:block_id>/', views.complete_focus_block, name='complete_focus_block'),
    
    # All focus blocks unified view
    path('focus-blocks/', views.all_focus_blocks, name='all_focus_blocks'),
    
    # Concept unit detail URL
    path('concept-unit/<uuid:unit_id>/', views.concept_unit_detail, name='concept_unit_detail'),
    
    # Bulk operations
    path('bulk-upload/', views.bulk_upload, name='bulk_upload'),
    path('bulk-manage/', views.bulk_manage, name='bulk_manage'),
    path('bulk-delete/', views.bulk_delete_pdfs, name='bulk_delete_pdfs'),
    
    # Deduplication stats
    path('deduplication-stats/', views.deduplication_stats, name='deduplication_stats'),
    
    # Focus Mode URLs
    path('focus/<uuid:focus_block_id>/start/', views.start_focus_mode, name='start_focus_mode'),
    path('focus/session/<uuid:session_id>/', views.focus_mode, name='focus_mode'),
    path('focus/<uuid:session_id>/update/', views.update_focus_progress, name='update_focus_progress'),
    path('focus/<uuid:session_id>/complete/', views.complete_focus_session, name='complete_focus_session'),
] 