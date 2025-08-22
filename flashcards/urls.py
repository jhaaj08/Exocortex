from django.urls import path
from . import views

app_name = 'flashcards'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
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
    
    # Focus Schedule Dashboard
    path('focus-schedule/', views.focus_schedule, name='focus_schedule'),
    
    # Focus Mode URLs (Timer-based sessions)
    path('focus/<uuid:focus_block_id>/start/', views.start_focus_mode, name='start_focus_mode'),
    path('focus/session/<uuid:session_id>/', views.focus_mode, name='focus_mode'),
    path('focus/<uuid:session_id>/update/', views.update_focus_progress, name='update_focus_progress'),
    path('focus/<uuid:session_id>/complete/', views.complete_focus_session, name='complete_focus_session'),
    path('api/focus-block/<uuid:focus_block_id>/complete/', views.complete_focus_block_api, name='complete_focus_block_api'),
    path('api/focus-completions/', views.get_focus_completions_api, name='get_focus_completions_api'),
    path('mark-segment-complete/', views.mark_segment_complete, name='mark_segment_complete'),
    path('api/session-progress/', views.get_session_progress, name='get_session_progress'),
    path('save-block-rating/', views.save_block_rating, name='save_block_rating'),
    
    # Focus Block Study URLs
    path('study/focus-blocks/', views.start_focus_study_session, name='start_focus_study_session'),
    path('study/focus-blocks/<uuid:block_id>/', views.focus_block_study, name='focus_block_study'),
    
    
    # Progress/Report URLs
    path('pdf/<int:pdf_id>/progress/', views.pdf_progress, name='pdf_progress'),
    
    # Debug URLs
    path('debug/focus-chunks/', views.debug_focus_chunks, name='debug_focus_chunks'),
    path('migrate/focus-blocks/', views.migrate_focus_blocks, name='migrate_focus_blocks'),
    path('study-planner/', views.generate_study_schedule_view, name='study_planner'),
    path('study-path/<str:path_type>/', views.start_study_path, name='start_study_path'),
    path('study-path/<uuid:block_id>/<str:session_key>/', views.focus_block_study_with_path, name='focus_block_study_with_path'),
    path('study-path/<uuid:block_id>/<str:session_key>/complete/', views.complete_study_path_block, name='complete_study_path_block'),
    path('debug-focus-data/', views.debug_focus_block_data, name='debug_focus_data'),
    path('study/advanced/<uuid:block_id>/', views.advanced_focus_study, name='advanced_focus_study'),
    path('session/<uuid:session_id>/complete/', views.complete_advanced_session, name='complete_advanced_session'),
    path('session/<uuid:session_id>/update/', views.update_session_progress, name='update_session_progress'),
    path('api/session/<uuid:session_id>/analytics/', views.session_analytics_api, name='session_analytics_api'),
    
    # Knowledge Graph URLs
    path('knowledge-graph/', views.knowledge_graph, name='knowledge_graph'),
    path('knowledge-graph/<uuid:pdf_id>/', views.knowledge_graph, name='knowledge_graph_pdf'),
    
    # For backward compatibility with old template references
    path('new-format-knowledge-graph/', views.knowledge_graph, name='new_format_knowledge_graph'),
    
    # API endpoints for graph data
    path('api/graph-data/<uuid:pdf_id>/', views.graph_data_api, name='graph_data_api'),
    path('api/study-paths/<uuid:pdf_id>/', views.study_paths_api, name='study_paths_api'),
    path('study/session/start/', views.start_study_session, name='start_study_session_api'),
    path('study/session/end/<uuid:session_id>/', views.end_study_session, name='end_study_session_api'),
    
    # Intelligent Plan URLs
    path('plan/preview/<str:plan_type>/', views.preview_intelligent_plan, name='preview_intelligent_plan'),
    path('plan/start/<str:plan_type>/', views.start_intelligent_plan, name='start_intelligent_plan'),
    path('resume/<str:session_key>/', views.resume_study_session, name='resume_study_session'),
    
    # Offline Study URLs
    path('offline/', views.offline_study_interface, name='offline_study'),
    path('export/study-pack/', views.export_study_pack, name='export_all_study_pack'),
    path('export/study-pack/<int:pdf_id>/', views.export_study_pack, name='export_pdf_study_pack'),
    path('api/study-pack/<uuid:block_id>/', views.study_pack_api, name='study_pack_api'),
    path('api/cache-blocks/', views.cache_blocks_for_offline, name='cache_blocks_offline'),
    
    # Offline Progress Sync URLs
    path('api/sync-progress/', views.sync_offline_progress, name='sync_offline_progress'),
    path('api/sync-status/', views.sync_status, name='sync_status'),
] 