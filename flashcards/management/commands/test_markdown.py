from django.core.management.base import BaseCommand
from flashcards.models import Folder
from flashcards.services import process_folder_content
import json

class Command(BaseCommand):
    help = 'Test markdown processing for a folder'

    def add_arguments(self, parser):
        parser.add_argument('folder_id', type=int, help='Folder ID to process')
        parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    def handle(self, *args, **options):
        folder_id = options['folder_id']
        verbose = options['verbose']
        
        try:
            folder = Folder.objects.get(id=folder_id)
        except Folder.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'Folder with ID {folder_id} does not exist')
            )
            return
        
        self.stdout.write(f'Processing folder: {folder.name}')
        self.stdout.write(f'Path: {folder.path}')
        self.stdout.write('-' * 50)
        
        # Process the folder
        extracted_content = process_folder_content(folder)
        
        if not extracted_content:
            self.stdout.write(
                self.style.WARNING('No content extracted from markdown files')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Extracted {len(extracted_content)} potential flashcards')
        )
        
        # Display results
        for i, content in enumerate(extracted_content, 1):
            self.stdout.write(f'\n--- Flashcard {i} ---')
            self.stdout.write(f'Source: {content["source_file"]}')
            self.stdout.write(f'Type: {content["content_type"]}')
            self.stdout.write(f'Context: {content["context"]}')
            
            if verbose:
                self.stdout.write(f'Question: {content["question"]}')
                self.stdout.write(f'Answer: {content["answer"][:200]}{"..." if len(content["answer"]) > 200 else ""}')
                self.stdout.write(f'Original Content: {content["original_content"]}')
            else:
                self.stdout.write(f'Question: {content["question"][:80]}{"..." if len(content["question"]) > 80 else ""}')
        
        # Summary by content type
        content_types = {}
        for content in extracted_content:
            content_type = content['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        self.stdout.write('\n--- Summary by Content Type ---')
        for content_type, count in content_types.items():
            self.stdout.write(f'{content_type}: {count} flashcards')
        
        # Save to JSON file for inspection
        output_file = f'markdown_test_output_{folder_id}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_content, f, indent=2, ensure_ascii=False)
        
        self.stdout.write(f'\nDetailed output saved to: {output_file}') 