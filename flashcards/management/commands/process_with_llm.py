from django.core.management.base import BaseCommand
from flashcards.models import Folder
from flashcards.services import process_folder_content
from flashcards.llm_service import process_folder_with_llm

class Command(BaseCommand):
    help = 'Process a folder with full pipeline: markdown extraction + LLM enhancement'

    def add_arguments(self, parser):
        parser.add_argument('folder_id', type=int, help='Folder ID to process')
        parser.add_argument('--dry-run', action='store_true', help='Extract content but don\'t save to database')

    def handle(self, *args, **options):
        folder_id = options['folder_id']
        dry_run = options['dry_run']
        
        try:
            folder = Folder.objects.get(id=folder_id)
        except Folder.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'Folder with ID {folder_id} does not exist')
            )
            return
        
        if folder.processed and not dry_run:
            self.stdout.write(
                self.style.WARNING(f'Folder "{folder.name}" is already processed. Use --dry-run to test again.')
            )
            return
        
        self.stdout.write(f'🔄 Processing folder: {folder.name}')
        self.stdout.write(f'📁 Path: {folder.path}')
        self.stdout.write('-' * 60)
        
        # Step 1: Extract content from markdown files
        self.stdout.write('📝 Step 1: Extracting content from markdown files...')
        raw_content = process_folder_content(folder)
        
        if not raw_content:
            self.stdout.write(
                self.style.WARNING('❌ No content extracted from markdown files')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Extracted {len(raw_content)} content items')
        )
        
        if dry_run:
            self.stdout.write('🧪 DRY RUN: Stopping before LLM processing')
            return
        
        # Step 2: Process with LLM
        self.stdout.write('🤖 Step 2: Enhancing flashcards with AI...')
        
        result = process_folder_with_llm(folder, raw_content)
        
        if result['success']:
            self.stdout.write(
                self.style.SUCCESS(
                    f'🎉 SUCCESS: {result["message"]}'
                )
            )
            self.stdout.write(f'📊 Processed: {result["processed_count"]} items')
            self.stdout.write(f'💾 Saved: {result["saved_count"]} flashcards')
            
            # Show some sample flashcards
            self.stdout.write('\n📚 Sample generated flashcards:')
            sample_cards = folder.flashcards.all()[:3]
            for i, card in enumerate(sample_cards, 1):
                self.stdout.write(f'\n--- Sample {i} ---')
                self.stdout.write(f'Q: {card.question}')
                self.stdout.write(f'A: {card.answer[:100]}{"..." if len(card.answer) > 100 else ""}')
                self.stdout.write(f'Difficulty: {card.difficulty_level}')
                self.stdout.write(f'Source: {card.source_file}')
        
        else:
            self.stdout.write(
                self.style.ERROR(f'❌ ERROR: {result["message"]}')
            )
            
            if result.get('error') == 'api_key_missing':
                self.stdout.write(
                    self.style.WARNING(
                        '\n💡 To use AI enhancement, you need to:'
                        '\n1. Get an OpenAI API key from https://platform.openai.com/api-keys'
                        '\n2. Copy env_example.txt to .env'
                        '\n3. Add your API key to the .env file'
                    )
                ) 