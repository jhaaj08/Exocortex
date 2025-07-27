from django.core.management.base import BaseCommand
from flashcards.models import PDFDocument

class Command(BaseCommand):
    help = 'Clean up duplicate PDF entries'

    def handle(self, *args, **options):
        # Find PDFs with same name and content
        pdf_names = PDFDocument.objects.values('name').distinct()
        
        deleted_count = 0
        for name_dict in pdf_names:
            name = name_dict['name']
            pdfs = PDFDocument.objects.filter(name=name).order_by('created_at')
            
            if pdfs.count() > 1:
                # Keep the first one, delete the rest
                keep_pdf = pdfs.first()
                duplicates = pdfs.exclude(id=keep_pdf.id)
                
                self.stdout.write(f"Found {duplicates.count()} duplicates of '{name}'")
                
                for dup in duplicates:
                    self.stdout.write(f"  Deleting: {dup.id} - {dup.created_at}")
                    dup.delete()
                    deleted_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully deleted {deleted_count} duplicate PDFs')
        )
