from django.core.management.base import BaseCommand
from flashcards.models import PDFDocument, ConceptUnit, ChunkLabel

class Command(BaseCommand):
    help = 'Clean up concept units that exist without proper chunk labels'
    
    def handle(self, *args, **options):
        # Find PDFs with concept units but no chunk labels
        inconsistent_pdfs = PDFDocument.objects.filter(
            concept_units__isnull=False,
            text_chunks__chunk_label__isnull=True
        ).distinct()
        
        for pdf in inconsistent_pdfs:
            chunk_labels_count = ChunkLabel.objects.filter(text_chunk__pdf_document=pdf).count()
            concept_units_count = pdf.concept_units.count()
            
            if chunk_labels_count == 0 and concept_units_count > 0:
                self.stdout.write(
                    self.style.WARNING(f'Found inconsistency in PDF {pdf.id}: {concept_units_count} concept units but 0 labeled chunks')
                )
                
                # Clean up orphaned concept units
                deleted_count = pdf.concept_units.all().delete()[0]
                pdf.concepts_analyzed = False
                pdf.save()
                
                self.stdout.write(
                    self.style.SUCCESS(f'Cleaned up {deleted_count} orphaned concept units for PDF {pdf.id}')
                )
        
        self.stdout.write(self.style.SUCCESS('Cleanup complete'))
