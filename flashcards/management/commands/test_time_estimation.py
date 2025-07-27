from django.core.management.base import BaseCommand
from flashcards.models import PDFDocument, ConceptUnit
from flashcards.reading_time_service import optimize_reading_time

class Command(BaseCommand):
    help = 'Test reading time optimization for latest PDF'
    
    def handle(self, *args, **options):
        # Get latest PDF
        pdf = PDFDocument.objects.filter(concepts_analyzed=True).last()
        
        if not pdf:
            self.stdout.write(self.style.ERROR("No PDFs with concepts found"))
            return
        
        self.stdout.write(f"Testing time optimization for: {pdf.name}")
        
        # Show current concept units
        units = pdf.concept_units.all().order_by('concept_order')
        self.stdout.write(f"Found {units.count()} concept units:")
        
        for unit in units:
            self.stdout.write(f"  Unit {unit.concept_order}: {unit.word_count} words, {unit.estimated_reading_time:.1f} min")
        
        # Test time optimization
        success, message = optimize_reading_time(pdf)
        
        if success:
            self.stdout.write(self.style.SUCCESS(f"✅ {message}"))
        else:
            self.stdout.write(self.style.ERROR(f"❌ {message}"))
        
        # Show results
        units = pdf.concept_units.all().order_by('concept_order')
        self.stdout.write(f"After optimization - {units.count()} concept units:")
        
        for unit in units:
            self.stdout.write(f"  Unit {unit.concept_order}: {unit.word_count} words, {unit.estimated_reading_time:.1f} min, optimized: {unit.time_optimized}")
