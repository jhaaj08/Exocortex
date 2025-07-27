from django.core.management.base import BaseCommand
from flashcards.models import FocusBlock

class Command(BaseCommand):
    help = 'Regenerate titles for existing focus blocks'

    def handle(self, *args, **options):
        focus_blocks = FocusBlock.objects.all()
        
        self.stdout.write(f'Found {focus_blocks.count()} focus blocks to update')
        
        updated_count = 0
        for block in focus_blocks:
            if block.compact7_data:
                new_title = self.generate_focus_block_title(
                    block.compact7_data, 
                    block.main_concept_unit, 
                    block.block_order
                )
                
                if new_title != block.title:
                    old_title = block.title
                    block.title = new_title
                    block.save()
                    
                    self.stdout.write(f'Updated: "{old_title}" → "{new_title}"')
                    updated_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully updated {updated_count} focus block titles')
        )
    
    def generate_focus_block_title(self, compact7_data, main_unit, order):
        """Generate a meaningful title for the focus block"""
        
        # Try different sources for title, in order of preference
        title_sources = []
        
        # 1. Use core goal (cleaned up)
        if 'core' in compact7_data and 'goal' in compact7_data['core']:
            goal = compact7_data['core']['goal']
            clean_goal = self.clean_goal_for_title(goal)
            if clean_goal:
                title_sources.append(clean_goal)
        
        # 2. Use first segment title
        if 'core' in compact7_data and 'segments' in compact7_data['core']:
            segments = compact7_data['core']['segments']
            if segments and isinstance(segments, list) and len(segments) > 0:
                if 'title' in segments[0]:
                    segment_title = segments[0]['title']
                    title_sources.append(segment_title)
        
        # 3. Use concept unit title (cleaned)
        if main_unit.title:
            clean_unit_title = main_unit.title.replace('Combined Concept', '').strip()
            clean_unit_title = clean_unit_title.replace('Concept Unit', '').strip()
            clean_unit_title = clean_unit_title.replace(':', '').strip()
            if clean_unit_title:
                title_sources.append(clean_unit_title)
        
        # 4. Use primary labels
        if main_unit.primary_labels:
            labels_title = ' & '.join(main_unit.primary_labels[:2])
            title_sources.append(f"Learn {labels_title}")
        
        # 5. Fallback
        title_sources.append(f"Focus Session {order}")
        
        # Pick the best title
        for potential_title in title_sources:
            if potential_title and len(potential_title.strip()) > 5:
                # Format nicely
                final_title = f"{order}. {potential_title}"
                
                # Ensure it fits in database field
                if len(final_title) > 250:
                    final_title = final_title[:247] + "..."
                
                return final_title
        
        # Ultimate fallback
        return f"{order}. Focus Session"
    
    def clean_goal_for_title(self, goal):
        """Clean up a learning goal to make a good title"""
        if not goal:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Understand ",
            "Learn about ",
            "Explore ",
            "Study ",
            "Differentiate ",
            "Identify ",
            "Analyze ",
            "The goal is to ",
            "Students will ",
            "You will ",
            "Learn ",
            "Master "
        ]
        
        clean_goal = goal.strip()
        for prefix in prefixes_to_remove:
            if clean_goal.lower().startswith(prefix.lower()):
                clean_goal = clean_goal[len(prefix):].strip()
                break
        
        # Capitalize first letter
        if clean_goal:
            clean_goal = clean_goal[0].upper() + clean_goal[1:]
        
        # Remove trailing periods
        clean_goal = clean_goal.rstrip('.')
        
        # Limit length for title
        if len(clean_goal) > 60:
            # Find a good break point
            words = clean_goal.split()
            shortened = []
            char_count = 0
            
            for word in words:
                if char_count + len(word) + 1 > 60:
                    break
                shortened.append(word)
                char_count += len(word) + 1
            
            if shortened:
                clean_goal = ' '.join(shortened)
        
        return clean_goal
