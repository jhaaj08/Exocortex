from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from flashcards.models import StudySession, FocusBlock, UserBlockState


class Command(BaseCommand):
    help = 'Backfill UserBlockState records from existing StudySession completion data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating records',
        )
        parser.add_argument(
            '--user',
            type=str,
            help='Backfill for specific username only',
        )
    
    def handle(self, *args, **options):
        dry_run = options['dry_run']
        target_user = options['user']
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No records will be created'))
        
        # Get all users or specific user
        if target_user:
            try:
                users = [User.objects.get(username=target_user)]
                self.stdout.write(f"Backfilling for user: {target_user}")
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"User '{target_user}' not found"))
                return
        else:
            users = User.objects.all()
            self.stdout.write(f"Backfilling for {users.count()} users")
        
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        for user in users:
            self.stdout.write(f"\n🔍 Processing user: {user.username}")
            
            # Find study sessions for this user (assuming admin user for now)
            # Note: StudySession doesn't have user field yet, so we'll use admin user as fallback
            user_sessions = StudySession.objects.filter(
                block_completion_data__isnull=False
            ).exclude(block_completion_data={})
            
            for session in user_sessions:
                for block_id_str, completion_data in session.block_completion_data.items():
                    try:
                        block = FocusBlock.objects.get(id=block_id_str)
                    except FocusBlock.DoesNotExist:
                        self.stdout.write(f"  ⚠️  Block {block_id_str} not found")
                        skipped_count += 1
                        continue
                    
                    # Check if UserBlockState already exists
                    state, created = UserBlockState.objects.get_or_create(
                        user=user,
                        block=block,
                        defaults={
                            'status': 'review',  # Assume completed blocks are in review
                            'review_count': 1,
                            'avg_rating': completion_data.get('proficiency_rating', 3),
                            'total_time_spent': completion_data.get('time_spent_seconds', 0),
                            'ease_factor': 2.5,
                            'stability': 1.0,
                        }
                    )
                    
                    if not dry_run:
                        if created:
                            # Set scheduling based on rating
                            rating = completion_data.get('proficiency_rating', 3)
                            time_spent = completion_data.get('time_spent_seconds', 0)
                            
                            state.update_from_rating(rating, time_spent)
                            created_count += 1
                            
                            self.stdout.write(
                                f"  ✅ Created: {block.title[:50]}... "
                                f"(rating: {rating}, next due: {state.days_until_due()} days)"
                            )
                        else:
                            # Update existing record if new data is more recent
                            existing_rating = state.avg_rating or 0
                            new_rating = completion_data.get('proficiency_rating', 3)
                            
                            if new_rating != existing_rating:
                                state.avg_rating = new_rating
                                state.total_time_spent += completion_data.get('time_spent_seconds', 0)
                                state.save()
                                updated_count += 1
                                
                                self.stdout.write(
                                    f"  🔄 Updated: {block.title[:50]}... "
                                    f"(rating: {existing_rating} → {new_rating})"
                                )
                            else:
                                skipped_count += 1
                    else:
                        # Dry run: just log what would happen
                        action = "CREATE" if created else "UPDATE"
                        rating = completion_data.get('proficiency_rating', 3)
                        self.stdout.write(
                            f"  [{action}] {block.title[:50]}... (rating: {rating})"
                        )
                        if created:
                            created_count += 1
                        else:
                            updated_count += 1
        
        # Summary
        self.stdout.write(f"\n{'='*50}")
        self.stdout.write(self.style.SUCCESS(f"📊 BACKFILL SUMMARY:"))
        self.stdout.write(f"  ✅ Created: {created_count} records")
        self.stdout.write(f"  🔄 Updated: {updated_count} records")
        self.stdout.write(f"  ⏭️  Skipped: {skipped_count} records")
        
        if dry_run:
            self.stdout.write(self.style.WARNING("\n⚠️  This was a DRY RUN - no actual changes made"))
            self.stdout.write("Run without --dry-run to apply changes")
        else:
            self.stdout.write(self.style.SUCCESS(f"\n🎉 Backfill completed!"))
