from django.core.management.base import BaseCommand
from openai import OpenAI
import os

class Command(BaseCommand):
    help = 'Test OpenAI API connection and configuration'
    
    def handle(self, *args, **options):
        # Test API key
        api_key = os.getenv('OPENAI_API_KEY')
        
        self.stdout.write(f"API Key exists: {bool(api_key)}")
        self.stdout.write(f"API Key length: {len(api_key) if api_key else 0}")
        self.stdout.write(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")
        
        if not api_key:
            self.stdout.write(self.style.ERROR("❌ No OpenAI API key found"))
            return
        
        try:
            # Test API call
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Say 'Hello, API is working!'"}
                ],
                max_tokens=10
            )
            
            result = response.choices[0].message.content
            self.stdout.write(self.style.SUCCESS(f"✅ API Test Successful: {result}"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ API Test Failed: {str(e)}"))
