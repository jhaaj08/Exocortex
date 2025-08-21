web: python manage.py migrate && gunicorn flashcard_project.wsgi:application --bind 0.0.0.0:$PORT --timeout 300
worker: celery -A flashcard_project worker --loglevel=info --concurrency=2
beat: celery -A flashcard_project beat --loglevel=info
