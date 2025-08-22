web: python manage.py migrate && python manage.py ensure_superuser && gunicorn flashcard_project.wsgi:application --bind 0.0.0.0:$PORT --timeout 300 --workers 2
worker: celery -A flashcard_project worker --loglevel=info --concurrency=1 --pool=solo
beat: celery -A flashcard_project beat --loglevel=info
