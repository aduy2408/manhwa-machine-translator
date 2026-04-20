import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "manhwa_translator",
    broker=redis_url,
    backend=redis_url,
    include=["worker"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,

    # Route scraping tasks to a dedicated queue so concurrency can be
    # capped at 1 (only one Chromium browser open at a time).
    task_routes={
        "worker.scrape_chapter_task": {"queue": "scrape"},
        "worker.fetch_poster_task":   {"queue": "scrape"},
    }
)
