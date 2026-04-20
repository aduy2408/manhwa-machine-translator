import os
import cv2
import numpy as np
from celery_app import celery_app
from database import SessionLocal
from models import Series, RawChapter, RawPage, TranslatedChapter, TranslatedPage
from text_detector import TextDetector
from translator import Translator
from image_processor import ImageProcessor
from scraper import scrape_newtoki_chapter, fetch_series_poster, fetch_chapter_list

# Initialize models lazily or globally per worker process
detector = None
translator = None

def get_detector():
    global detector
    if detector is None:
        detector = TextDetector()
    return detector

def get_translator():
    global translator
    if translator is None:
        translator = Translator()
    return translator


@celery_app.task(bind=True)
def fetch_poster_task(self, series_id: int, series_url: str):
    """
    Background task: fetches the series poster from the Newtoki index page
    and updates series.cover_image in the DB.
    """
    db = SessionLocal()
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, ".library_data")
        save_path = os.path.join(DATA_DIR, f"cover_{series_id}.jpg")

        result_path = fetch_series_poster(series_url, save_path)

        if result_path:
            series = db.query(Series).filter(Series.id == series_id).first()
            if series:
                series.cover_image = f"/data/cover_{series_id}.jpg"
                db.commit()
                print(f"[fetch_poster_task] Poster saved for series {series_id}")
        else:
            print(f"[fetch_poster_task] Failed to fetch poster for series {series_id}")

        return {"status": "ok"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@celery_app.task(bind=True)
def process_chapter(self, raw_chapter_id: int, translated_chapter_id: int, context: str = None):
    db = SessionLocal()
    try:
        raw_chapter = db.query(RawChapter).filter(RawChapter.id == raw_chapter_id).first()
        translated_chapter = db.query(TranslatedChapter).filter(TranslatedChapter.id == translated_chapter_id).first()
        
        if not raw_chapter or not translated_chapter:
            return {"status": "error", "message": "Chapter not found"}

        translated_chapter.status = "processing"
        translated_chapter.task_id = self.request.id
        db.commit()

        pages = db.query(RawPage).filter(RawPage.chapter_id == raw_chapter_id).order_by(RawPage.page_number).all()
        if not pages:
            translated_chapter.status = "failed"
            db.commit()
            return {"status": "error", "message": "No pages found"}

        # Step 1: Detect text in all pages
        det = get_detector()
        all_texts = []
        detection_results = []

        total_pages = len(pages)
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': total_pages, 'status': 'Detecting text...'})

        for i, page in enumerate(pages):
            image = cv2.imread(page.original_path)
            if image is None:
                continue
            
            result = det.detect_and_recognize_text(image)
            if result:
                polygons, recognized_texts = result
                detection_results.append((page, image, polygons, recognized_texts))
                for item in recognized_texts:
                    korean_text = item['text']
                    if korean_text not in all_texts:
                        all_texts.append(korean_text)
            else:
                # Still map the page even if no text detected, so we don't drop pages!
                detection_results.append((page, image, None, None))
            
            # Update progress
            self.update_state(state='PROGRESS', meta={
                'current': i + 1, 
                'total': total_pages, 
                'status': 'Detecting text...'
            })

        # Step 2: Translate all unique texts
        self.update_state(state='PROGRESS', meta={'current': total_pages, 'total': total_pages, 'status': 'Translating text...'})
        all_translations = {}
        if all_texts:
            trans = get_translator()
            all_translations = trans.translate_batch(all_texts, context=context)

        # Step 3: Draw translations and save
        total_detection_results = len(detection_results)
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': total_detection_results, 'status': 'Rendering images...'})
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, ".library_data")
        trans_dir = os.path.join(DATA_DIR, f"series_{translated_chapter.series_id}", f"translated_chapter_{translated_chapter.chapter_number}", "translated")
        os.makedirs(trans_dir, exist_ok=True)
        
        for i, (page, image, polygons, recognized_texts) in enumerate(detection_results):
            image_with_translations = image.copy()
            if polygons is not None and recognized_texts is not None:
                ImageProcessor.draw_translations_on_image(
                    image_with_translations, polygons, recognized_texts, all_translations
                )
            
            trans_filename = os.path.basename(page.original_path)
            if not trans_filename:
                trans_filename = f"page_{page.page_number:03d}.webp"
            
            final_trans_path = os.path.join(trans_dir, trans_filename)
            
            # Save the translated image
            cv2.imwrite(final_trans_path, image_with_translations)
            
            new_page = TranslatedPage(
                chapter_id=translated_chapter.id,
                page_number=page.page_number,
                translated_path=final_trans_path
            )
            db.add(new_page)
            
            self.update_state(state='PROGRESS', meta={
                'current': i + 1, 
                'total': total_detection_results, 
                'status': 'Rendering images...'
            })

        translated_chapter.status = "translated"
        db.commit()

        return {"status": "success", "message": f"Processed {total_pages} pages"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'translated_chapter' in locals() and translated_chapter:
            translated_chapter.status = "failed"
            db.commit()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@celery_app.task(bind=True)
def scrape_chapter_task(self, raw_chapter_id: int):
    db = SessionLocal()
    try:
        raw_chapter = db.query(RawChapter).filter(RawChapter.id == raw_chapter_id).first()
        if not raw_chapter:
            return {"status": "error", "message": "Chapter not found"}

        # Resolve index URL from the parent series
        series = db.query(Series).filter(Series.id == raw_chapter.series_id).first()
        if not series or not series.source_url:
            raw_chapter.status = "failed"
            db.commit()
            return {"status": "error", "message": "Series has no source_url set. Please edit the series and add the Newtoki URL."}

        index_url = series.source_url

        raw_chapter.status = "scraping"
        raw_chapter.task_id = self.request.id
        db.commit()

        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 1, 'status': 'Initializing Scraper...'})

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(BASE_DIR, ".library_data")
        chapter_dir = os.path.join(DATA_DIR, f"series_{raw_chapter.series_id}", f"raw_chapter_{raw_chapter.chapter_number}")
        orig_dir = os.path.join(chapter_dir, "original")
        os.makedirs(orig_dir, exist_ok=True)

        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 1, 'status': 'Opening Newtoki...'})

        # Use stored data_index (Newtoki internal ID) for navigation
        nav_index = raw_chapter.data_index if raw_chapter.data_index else raw_chapter.chapter_number
        downloaded_images = scrape_newtoki_chapter(index_url, nav_index, orig_dir)

        if not downloaded_images:
            raise Exception("No images were downloaded.")

        self.update_state(state='PROGRESS', meta={'current': 1, 'total': 1, 'status': 'Saving to database...'})

        for i, original_path in enumerate(downloaded_images):
            page = RawPage(
                chapter_id=raw_chapter.id,
                page_number=i+1,
                original_path=original_path
            )
            db.add(page)

        raw_chapter.status = "scraped"
        db.commit()

        return {"status": "success", "message": f"Scraped {len(downloaded_images)} images"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'raw_chapter' in locals() and raw_chapter:
            raw_chapter.status = "failed"
            db.commit()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
