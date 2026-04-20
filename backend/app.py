"""FastAPI backend for Korean text translation app."""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
from natsort import natsorted

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, ".library_data")
WEB_DIR = os.path.join(BASE_DIR, "web", "public")

from database import engine, Base, get_db
from models import Series, RawChapter, RawPage, TranslatedChapter, TranslatedPage
from worker import process_chapter, scrape_chapter_task, fetch_poster_task
from scraper import fetch_chapter_list

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Korean Manhwa Translator",
    description="Detect and translate Korean text in manhwa images",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup discreet folder for images so the frontend can retrieve them
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# --- API Endpoints ---

@app.get("/api/series")
def list_series(db: Session = Depends(get_db)):
    series = db.query(Series).all()
    res = []
    for s in series:
        raw_count = db.query(RawChapter).filter(RawChapter.series_id == s.id).count()
        trans_count = db.query(TranslatedChapter).filter(TranslatedChapter.series_id == s.id).count()
        res.append({
            "id": s.id,
            "title": s.title,
            "description": s.description,
            "cover_image": s.cover_image,
            "source_url": s.source_url,
            "raw_chapter_count": raw_count,
            "translated_chapter_count": trans_count
        })
    return res

@app.post("/api/series")
async def create_series(
    title: str = Form(...),
    description: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    series = Series(title=title, description=description, source_url=source_url)
    db.add(series)
    db.commit()
    db.refresh(series)

    # Auto-fetch poster from the Newtoki series page if a URL was supplied
    if source_url:
        fetch_poster_task.delay(series.id, source_url)

    return series

@app.get("/api/series/{series_id}")
def get_series(series_id: int, db: Session = Depends(get_db)):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    
    raw_chapters = db.query(RawChapter).filter(RawChapter.series_id == series_id).order_by(RawChapter.chapter_number).all()
    translated_chapters = db.query(TranslatedChapter).filter(TranslatedChapter.series_id == series_id).order_by(TranslatedChapter.chapter_number).all()
    
    return {
        "id": series.id,
        "title": series.title,
        "description": series.description,
        "cover_image": series.cover_image,
        "source_url": series.source_url,
        "raw_chapters": raw_chapters,
        "translated_chapters": translated_chapters
    }


@app.delete("/api/series/{series_id}")
def delete_series(series_id: int, db: Session = Depends(get_db)):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    # Delete series data folder (raw + translated pages)
    series_dir = os.path.join(DATA_DIR, f"series_{series_id}")
    if os.path.exists(series_dir):
        shutil.rmtree(series_dir, ignore_errors=True)

    # Delete root-level cover image
    if series.cover_image:
        cover_filename = os.path.basename(series.cover_image)
        cover_path = os.path.join(DATA_DIR, cover_filename)
        if os.path.exists(cover_path):
            os.remove(cover_path)

    db.delete(series)
    db.commit()
    return {"message": f"Series '{series.title}' deleted"}

@app.post("/api/series/{series_id}/chapters")
async def add_chapter(
    series_id: int,
    chapter_number: int = Form(...),
    title: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
        
    # Manual uploading adds to RawChapter natively
    raw_chapter = RawChapter(series_id=series_id, chapter_number=chapter_number, title=title, status="scraped")
    db.add(raw_chapter)
    db.commit()
    db.refresh(raw_chapter)
    
    chapter_dir = os.path.join(DATA_DIR, f"series_{series_id}", f"raw_chapter_{raw_chapter.chapter_number}")
    orig_dir = os.path.join(chapter_dir, "original")
    os.makedirs(orig_dir, exist_ok=True)
    
    # Sort files naturally
    sorted_files = natsorted(files, key=lambda x: x.filename)
    
    for i, file in enumerate(sorted_files):
        ext = file.filename.split(".")[-1]
        page_name = f"page_{i+1:03d}.{ext}"
        orig_path = os.path.join(orig_dir, page_name)
        
        with open(orig_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        page = RawPage(
            chapter_id=raw_chapter.id,
            page_number=i+1,
            original_path=orig_path
        )
        db.add(page)
        
    db.commit()
    
    # Auto-start translation
    trans_chapter = TranslatedChapter(series_id=series_id, chapter_number=chapter_number, title=title, status="processing")
    db.add(trans_chapter)
    db.commit()
    db.refresh(trans_chapter)
    
    task = process_chapter.delay(raw_chapter.id, trans_chapter.id, context)
    trans_chapter.task_id = task.id
    db.commit()
    
    return {"message": "Chapter uploaded and queued for processing", "raw_chapter_id": raw_chapter.id, "translated_chapter_id": trans_chapter.id}

@app.post("/api/series/{series_id}/scrape")
async def scrape_chapter(
    series_id: int,
    chapter_number: int = Form(...),
    data_index: Optional[int] = Form(None),
    title: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    if not series.source_url:
        raise HTTPException(
            status_code=400,
            detail="This series has no Newtoki URL. Edit the series and add one first."
        )

    raw_chapter = RawChapter(
        series_id=series_id,
        chapter_number=chapter_number,
        data_index=data_index,
        title=title,
        status="scraping"
    )
    db.add(raw_chapter)
    db.commit()
    db.refresh(raw_chapter)

    task = scrape_chapter_task.delay(raw_chapter.id)
    raw_chapter.task_id = task.id
    db.commit()

    return {"message": "Chapter scraping queued", "raw_chapter_id": raw_chapter.id, "task_id": task.id}


@app.get("/api/series/{series_id}/chapter-list")
async def get_chapter_list(series_id: int, db: Session = Depends(get_db)):
    """
    Scrape the series index page and return the full chapter list.
    Each entry: {data_index, chapter_number, title}
    """
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    if not series.source_url:
        raise HTTPException(status_code=400, detail="No source URL set for this series.")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR_LOC = os.path.join(BASE_DIR, ".library_data")
    save_dir = os.path.join(DATA_DIR_LOC, f"series_{series_id}", "tmp")

    chapters = fetch_chapter_list(series.source_url, save_dir)

    # Mark which data_indexes are already scraped/in-progress
    existing = {
        ch.data_index
        for ch in db.query(RawChapter).filter(RawChapter.series_id == series_id).all()
        if ch.data_index is not None
    }

    for ch in chapters:
        ch['already_scraped'] = ch['data_index'] in existing

    return {"chapters": chapters}

@app.post("/api/raw/{chapter_id}/translate")
async def translate_chapter(
    chapter_id: int,
    context: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    raw_chapter = db.query(RawChapter).filter(RawChapter.id == chapter_id).first()
    if not raw_chapter:
        raise HTTPException(status_code=404, detail="Raw chapter not found")
        
    if raw_chapter.status != "scraped":
        raise HTTPException(status_code=400, detail="Chapter is not fully scraped yet")
        
    trans_chapter = TranslatedChapter(
        series_id=raw_chapter.series_id, 
        chapter_number=raw_chapter.chapter_number, 
        title=raw_chapter.title, 
        status="processing"
    )
    db.add(trans_chapter)
    db.commit()
    db.refresh(trans_chapter)
    
    task = process_chapter.delay(raw_chapter.id, trans_chapter.id, context)
    trans_chapter.task_id = task.id
    db.commit()
    
    return {"message": "Translation queued", "translated_chapter_id": trans_chapter.id}

@app.get("/api/raw/{chapter_id}/status")
def get_raw_chapter_status(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(RawChapter).filter(RawChapter.id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
        
    progress = None
    if chapter.status == "scraping" and chapter.task_id:
        task_res = scrape_chapter_task.AsyncResult(chapter.task_id)
        if task_res.state == 'PROGRESS':
            progress = task_res.info
        elif task_res.state == 'SUCCESS':
            db.refresh(chapter)
        elif task_res.state == 'FAILURE':
            chapter.status = "failed"
            db.commit()
            
    return {
        "chapter_id": chapter.id,
        "status": chapter.status,
        "progress": progress
    }

@app.get("/api/translated/{chapter_id}/status")
def get_translated_chapter_status(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(TranslatedChapter).filter(TranslatedChapter.id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
        
    progress = None
    if chapter.status == "processing" and chapter.task_id:
        task_res = process_chapter.AsyncResult(chapter.task_id)
        if task_res.state == 'PROGRESS':
            progress = task_res.info
        elif task_res.state == 'SUCCESS':
            db.refresh(chapter)
        elif task_res.state == 'FAILURE':
            chapter.status = "failed"
            db.commit()
            
    return {
        "chapter_id": chapter.id,
        "status": chapter.status,
        "progress": progress
    }

@app.delete("/api/raw/{chapter_id}")
def delete_raw_chapter(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(RawChapter).filter(RawChapter.id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # Path deletion could be added here if needed
    chapter_dir = os.path.join(DATA_DIR, f"series_{chapter.series_id}", f"raw_chapter_{chapter.chapter_number}")
    if os.path.exists(chapter_dir):
        shutil.rmtree(chapter_dir, ignore_errors=True)
        
    db.delete(chapter)
    db.commit()
    return {"message": "Deleted"}

@app.delete("/api/translated/{chapter_id}")
def delete_translated_chapter(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(TranslatedChapter).filter(TranslatedChapter.id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
        
    chapter_dir = os.path.join(DATA_DIR, f"series_{chapter.series_id}", f"translated_chapter_{chapter.chapter_number}")
    if os.path.exists(chapter_dir):
        shutil.rmtree(chapter_dir, ignore_errors=True)
        
    db.delete(chapter)
    db.commit()
    return {"message": "Deleted"}

@app.get("/api/raw/{chapter_id}/pages")
def get_raw_pages(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(RawChapter).filter(RawChapter.id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
        
    pages = db.query(RawPage).filter(RawPage.chapter_id == chapter_id).order_by(RawPage.page_number).all()
    page_urls = []
    for p in pages:
        orig = f"/data/{os.path.relpath(p.original_path, DATA_DIR)}" if p.original_path else None
        if orig: orig = orig.replace('\\', '/')
        page_urls.append({"id": p.id, "page_number": p.page_number, "url": orig})
        
    return {"chapter_id": chapter.id, "status": chapter.status, "pages": page_urls}

@app.get("/api/translated/{chapter_id}/pages")
def get_translated_pages(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(TranslatedChapter).filter(TranslatedChapter.id == chapter_id).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
        
    pages = db.query(TranslatedPage).filter(TranslatedPage.chapter_id == chapter_id).order_by(TranslatedPage.page_number).all()
    page_urls = []
    for p in pages:
        orig = f"/data/{os.path.relpath(p.translated_path, DATA_DIR)}" if p.translated_path else None
        if orig: orig = orig.replace('\\', '/')
        page_urls.append({"id": p.id, "page_number": p.page_number, "url": orig})
        
    return {"chapter_id": chapter.id, "status": chapter.status, "pages": page_urls}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
