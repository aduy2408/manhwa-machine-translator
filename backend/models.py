from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Series(Base):
    __tablename__ = "series"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, nullable=True)
    cover_image = Column(String, nullable=True)
    source_url = Column(String, nullable=True)  # Newtoki series index URL

    raw_chapters = relationship("RawChapter", back_populates="series", cascade="all, delete-orphan")
    translated_chapters = relationship("TranslatedChapter", back_populates="series", cascade="all, delete-orphan")


class RawChapter(Base):
    __tablename__ = "raw_chapters"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(Integer, ForeignKey("series.id"))
    chapter_number = Column(Integer)
    data_index = Column(Integer, nullable=True)   # Newtoki internal li[data-index] value
    title = Column(String, nullable=True)
    status = Column(String, default="scraped")  # scraping, scraped, failed
    task_id = Column(String, nullable=True)

    series = relationship("Series", back_populates="raw_chapters")
    pages = relationship("RawPage", back_populates="chapter", cascade="all, delete-orphan", order_by="RawPage.page_number")


class RawPage(Base):
    __tablename__ = "raw_pages"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("raw_chapters.id"))
    page_number = Column(Integer)
    original_path = Column(String)

    chapter = relationship("RawChapter", back_populates="pages")

class TranslatedChapter(Base):
    __tablename__ = "translated_chapters"

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(Integer, ForeignKey("series.id"))
    chapter_number = Column(Integer)
    title = Column(String, nullable=True)
    status = Column(String, default="processing")  # processing, translated, failed
    task_id = Column(String, nullable=True)

    series = relationship("Series", back_populates="translated_chapters")
    pages = relationship("TranslatedPage", back_populates="chapter", cascade="all, delete-orphan", order_by="TranslatedPage.page_number")

class TranslatedPage(Base):
    __tablename__ = "translated_pages"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("translated_chapters.id"))
    page_number = Column(Integer)
    translated_path = Column(String, nullable=True)

    chapter = relationship("TranslatedChapter", back_populates="pages")
