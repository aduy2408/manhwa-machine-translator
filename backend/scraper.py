import os
import time
import requests
import ddddocr
from DrissionPage import ChromiumPage


def fetch_series_poster(series_url: str, save_path: str) -> str | None:
    """
    Visit the Newtoki series index page and extract the poster image.

    The poster is the first <img> inside an element with class="view-img".
    Downloads the image and saves it to `save_path`.

    Args:
        series_url: Newtoki series index URL
        save_path:  Absolute path (incl. filename) where the image is saved

    Returns:
        save_path on success, None on failure
    """
    page = ChromiumPage()
    try:
        print(f"[Scraper] Fetching poster from: {series_url}")
        page.get(series_url)
        solve_captcha_if_present(page, os.path.dirname(save_path))
        time.sleep(1)

        # Locate the poster: img inside .view-img
        view_img_el = page.ele('.view-img', timeout=5)
        if not view_img_el:
            print("[Scraper] .view-img element not found")
            return None

        img_el = view_img_el.ele('tag:img', timeout=3)
        if not img_el:
            # Maybe the element itself is the img
            img_el = view_img_el if view_img_el.tag == 'img' else None

        if not img_el:
            print("[Scraper] No <img> found inside .view-img")
            return None

        img_url = img_el.attr('src') or img_el.attr('data-src')
        if not img_url:
            print("[Scraper] img has no src attribute")
            return None

        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif img_url.startswith('/'):
            from urllib.parse import urlparse
            parsed = urlparse(series_url)
            img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"

        print(f"[Scraper] Downloading poster: {img_url}")
        headers = {'Referer': series_url, 'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(img_url, headers=headers, timeout=15)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(resp.content)

        print(f"[Scraper] Poster saved to: {save_path}")
        return save_path

    except Exception as e:
        print(f"[Scraper] fetch_series_poster error: {e}")
        return None
    finally:
        page.quit()


def solve_captcha_if_present(page, save_dir):
    time.sleep(0.2)  # Allow time for potential captcha to render
    captcha_ele = page.ele('.captcha_img', timeout=1)
    if captcha_ele:
        print("Captcha element found, attempting to bypass...")
        ocr = ddddocr.DdddOcr(show_ad=False)
        os.makedirs(save_dir, exist_ok=True)
        image_name = os.path.join(save_dir, 'captcha.png')
        
        captcha_ele.get_screenshot(path=image_name)
        with open(image_name, 'rb') as f:
            captcha_text = ocr.classification(f.read())
            
        print(f"Solved captcha: {captcha_text}")
        page.ele('#captcha_key').input(captcha_text)
        
        btn = page.ele('css:button[type="submit"].btn-color')
        if btn:
            btn.click()
        else:
            form = page.ele('tag:form')
            form.submit()
            
        page.wait.load_start()
        time.sleep(0.2)
        return True
    return False

def fetch_chapter_list(series_url: str, save_dir: str) -> list:
    """
    Scrape all available chapters from a Newtoki series index page.

    Returns a list of dicts:
        {
            'data_index':     int  — Newtoki internal li[data-index] value,
            'chapter_number': int  — display number shown left of title (may be None),
            'title':          str  — chapter title text,
        }
    Ordered newest-first (as they appear on the page).
    """
    page = ChromiumPage()
    chapters = []
    try:
        print(f"[Scraper] Fetching chapter list: {series_url}")
        page.get(series_url)
        os.makedirs(save_dir, exist_ok=True)
        solve_captcha_if_present(page, save_dir)
        time.sleep(1)

        items = page.eles('css:li.list-item')
        print(f"[Scraper] Found {len(items)} list items")

        for item in items:
            data_index = item.attr('data-index')
            if not data_index:
                continue

            # Left column: chapter sequence number
            chapter_number = None
            num_el = item.ele('.list-num', timeout=0.1)
            if num_el:
                try:
                    chapter_number = int(num_el.text.strip())
                except ValueError:
                    pass

            # Title text (strip trailing spaces / tag numbers)
            title = ''
            subject_el = item.ele('.wr-subject', timeout=0.1)
            if subject_el:
                title = subject_el.text.strip()

            chapters.append({
                'data_index':     int(data_index),
                'chapter_number': chapter_number,
                'title':          title,
            })

        return chapters

    except Exception as e:
        print(f"[Scraper] fetch_chapter_list error: {e}")
        return []
    finally:
        page.quit()


def scrape_newtoki_chapter(index_url: str, data_index: int, save_dir: str):
    """
    Scrapes a chapter from Newtoki using robust UI automation via DrissionPage.
    Includes logic to bypass captchas using ddddocr.

    Args:
        index_url:   Series index page URL
        data_index:  Newtoki internal li[data-index] value (NOT the display chapter number)
        save_dir:    Directory to save downloaded page images

    Returns:
        List of absolute paths to downloaded images.
    """
    page = ChromiumPage()
    images_downloaded = []
    
    try:
        # 1. Get index url
        print(f"Loading index URL: {index_url}")
        page.get(index_url)
        solve_captcha_if_present(page, save_dir)
        
        # 2. Extract chapter URL using the Newtoki internal data-index
        specific_item = page.ele(f'css:li.list-item[data-index="{data_index}"]')
        
        if not specific_item:
            raise Exception(f"data-index={data_index} not found on the index page.")
        
        wr_subject = specific_item.ele('.wr-subject')
        if wr_subject.tag == 'a':
            chapter_url = wr_subject.link
        else:
            chapter_url = wr_subject.ele('tag:a').link
            
        print(f"URL for data-index={data_index}: {chapter_url}")
        
        # 3. Go to the chapter url with retry logic
        max_retries = 3
        article = None
        for attempt in range(max_retries):
            print(f"Loading chapter URL (Attempt {attempt+1}/{max_retries}): {chapter_url}")
            page.get(chapter_url)
            
            # This handles captcha if it appears
            captcha_found = solve_captcha_if_present(page, save_dir)
            
            # Wait ~1.5 seconds to see if images load (as requested: "after like 1 seconds")
            time.sleep(1)
            
            # Check for characteristic elements of the chapter page
            article = page.ele('css:article[itemprop="articleBody"]', timeout=2)
            if article and article.ele('tag:img', timeout=1):
                print("Images/content detected, proceeding with scrape.")
                break
            else:
                msg = "Captcha was solved but no images seen." if captcha_found else "No images/content detected."
                print(f"{msg} Reloading to try again (Attempt {attempt+1}/{max_retries})...")
        else:
            raise Exception("Failed to load chapter images after multiple retries.")
            
        # 4. Scroll manually to bypass lazy-loading
        print("Scrolling page to trigger lazy loaded images...")
        page.scroll.to_top()
        for _ in range(70):
            page.scroll.down(2500)
            time.sleep(0.1)
        page.scroll.to_bottom()
        # time.sleep(0.4)
            
        # 5. Extract images
        # article is already found in the loop above
        if not article:
            article = page.ele('css:article[itemprop="articleBody"]')
            
        if not article:
            raise Exception("Article body not found. Might be blocked or invalid page.")
            
        print("Scrolling to the bottom to load images...")
        page.scroll.to_bottom()
        time.sleep(3)  # Wait for final images to load
        
        # Newtoki uses varying structures (sometimes #html_encoder_div, sometimes direct <p>, sometimes <div>)
        # We greedily find ALL images inside the article body and filter by standard manga dimensions!
        images = article.eles('tag:img')
        print(f"Found {len(images)} potential image tags to evaluate.")
        
        if not images:
            print("--- HTML DEBUG (Article) ---")
            print(article.html[:2000]) # Print first 2000 chars for debugging
            print("----------------------------")
            raise Exception("No img tags found inside article body.")
            
        downloaded_count = 0
        os.makedirs(save_dir, exist_ok=True)
        
        for img in images:
            w = img.run_js('return this.naturalWidth;')
            h = img.run_js('return this.naturalHeight;')
            
            if not w or not h:
                continue
                
            # Manga pages are typically large. Filter out UI icons (usually < 200px)
            if w > 400 and h > 400:
                downloaded_count += 1
                ext = "jpg"
                file_name = f"page_{downloaded_count:03d}.{ext}"
                img.save(path=save_dir, name=file_name)
                
                full_path = os.path.join(save_dir, file_name)
                images_downloaded.append(full_path)
                print(f"Saved {file_name} (Size: {w}x{h})")
                
        if downloaded_count == 0:
            print("--- HTML DEBUG (Article) ---")
            print(article.html[:2000])
            print("----------------------------")
            raise Exception("Found img tags, but none matched the >400x400 dimension criteria.")
            
    finally:
        # Guarantee browser quits
        page.quit()

    return images_downloaded
