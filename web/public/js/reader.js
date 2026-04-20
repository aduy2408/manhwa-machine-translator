/* ============================================================
   reader.js — manga reader view
   ============================================================ */

async function openReader(chapterId, chNum, type) {
    AppState.currentReaderMode = type;
    try {
        const res  = await fetch(`${API}/${type}/${chapterId}/pages`);
        const data = await res.json();

        AppState.readerPages       = data.pages;
        AppState.showingTranslation = (type === 'translated');

        document.getElementById('reader-title').textContent =
            `Chapter ${chNum} (${type.toUpperCase()})`;

        renderReaderPages();
        switchView('reader-view');
    } catch (e) {
        console.error(e);
        alert('Failed to load chapter pages');
    }
}

function renderReaderPages() {
    const container = document.getElementById('reader-pages');
    container.innerHTML = '';

    AppState.readerPages.forEach(p => {
        const img = document.createElement('img');
        img.className = 'manga-page';
        img.loading = 'lazy';

        if (!p.url) {
            img.alt = 'Image not found';
        } else {
            img.src = p.url.startsWith('http') ? p.url : `http://127.0.0.1:5000${p.url}`;
        }

        container.appendChild(img);
    });

    // Toggle button not used in split-view mode
    const btn = document.getElementById('toggle-translation-btn');
    if (btn) btn.style.display = 'none';
}

function toggleTranslationState() {
    AppState.showingTranslation = !AppState.showingTranslation;
    renderReaderPages();
}
