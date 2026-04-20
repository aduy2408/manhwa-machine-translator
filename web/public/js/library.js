/* ============================================================
   library.js — library view: list series + add series modal
   ============================================================ */

async function loadLibrary() {
    try {
        const res = await fetch(`${API}/series`);
        const seriesList = await res.json();
        const grid = document.getElementById('series-grid');
        grid.innerHTML = '';

        if (seriesList.length === 0) {
            grid.innerHTML =
                '<div style="grid-column:1/-1;text-align:center;padding:3rem;color:var(--text-muted);">' +
                'No series found. Add one to get started!</div>';
            return;
        }

        seriesList.forEach(s => {
            const cover = s.cover_image
                ? (s.cover_image.startsWith('http') ? s.cover_image : `http://127.0.0.1:5000${s.cover_image}`)
                : PLACEHOLDER_IMG;

            const card = document.createElement('div');
            card.className = 'card';
            card.style.position = 'relative';
            card.onclick = () => {
                AppState.currentSeriesId = s.id;
                switchView('series-view', () => loadSeries(s.id));
            };
            card.innerHTML = `
                <div class="card-img-wrapper">
                    <img src="${cover}" alt="${s.title}">
                </div>
                <div class="card-content">
                    <div class="card-title">${s.title}</div>
                    <div class="card-meta">${(s.raw_chapter_count || 0)} Chapters</div>
                </div>
                <button class="card-delete-btn" title="Delete series"
                    onclick="event.stopPropagation(); deleteSeries(${s.id}, '${s.title.replace(/'/g, "\\'")}')"
                >&#10005;</button>
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        console.error('Failed to load library', err);
    }
}

async function deleteSeries(id, title) {
    if (!confirm(`Delete "${title}" and ALL its chapters? This cannot be undone.`)) return;
    try {
        const res = await fetch(`${API}/series/${id}`, { method: 'DELETE' });
        if (res.ok) {
            loadLibrary();
        } else {
            const err = await res.json();
            alert('Failed to delete: ' + (err.detail || 'Unknown error'));
        }
    } catch (err) {
        console.error(err);
    }
}

async function submitAddSeries(e) {
    e.preventDefault();
    const form = e.target;
    const btn = document.getElementById('addSeriesBtn');
    const formData = new FormData(form);

    btn.disabled = true;
    btn.textContent = 'Creating…';

    try {
        const res = await fetch(`${API}/series`, { method: 'POST', body: formData });
        if (res.ok) {
            const series = await res.json();
            closeModal('addSeriesModal');
            form.reset();
            document.getElementById('poster-preview-wrap').style.display = 'none';

            // Navigate to the new series and open the chapter-select modal
            AppState.currentSeriesId = series.id;
            switchView('series-view', () => {
                loadSeries(series.id);
                // Give the series view a moment to render then open chapter picker
                setTimeout(() => openChapterSelectModal(series.id), 300);
            });
        } else {
            alert('Failed to create series');
        }
    } catch (err) {
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create Series';
    }
}

// Show "auto-fetch" hint when a URL is typed into the source URL field
function initAddSeriesModal() {
    const urlInput = document.getElementById('seriesSourceUrl');
    if (!urlInput) return;
    urlInput.addEventListener('input', function () {
        document.getElementById('poster-preview-wrap').style.display =
            this.value ? 'block' : 'none';
    });
}
