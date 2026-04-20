/* ============================================================
   chapters.js — chapter list fetch + selection modal logic
   ============================================================ */

/**
 * Open the chapter-select modal for a given series.
 * Fetches the live chapter list from Newtoki and renders it as
 * a checkbox table. Called both after series creation and from
 * the "Scrape Newtoki" button in the series detail.
 *
 * @param {number} seriesId
 */
async function openChapterSelectModal(seriesId) {
    const modal = document.getElementById('chapterSelectModal');
    const body  = document.getElementById('chapter-select-body');
    const info  = document.getElementById('chapter-select-info');
    const btn   = document.getElementById('scrapeSelectedBtn');

    // Reset state
    body.innerHTML = '';
    info.textContent = 'Fetching chapter list from Newtoki…';
    btn.disabled = true;
    openModal('chapterSelectModal');

    try {
        const res  = await fetch(`${API}/series/${seriesId}/chapter-list`);
        if (!res.ok) {
            const err = await res.json();
            info.textContent = '⚠ ' + (err.detail || 'Failed to fetch chapter list');
            return;
        }
        const data = await res.json();
        const chapters = data.chapters || [];

        if (chapters.length === 0) {
            info.textContent = 'No chapters found on the series page.';
            return;
        }

        // Build table
        info.textContent = `${chapters.length} chapters found. Select chapters to scrape:`;
        body.innerHTML = _buildChapterTable(chapters);

        // Enable scrape button when at least one box is checked
        body.addEventListener('change', () => {
            const checked = body.querySelectorAll('input[type=checkbox]:checked');
            btn.disabled = checked.length === 0;
        });

        btn.disabled = true; // none selected yet

    } catch (err) {
        console.error(err);
        info.textContent = '⚠ Network error: ' + err.message;
    }
}

function _buildChapterTable(chapters) {
    const rows = chapters.map(ch => {
        const alreadyClass = ch.already_scraped ? 'chapter-row-done' : '';
        const checked      = '';   // start unchecked; user selects
        const disabledAttr = ch.already_scraped ? 'disabled title="Already scraped"' : '';
        const numDisplay   = ch.chapter_number != null ? ch.chapter_number : '—';
        return `
            <tr class="chapter-row ${alreadyClass}">
                <td class="ch-check">
                    <input type="checkbox" class="chapter-checkbox"
                        data-index="${ch.data_index}"
                        data-chapter-number="${ch.chapter_number ?? ''}"
                        data-title="${escapeHtml(ch.title)}"
                        ${checked} ${disabledAttr}>
                </td>
                <td class="ch-num">${numDisplay}</td>
                <td class="ch-title">${escapeHtml(ch.title)}</td>
                <td class="ch-status">${ch.already_scraped
                    ? '<span class="status-badge status-scraped">scraped</span>'
                    : ''}</td>
            </tr>`;
    }).join('');

    return `<table class="chapter-select-table">
        <thead>
            <tr>
                <th><input type="checkbox" id="selectAllChapters"
                    onchange="toggleSelectAll(this)" title="Select / deselect all"></th>
                <th>#</th>
                <th>Title</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>${rows}</tbody>
    </table>`;
}

function toggleSelectAll(masterCb) {
    const body = document.getElementById('chapter-select-body');
    body.querySelectorAll('input[type=checkbox]:not([disabled])').forEach(cb => {
        cb.checked = masterCb.checked;
    });
    const btn = document.getElementById('scrapeSelectedBtn');
    btn.disabled = !masterCb.checked ||
        body.querySelectorAll('input[type=checkbox]:checked').length === 0;
}

async function scrapeSelected() {
    const seriesId = AppState.currentSeriesId;
    const body     = document.getElementById('chapter-select-body');
    const btn      = document.getElementById('scrapeSelectedBtn');
    const checked  = [...body.querySelectorAll('input[type=checkbox]:checked')];

    if (checked.length === 0) return;

    btn.disabled = true;
    btn.textContent = `Queuing ${checked.length} chapters…`;

    let queued = 0;
    for (const cb of checked) {
        const fd = new FormData();
        fd.append('data_index',     cb.dataset.index);
        fd.append('chapter_number', cb.dataset.chapterNumber || cb.dataset.index);
        fd.append('title',          cb.dataset.title || '');

        try {
            const res = await fetch(`${API}/series/${seriesId}/scrape`, {
                method: 'POST', body: fd
            });
            if (res.ok) queued++;
        } catch (e) {
            console.error('Scrape queue error', e);
        }
    }

    closeModal('chapterSelectModal');
    btn.textContent = 'Scrape Selected';

    // Refresh series view if we're on it, else go to it
    if (document.getElementById('series-view').classList.contains('active')) {
        loadSeries(seriesId);
    } else {
        switchView('series-view', () => loadSeries(seriesId));
    }
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
