/* ============================================================
   app.js — bootstrap: wire up event listeners and init
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // Initial view
    loadLibrary();

    // Add Series modal — show auto-fetch hint when URL is typed
    const urlInput = document.getElementById('seriesSourceUrl');
    if (urlInput) {
        urlInput.addEventListener('input', function () {
            document.getElementById('poster-preview-wrap').style.display =
                this.value ? 'block' : 'none';
        });
    }
});
