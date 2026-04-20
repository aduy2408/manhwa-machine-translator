/* ============================================================
   config.js — shared constants and app state
   ============================================================ */

const API = 'http://localhost:5000/api';

// Placeholder image (SVG: "No Cover")
const PLACEHOLDER_IMG =
    'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iNDUwIj' +
    '48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMWUyOTNiIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGRvbWluYW50LWJhc2' +
    'VsaW5lPSJtaWRkbGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM2NDc0OGIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpem' +
    'U9IjI0Ij5ObyBDb3ZlcjwvdGV4dD48L3N2Zz4=';

// Mutable app state (written by modules, read cross-module via window.AppState)
window.AppState = {
    currentSeriesId: null,
    currentSeriesSourceUrl: null,
    currentReaderMode: 'raw',  // 'raw' | 'translated'
    readerPages: [],
    showingTranslation: true,
    pollIntervals: {},
};
