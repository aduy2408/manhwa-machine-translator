import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [translations, setTranslations] = useState({});

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
    setResults([]);
    setCurrentImageIndex(0);
  };

  const handleProcessImages = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select images');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios.post(`${API_BASE}/batch-process`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResults(response.data.results);
      setTranslations(response.data.translations);
      setCurrentImageIndex(0);
    } catch (error) {
      console.error('Error processing images:', error);
      alert('Error processing images: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleProcessSingleImage = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE}/process`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResults([response.data]);
      setTranslations(response.data.translations);
      setCurrentImageIndex(0);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing image: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const currentResult = results[currentImageIndex];

  return (
    <div className="app">
      <div className="container">
        <div className="sidebar">
          <div className="upload-section">
            <h3>Upload Images</h3>
            
            <div className="upload-option">
              <label className="upload-label">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleProcessSingleImage}
                  disabled={loading}
                />
                <span>Single Image</span>
              </label>
            </div>

            <div className="upload-option">
              <label className="upload-label">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileSelect}
                  disabled={loading}
                />
                <span>Multiple Images</span>
              </label>
            </div>

            {selectedFiles.length > 0 && (
              <div className="file-list">
                <h4>Selected Files ({selectedFiles.length})</h4>
                <ul>
                  {selectedFiles.map((file, idx) => (
                    <li key={idx}>{file.name}</li>
                  ))}
                </ul>
                <button
                  onClick={handleProcessImages}
                  disabled={loading}
                  className="process-btn"
                >
                  {loading ? 'Processing...' : 'Process All'}
                </button>
              </div>
            )}
          </div>

          {results.length > 0 && (
            <div className="results-section">
              <h3>Results ({results.length})</h3>
              <div className="image-nav">
                <button
                  onClick={() => setCurrentImageIndex(Math.max(0, currentImageIndex - 1))}
                  disabled={currentImageIndex === 0}
                >
                  ← Prev
                </button>
                <span className="image-counter">
                  {currentImageIndex + 1} / {results.length}
                </span>
                <button
                  onClick={() => setCurrentImageIndex(Math.min(results.length - 1, currentImageIndex + 1))}
                  disabled={currentImageIndex === results.length - 1}
                >
                  Next →
                </button>
              </div>

              {currentResult && (
                <div className="detected-texts">
                  <h4>Detected Texts</h4>
                  <ul>
                    {currentResult.detected_texts.map((item, idx) => (
                      <li key={idx}>
                        <span className="korean">{item.text}</span>
                        <span className="translation">
                          {translations[item.text] || '...'}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="preview">
          {currentResult ? (
            <div className="image-container">
              <img src={currentResult.image} alt="Processed" />
            </div>
          ) : (
            <div className="placeholder">
              <p>Select and upload images to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
