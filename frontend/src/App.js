import { useState } from 'react';
import './App.css';

function App() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [lyrics, setLyrics] = useState("");
  const handleGenerateSong = async () => {
    setIsGenerating(true);
    
    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"}
      });
      const data = await res.json();
      setLyrics(data.text);
    } catch(err) {
      console.error(err.error);
      setLyrics("Error generating lyrics");
    }
    setIsGenerating(false); 
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Song Generator</h1>
        <p className="subtitle">Create your next hit song</p>
        <button 
          className="generate-button"
          onClick={handleGenerateSong}
          disabled={isGenerating}
        >
          {isGenerating ? 'Generating...' : 'Generate Song'}
        </button>
        <div className="lyrics-box">
          {lyrics}
        </div>
      </header>
    </div>
  );
}

export default App;
