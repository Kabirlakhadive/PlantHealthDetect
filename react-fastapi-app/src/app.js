import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handlePredict = async () => {
    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Plant Disease Detection</h1>
      <input type="file" onChange={handleFileChange} accept="image/*" />
      <button onClick={handlePredict}>Predict</button>
      {loading && <p>Loading...</p>}
      {result && (
        <div>
          <h2>Prediction Result:</h2>
          <p>Class: {result.class}</p>
          <p>Confidence: {result.confidence.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
}

export default App;
