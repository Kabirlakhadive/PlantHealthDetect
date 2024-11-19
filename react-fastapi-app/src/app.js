import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0); // For the loading bar

  // Handle file selection via drag-and-drop
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Handle drag-and-drop events
  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
    accept: "image/*", // Only accept image files
  });

  // Handle prediction
  const handlePredict = async () => {
    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setProgress(0); // Reset progress when a new prediction starts
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      // Simulate loading progress
      let progressInterval = setInterval(() => {
        setProgress((oldProgress) => {
          if (oldProgress < 90) return oldProgress + 10;
          return oldProgress;
        });
      }, 500);

      const data = await response.json();
      clearInterval(progressInterval); // Clear the loading animation
      setResult(data);
      setProgress(100); // Set progress to 100 when done
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  // Check if the class contains 'healthy' for styling purposes
  const getClassBackgroundColor = (className) => {
    return className.toLowerCase().includes("healthy")
      ? "background-green"
      : "background-red";
  };

  return (
    <div className="App">
      <h1>Plant Disease Detection</h1>
      <div className="dropzone" {...getRootProps()}>
        <input {...getInputProps()} />
        {file && (
          <img src={URL.createObjectURL(file)} alt="Selected" className="image-preview" />
        )}
        <p className="dropzone-text">Drag & drop an image here, or click to select one</p>
      </div>
      <button onClick={handlePredict} className="predict-button">
        Predict
      </button>

      {loading && (
        <div>
          <p>Loading...</p>
          <progress value={progress} max={100}></progress>
        </div>
      )}

      {result && (
        <div className={`prediction-result ${getClassBackgroundColor(result.class)}`}>
          <h2>Prediction Result:</h2>
          <p>Class: {result.class}</p>
          <p>Confidence: {result.confidence.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
}

export default App;
