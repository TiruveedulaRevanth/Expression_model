import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

const App = () => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [expression, setExpression] = useState('None'); // State for predicted expression
  let lastPredictionTime = 0;

  // Function to send landmarks to backend
  const predictExpression = async (keypoints) => {
    try {
      // Log landmark count
      console.log('Landmarks count:', keypoints.length); // Should be 468
      // Validate keypoints (expect 468 landmarks with x, y, z)
      if (!keypoints || keypoints.length !== 468) {
        throw new Error(`Expected 468 landmarks, got ${keypoints ? keypoints.length : 0}`);
      }

      // Flatten landmarks to 1D array (468 * 3 = 1404 values: x, y, z)
      const flatLandmarks = keypoints.reduce((acc, landmark) => {
        return [...acc, landmark.x, landmark.y, landmark.z];
      }, []);
      console.log('Flat landmarks length:', flatLandmarks.length); // Should be 1404

      // Send POST request to backend
      const response = await fetch('http://localhost:5000/predict', { // Updated port
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: flatLandmarks }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      setExpression(data.expression); // Update UI with predicted expression
    } catch (error) {
      console.error('Prediction error:', error.message);
      setExpression('Error');
    }
  };

  useEffect(() => {
    const start = async () => {
      // Initialize TensorFlow.js backend
      await tf.setBackend('webgl');
      await tf.ready();

      // Create FaceMesh detector with adjusted settings
      const detector = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        { 
          runtime: 'tfjs', 
          refineLandmarks: false, 
          maxFaces: 1,
          minDetectionConfidence: 0.5, // Lowered for sensitivity
          minTrackingConfidence: 0.5
        }
      );

      // Start webcam
      const video = videoRef.current;
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user' // Ensure front-facing camera
        }
      });
      video.srcObject = stream;
      await video.play();

      // Set canvas dimensions
      const canvas = canvasRef.current;
      canvas.width = 640;
      canvas.height = 480;

      // Process frames
      const draw = async () => {
        const timestamp = performance.now();
        console.log('Frame timestamp:', timestamp);

        let faces;
        try {
          faces = await detector.estimateFaces(video, { flipHorizontal: false }); // No landmark flipping
        } catch (error) {
          console.error('Face detection error:', error);
          setExpression('Detection Error');
          requestAnimationFrame(draw);
          return;
        }

        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();

        // Flip canvas to correct mirrored video feed
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);

        if (faces && faces.length > 0) {
          const face = faces[0]; // Process first face
          const keypoints = face.keypoints;

          // Draw landmarks without adjustment (test original coordinates)
          keypoints.forEach(({ x, y }) => {
            ctx.beginPath();
            ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
            ctx.fillStyle = 'lime';
            ctx.fill();
          });

          // Predict expression every 3000ms (3 seconds)
          if (timestamp - lastPredictionTime >= 3000) {
            lastPredictionTime = timestamp;
            await predictExpression(keypoints); // Use original keypoints for backend
          }
        } else {
          console.log('No face detected');
          setExpression('No face detected');
        }

        ctx.restore();
        requestAnimationFrame(draw);
      };

      draw();
    };

    start();

    // Cleanup
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div style={{
      height: '100vh',
      margin: 0,
      padding: 0,
      background: 'linear-gradient(to right, #1e3c72, #2a5298)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Arial, sans-serif',
      color: '#fff',
    }}>
      <h1 style={{ marginBottom: '20px', fontSize: '2rem', fontWeight: 'bold' }}>
        Real-Time FaceMesh Detection
      </h1>
      <div style={{
        position: 'relative',
        width: 640,
        height: 480,
        borderRadius: '16px',
        overflow: 'hidden',
        boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
        backdropFilter: 'blur(8px)',
        background: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <canvas ref={canvasRef} width={640} height={480} style={{
          position: 'absolute',
          top: 0,
          left: 0
        }} />
        <video ref={videoRef} width={640} height={480} style={{
          display: 'none',
        }} />
      </div>
      <h2 style={{ marginTop: '20px', fontSize: '1.5rem' }}>
        Predicted Expression: {expression}
      </h2>
    </div>
  );
};

export default App;