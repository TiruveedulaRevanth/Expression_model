import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

const App = () => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [expression, setExpression] = useState('None');
  const [confidence, setConfidence] = useState(0);
  let lastPredictionTime = 0;

  // Normalize landmarks relative to bounding box
  const normalizeLandmarks = (keypoints, box) => {
    const { xMin, yMin, width, height } = box;
    const centerX = xMin + width / 2;
    const centerY = yMin + height / 2;
    const scale = Math.max(width, height);
    return keypoints.map(({ x, y, z }) => ({
      x: (x - centerX) / scale,
      y: (y - centerY) / scale,
      z: z / scale,
    }));
  };

  // Predict expression
  const predictExpression = async (keypoints, box) => {
    try {
      console.log('Predicting expression with keypoints:', keypoints.length);
      if (!keypoints || keypoints.length !== 468) {
        throw new Error(`Expected 468 landmarks, got ${keypoints ? keypoints.length : 0}`);
      }

      const normalizedKeypoints = normalizeLandmarks(keypoints, box);
      const flatLandmarks = normalizedKeypoints.reduce((acc, { x, y, z }) => {
        return [...acc, x, y, z];
      }, []);
      console.log('Flat landmarks length:', flatLandmarks.length, 'Sample:', flatLandmarks.slice(0, 5));

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: flatLandmarks }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Prediction response:', data);
      if (data.error) {
        throw new Error(data.error);
      }

      setExpression(data.expression);
      setConfidence(data.confidence);
    } catch (error) {
      console.error('Prediction error:', error.message);
      setExpression('Error');
      setConfidence(0);
    }
  };

  useEffect(() => {
    const start = async () => {
      // Initialize TensorFlow.js
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TF.js Backend:', tf.getBackend());
      } catch (error) {
        console.error('WebGL failed, using CPU:', error);
        await tf.setBackend('cpu');
        await tf.ready();
      }

      // Create FaceMesh detector
      let detector;
      try {
        detector = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          {
            runtime: 'tfjs',
            refineLandmarks: false,
            maxFaces: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }
        );
        console.log('FaceMesh detector initialized successfully');
      } catch (error) {
        console.error('Failed to initialize FaceMesh detector:', error);
        setExpression('Detector Error');
        return;
      }

      // Start webcam
      const video = videoRef.current;
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
        });
        video.srcObject = stream;
        await video.play();
        console.log('Webcam stream started');
      } catch (error) {
        console.error('Webcam access failed:', error);
        setExpression('Webcam Error');
        return;
      }

      // Set canvas dimensions
      const canvas = canvasRef.current;
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('Canvas set to:', canvas.width, 'x', canvas.height);
      };

      // Ensure canvas has default dimensions
      if (!canvas.width || !canvas.height) {
        canvas.width = 640;
        canvas.height = 480;
        console.log('Set default canvas dimensions: 640 x 480');
      }

      // Process frames
      const draw = async () => {
        try {
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            console.error('Failed to get 2D context');
            setExpression('Canvas Error');
            requestAnimationFrame(draw);
            return;
          }

          // Clear canvas and draw mirrored video
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.save();
          ctx.scale(-1, 1); // Mirror video
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
          ctx.restore();

          const timestamp = performance.now();
          let faces;
          try {
            faces = await detector.estimateFaces(video, { flipHorizontal: false });
            console.log('Faces detected:', faces.length);
          } catch (error) {
            console.error('Face detection error:', error);
            setExpression('Detection Error');
            requestAnimationFrame(draw);
            return;
          }

          if (faces && faces.length > 0) {
            const face = faces[0];
            const keypoints = face.keypoints;
            const box = face.box;

            // Draw bounding box
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(box.xMin, box.yMin, box.width, box.height);

            // Draw landmarks
            keypoints.forEach(({ x, y }, index) => {
              if (isFinite(x) && isFinite(y)) {
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI); // Larger dots
                ctx.fillStyle = 'lime';
                ctx.fill();
              } else {
                console.warn(`Invalid landmark ${index}: x=${x}, y=${y}`);
              }
            });

            // Predict every 2 seconds
            if (timestamp - lastPredictionTime >= 2000) {
              lastPredictionTime = timestamp;
              await predictExpression(keypoints, box);
            }
          } else {
            console.log('No face detected');
            setExpression('No face detected');
            setConfidence(0);
            // Draw placeholder text
            ctx.fillStyle = 'white';
            ctx.font = '20px Arial';
            ctx.fillText('No face detected - Please center your face', 20, 50);
          }
        } catch (error) {
          console.error('Draw loop error:', error);
          setExpression('Render Error');
        }
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
        console.log('Webcam stream stopped');
      }
    };
  }, []);

  return (
    <div
      style={{
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
      }}
    >
      <h1 style={{ marginBottom: '20px', fontSize: '2rem', fontWeight: 'bold' }}>
        Real-Time FaceMesh Detection
      </h1>
      <div
        style={{
          position: 'relative',
          width: '640px',
          height: '480px',
          borderRadius: '16px',
          overflow: 'visible',
          boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '640px',
            height: '480px',
            zIndex: 1,
          }}
        />
        <video
          ref={videoRef}
          style={{
            display: 'none', // Hide video, show canvas
            position: 'absolute',
            top: 0,
            left: 0,
            width: '640px',
            height: '480px',
            zIndex: 0,
          }}
        />
      </div>
      <h2 style={{ marginTop: '20px', fontSize: '1.5rem' }}>
        Predicted Expression: {expression} {confidence > 0 ? `(${Math.round(confidence * 100)}%)` : ''}
      </h2>
    </div>
  );
};

export default App;