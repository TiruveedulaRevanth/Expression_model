import React, { useRef, useEffect, useState } from 'react';

const BACKEND_URL = 'http://localhost:5000';

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [expression, setExpression] = useState('None');
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState(null);
  let lastPredictionTime = 0;

  const predictExpression = async (keypoints) => {
    setError(null);
    try {
      if (!keypoints || keypoints.length !== 468) {
        throw new Error(`Expected 468 landmarks, got ${keypoints?.length || 0}`);
      }

      const flat = keypoints.flatMap(({ x, y, z }) => [x, y, z]);
      if (flat.length !== 1404) {
        throw new Error(`Expected 1404 values after flattening, got ${flat.length}`);
      }

      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: flat }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error: ${response.status}`);
      }

      const data = await response.json();
      setExpression(data.expression || 'Unknown');
      setConfidence(data.confidence || 0);
    } catch (err) {
      console.error('Prediction Error:', err.message);
      setError(err.message);
      setExpression('Error');
      setConfidence(0);
    }
  };

  useEffect(() => {
    let isMounted = true;

    const loadFaceMesh = async () => {
      if (!isMounted) return;

      try {
        const script = document.createElement('script');
        script.src = '/mediapipe/face_mesh.js';
        script.async = true;
        document.body.appendChild(script);

        await new Promise((resolve, reject) => {
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load FaceMesh script'));
        });

        if (!window.FaceMesh) {
          throw new Error('window.FaceMesh is undefined after script load');
        }

        const faceMesh = new window.FaceMesh({
          locateFile: (file) => `/mediapipe/${file}`,
        });

        faceMesh.setOptions({
          maxNumFaces: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
          selfieMode: false, // non-mirrored
        });

        faceMesh.onResults((results) => {
          if (!isMounted) return;

          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas?.getContext('2d');

          if (!ctx || !canvas || !video || video.videoWidth === 0 || video.videoHeight === 0) {
            return;
          }

          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

// Draw FaceMesh Landmarks
if (results.multiFaceLandmarks?.length > 0) {
  const keypoints = results.multiFaceLandmarks[0];

  ctx.fillStyle = 'aqua';
  ctx.strokeStyle = 'aqua';
  ctx.lineWidth = 1;

  for (let i = 0; i < keypoints.length; i++) {
    const x = keypoints[i].x * canvas.width;
    const y = keypoints[i].y * canvas.height;
    ctx.beginPath();
    ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  const now = performance.now();
  if (now - lastPredictionTime > 250) {
    lastPredictionTime = now;
    predictExpression(keypoints);
  }
}


          if (results.multiFaceLandmarks?.length > 0) {
            const keypoints = results.multiFaceLandmarks[0];

            const now = performance.now();
            if (now - lastPredictionTime > 250) {
              lastPredictionTime = now;
              predictExpression(keypoints);
            }
          } else {
            setExpression('No Face');
            setConfidence(0);
            setError(null);
          }
        });

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 },
          },
        });

        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const video = videoRef.current;
        video.srcObject = stream;
        await video.play();

        video.onloadedmetadata = () => {
          if (!isMounted) return;
          const canvas = canvasRef.current;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        };

        const processFrame = async () => {
          if (!isMounted) return;
          if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
            try {
              await faceMesh.send({ image: videoRef.current });
            } catch (err) {
              console.error('FaceMesh processing error:', err);
              setError(`FaceMesh processing error: ${err.message}`);
            }
          }
          requestAnimationFrame(processFrame);
        };
        processFrame();
      } catch (err) {
        console.error('Initialization Error:', err);
        setError(`Initialization error: ${err.message}`);
      }
    };

    loadFaceMesh();

    return () => {
      isMounted = false;
      const stream = videoRef.current?.srcObject;
      stream?.getTracks()?.forEach((track) => track.stop());
    };
  }, []);

  return (
    <div
      style={{
        height: '100vh',
        background: 'linear-gradient(to right, #1e3c72, #2a5298)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontFamily: 'Arial, sans-serif',
        padding: '20px',
      }}
    >
      <h1 style={{ marginBottom: '1rem' }}>Facial Expression Recognition</h1>

      <div
        style={{
          position: 'relative',
          width: '640px',
          height: '480px',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
        }}
      >
        <video
          ref={videoRef}
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            zIndex: 1,
          }}
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />
      </div>

      <h2 style={{ marginTop: '20px' }}>
        Expression: {expression}{' '}
        {confidence > 0 ? `(${Math.round(confidence * 100)}%)` : ''}
      </h2>

      {error && (
        <p style={{ marginTop: '10px', color: '#ff5555' }}>Error: {error}</p>
      )}
    </div>
  );
};

export default App;
