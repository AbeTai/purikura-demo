(() => {
  const VISION_TASKS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
  const VISION_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
  const FACE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
  const OUTPUT_WIDTH = 960;
  const OUTPUT_HEIGHT = 1200;
  const OUTPUT_RATIO = OUTPUT_WIDTH / OUTPUT_HEIGHT;

  const root = document.querySelector("[data-camera-root]");
  if (!root) return;

  const form = root.closest("form");
  const fileInput = root.querySelector("input[type='file']");
  const fileLabel = root.querySelector("[data-file-label]");
  const uploadPanel = root.querySelector("[data-upload-panel]");
  const cameraPanel = root.querySelector("[data-camera-panel]");
  const cameraImage = root.querySelector("#camera_image");
  const cameraLandmarks = root.querySelector("#camera_landmarks");
  const video = root.querySelector("#camera-preview");
  const overlay = root.querySelector("#camera-overlay");
  const canvas = root.querySelector("#camera-canvas");
  const startButton = root.querySelector("[data-camera-start]");
  const captureButton = root.querySelector("[data-camera-capture]");
  const status = root.querySelector("[data-camera-status]");
  const tabs = Array.from(root.querySelectorAll("[data-input-mode]"));
  const submitButton = form?.querySelector("button[type='submit']");

  let stream = null;
  let faceLandmarker = null;
  let detectorReady = false;
  let detecting = false;
  let lastFaceBox = null;
  let lastFaceLandmarks = null;
  let lastVideoTime = -1;

  function setStatus(message) {
    if (status) status.textContent = message;
  }

  function setCaptureEnabled() {
    if (captureButton) {
      captureButton.disabled = !(stream && detectorReady && lastFaceBox);
    }
  }

  function setMode(mode) {
    const useCamera = mode === "camera";
    tabs.forEach((tab) => {
      const selected = tab.dataset.inputMode === mode;
      tab.classList.toggle("is-active", selected);
      tab.setAttribute("aria-selected", selected ? "true" : "false");
    });

    if (uploadPanel) uploadPanel.hidden = useCamera;
    if (cameraPanel) cameraPanel.hidden = !useCamera;
    if (fileInput) fileInput.required = !useCamera;

    if (useCamera) {
      startCamera();
    } else {
      stopCamera();
      if (cameraImage) cameraImage.value = "";
      if (cameraLandmarks) cameraLandmarks.value = "";
      setStatus("");
    }
  }

  async function ensureFaceLandmarker() {
    if (faceLandmarker) {
      detectorReady = true;
      return faceLandmarker;
    }

    setStatus("顔検出モデルを読み込んでいます。");
    const vision = await import(VISION_TASKS_URL);
    const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM_URL);
    faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: FACE_MODEL_URL,
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numFaces: 8,
    });
    detectorReady = true;
    return faceLandmarker;
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("このブラウザではカメラを利用できません。");
      return;
    }

    if (stream) {
      setCaptureEnabled();
      return;
    }

    lastFaceBox = null;
    lastFaceLandmarks = null;
    detectorReady = false;
    setCaptureEnabled();
    drawGuideOverlay();

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      await ensureFaceLandmarker();
      detecting = true;
      setStatus("顔を中央に入れてください。");
      requestAnimationFrame(detectLoop);
    } catch (error) {
      console.error(error);
      stopCamera();
      setStatus(detectorReady ? "カメラを起動できませんでした。" : "顔検出モデルを読み込めませんでした。");
    }
  }

  function stopCamera() {
    detecting = false;
    lastVideoTime = -1;
    lastFaceBox = null;
    lastFaceLandmarks = null;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    if (video) video.srcObject = null;
    clearOverlay();
    setCaptureEnabled();
  }

  function detectLoop() {
    if (!detecting || !stream || !faceLandmarker || !video.videoWidth || !video.videoHeight) {
      setCaptureEnabled();
      if (detecting) requestAnimationFrame(detectLoop);
      return;
    }

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      try {
        const result = faceLandmarker.detectForVideo(video, performance.now());
        const faces = result.faceLandmarks || [];
        if (faces.length > 0 && faces[0]?.length) {
          lastFaceBox = faceBoxFromLandmarkFaces(faces);
          lastFaceLandmarks = faces;
          drawFaceOverlay(lastFaceBox);
          setStatus("顔を検出しました。");
        } else {
          lastFaceBox = null;
          lastFaceLandmarks = null;
          drawGuideOverlay();
          setStatus("顔を中央に入れてください。");
        }
      } catch (error) {
        console.error(error);
        lastFaceBox = null;
        lastFaceLandmarks = null;
        drawGuideOverlay();
        setStatus("顔検出に失敗しました。顔を中央に入れてください。");
      }
      setCaptureEnabled();
    }

    requestAnimationFrame(detectLoop);
  }

  function faceBoxFromLandmarks(landmarks) {
    const xs = landmarks.map((point) => point.x).filter(Number.isFinite);
    const ys = landmarks.map((point) => point.y).filter(Number.isFinite);
    const xMin = clamp(Math.min(...xs), 0, 1);
    const xMax = clamp(Math.max(...xs), 0, 1);
    const yMin = clamp(Math.min(...ys), 0, 1);
    const yMax = clamp(Math.max(...ys), 0, 1);
    return {
      xMin,
      yMin,
      xMax,
      yMax,
      width: Math.max(0.01, xMax - xMin),
      height: Math.max(0.01, yMax - yMin),
      cx: (xMin + xMax) * 0.5,
      cy: (yMin + yMax) * 0.5,
    };
  }

  function submitCapturedImage() {
    if (!stream || !video.videoWidth || !video.videoHeight) {
      setStatus("カメラを起動してください。");
      return;
    }
    if (!lastFaceBox || !lastFaceLandmarks?.length) {
      setStatus("顔を検出してから撮影してください。");
      return;
    }

    const crop = cropFromFaceBox(lastFaceBox, video.videoWidth, video.videoHeight);
    canvas.width = OUTPUT_WIDTH;
    canvas.height = OUTPUT_HEIGHT;
    const context = canvas.getContext("2d", { alpha: false });
    context.save();
    context.translate(OUTPUT_WIDTH, 0);
    context.scale(-1, 1);
    context.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    context.restore();

    if (cameraImage) cameraImage.value = canvas.toDataURL("image/jpeg", 0.94);
    if (cameraLandmarks) {
      cameraLandmarks.value = JSON.stringify(
        lastFaceLandmarks.map((landmarks) => landmarksForOutputImage(landmarks, crop, video.videoWidth, video.videoHeight)),
      );
    }
    if (fileInput) fileInput.value = "";
    if (fileLabel) fileLabel.textContent = "画像を選択";
    form?.requestSubmit(submitButton || undefined);
  }

  function landmarksForOutputImage(landmarks, crop, videoWidth, videoHeight) {
    return landmarks.map((point) => {
      const rawX = point.x * videoWidth;
      const rawY = point.y * videoHeight;
      const croppedX = clamp((rawX - crop.x) / crop.width, -0.25, 1.25);
      const croppedY = clamp((rawY - crop.y) / crop.height, -0.25, 1.25);
      return {
        x: clamp(1 - croppedX, -0.25, 1.25),
        y: croppedY,
      };
    });
  }

  function cropFromFaceBox(box, videoWidth, videoHeight) {
    const faceW = box.width * videoWidth;
    const faceH = box.height * videoHeight;
    const faceCenterX = box.cx * videoWidth;
    const faceCenterY = box.cy * videoHeight;
    const cropH = Math.max(faceH * 3.1, (faceW / OUTPUT_RATIO) * 1.2, videoHeight * 0.78);
    const cropW = cropH * OUTPUT_RATIO;
    const centerY = faceCenterY + faceH * 0.58;
    return clampCropToVideo(faceCenterX - cropW * 0.5, centerY - cropH * 0.5, cropW, cropH, videoWidth, videoHeight);
  }

  function faceBoxFromLandmarkFaces(faces) {
    return faceBoxFromLandmarks(faces.flat());
  }

  function clampCropToVideo(x, y, width, height, videoWidth, videoHeight) {
    let cropWidth = Math.min(width, videoWidth);
    let cropHeight = cropWidth / OUTPUT_RATIO;
    if (cropHeight > videoHeight) {
      cropHeight = videoHeight;
      cropWidth = cropHeight * OUTPUT_RATIO;
    }
    return {
      x: clamp(x, 0, videoWidth - cropWidth),
      y: clamp(y, 0, videoHeight - cropHeight),
      width: cropWidth,
      height: cropHeight,
    };
  }

  function resizeOverlay() {
    const rect = overlay.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * scale));
    const height = Math.max(1, Math.round(rect.height * scale));
    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }
    return { width, height, scale };
  }

  function drawFaceOverlay(box) {
    if (!overlay) return;
    const { width, height } = resizeOverlay();
    const context = overlay.getContext("2d");
    context.clearRect(0, 0, width, height);

    const rectX = (1 - box.xMax) * width;
    const rectY = box.yMin * height;
    const rectW = box.width * width;
    const rectH = box.height * height;
    context.strokeStyle = "#15a88f";
    context.lineWidth = Math.max(2, width * 0.006);
    context.strokeRect(rectX, rectY, rectW, rectH);
  }

  function drawGuideOverlay() {
    if (!overlay) return;
    const { width, height } = resizeOverlay();
    const context = overlay.getContext("2d");
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "rgba(255, 255, 255, 0.86)";
    context.lineWidth = Math.max(2, width * 0.005);
    const guideH = height * 0.52;
    const guideW = guideH * 0.72;
    context.setLineDash([width * 0.04, width * 0.025]);
    context.strokeRect((width - guideW) * 0.5, height * 0.16, guideW, guideH);
    context.setLineDash([]);
  }

  function clearOverlay() {
    if (!overlay) return;
    const context = overlay.getContext("2d");
    context.clearRect(0, 0, overlay.width, overlay.height);
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.inputMode)));
  startButton?.addEventListener("click", startCamera);
  captureButton?.addEventListener("click", submitCapturedImage);
  fileInput?.addEventListener("change", () => {
    if (fileLabel) fileLabel.textContent = fileInput.files?.[0]?.name || "画像を選択";
    if (cameraImage) cameraImage.value = "";
    if (cameraLandmarks) cameraLandmarks.value = "";
  });
  window.addEventListener("resize", () => {
    if (lastFaceBox) {
      drawFaceOverlay(lastFaceBox);
    } else if (stream) {
      drawGuideOverlay();
    }
  });
  window.addEventListener("beforeunload", stopCamera);
})();
