(() => {
  const root = document.querySelector("[data-camera-root]");
  if (!root) return;

  const form = root.closest("form");
  const fileInput = root.querySelector("#image");
  const fileLabel = root.querySelector("[data-file-label]");
  const uploadPanel = root.querySelector("[data-upload-panel]");
  const cameraPanel = root.querySelector("[data-camera-panel]");
  const cameraImage = root.querySelector("#camera_image");
  const video = root.querySelector("#camera-preview");
  const canvas = root.querySelector("#camera-canvas");
  const startButton = root.querySelector("[data-camera-start]");
  const captureButton = root.querySelector("[data-camera-capture]");
  const status = root.querySelector("[data-camera-status]");
  const tabs = Array.from(root.querySelectorAll("[data-input-mode]"));
  const submitButton = form.querySelector("button[type='submit']");
  let stream = null;

  const setStatus = (message) => {
    status.textContent = message;
  };

  const setMode = (mode) => {
    const cameraMode = mode === "camera";
    uploadPanel.hidden = cameraMode;
    cameraPanel.hidden = !cameraMode;
    fileInput.required = !cameraMode;
    tabs.forEach((tab) => {
      const active = tab.dataset.inputMode === mode;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    if (cameraMode) {
      void startCamera();
    } else {
      stopCamera();
      cameraImage.value = "";
      setStatus("");
    }
  };

  const stopCamera = () => {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    captureButton.disabled = true;
  };

  const startCamera = async () => {
    if (stream) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("このブラウザではカメラを利用できません。");
      return;
    }

    setStatus("カメラを起動しています。");
    startButton.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      captureButton.disabled = false;
      setStatus("");
    } catch (error) {
      stream = null;
      captureButton.disabled = true;
      setStatus("カメラを起動できませんでした。");
    } finally {
      startButton.disabled = false;
    }
  };

  const submitCapturedImage = () => {
    if (!video.videoWidth || !video.videoHeight) {
      setStatus("カメラ映像を取得できていません。");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { alpha: false });
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    cameraImage.value = canvas.toDataURL("image/jpeg", 0.94);
    fileInput.value = "";
    fileLabel.textContent = "画像を選択";
    submitButton.disabled = false;
    form.requestSubmit(submitButton);
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.inputMode));
  });

  fileInput.addEventListener("change", () => {
    cameraImage.value = "";
    fileLabel.textContent = fileInput.files?.[0]?.name || "画像を選択";
  });

  startButton.addEventListener("click", () => {
    void startCamera();
  });

  captureButton.addEventListener("click", submitCapturedImage);
  window.addEventListener("beforeunload", stopCamera);
})();
