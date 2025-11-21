console.log("Frontend script loaded.");

const startBtn = document.getElementById("start-rec");
const stopBtn = document.getElementById("stop-rec");
const statusEl = document.getElementById("rec-status");
const playback = document.getElementById("playback");

let mediaRecorder;
let chunks = [];

if (startBtn && stopBtn) {
  startBtn.addEventListener("click", async () => {
    chunks = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        playback.src = URL.createObjectURL(blob);
        playback.style.display = "block";

        // Build form data
        const formData = new FormData();
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        formData.append("audio", file);

        // 🔥 Send mode to backend:
        // If there is a selected <input name="mode"> use that, else default to "fast"
        const modeInput = document.querySelector('input[name="mode"]:checked');
        if (modeInput) {
          formData.append("mode", modeInput.value);
        } else {
          formData.append("mode", "fast"); // live recording → usually CNN (fast)
        }

        statusEl.textContent = "Uploading & analyzing...";

        fetch("/predict_audio", {
          method: "POST",
          body: formData,
        })
          .then((resp) => resp.text())
          .then((html) => {
            // Replace current page with result page
            document.open();
            document.write(html);
            document.close();
          })
          .catch((err) => {
            console.error(err);
            statusEl.textContent = "Error during analysis.";
          });
      };

      mediaRecorder.start();
      statusEl.textContent = "Recording...";
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Cannot access microphone.";
    }
  });

  stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
      statusEl.textContent = "Processing recording...";
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  });
}
