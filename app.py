<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CAPTCHA Solver Test (WebSocket)</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      padding: 20px;
    }
    input, button {
      margin-top: 10px;
    }
    img {
      max-width: 300px;
      display: block;
      margin-top: 10px;
    }
  </style>
</head>
<body>

  <h2>🔐 CAPTCHA Solver WebSocket</h2>
  <input type="file" id="imgInput" accept="image/*">
  <button onclick="sendImage()">🔎 Predict</button>

  <img id="preview" src="" alt="Preview" />
  <p><strong>📝 Prediction:</strong> <span id="result">--</span></p>

  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <script>
    const socket = io("https://captcha-solver-production-6bb0.up.railway.app", {
      transports: ["websocket"]
    });

    let base64Image = "";

    document.getElementById("imgInput").addEventListener("change", function(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(e) {
        base64Image = e.target.result;
        document.getElementById("preview").src = base64Image;
      };
      reader.readAsDataURL(file);
    });

    function sendImage() {
      if (!base64Image) {
        alert("📷 الرجاء اختيار صورة أولاً");
        return;
      }
      document.getElementById("result").innerText = "⏳ جاري التنبؤ...";
      socket.emit("predict", { image: base64Image });
    }

    socket.on("connect", () => {
      console.log("✅ WebSocket connected");
    });

    socket.on("disconnect", () => {
      console.log("❌ WebSocket disconnected");
    });

    socket.on("result", (data) => {
      if (data.error) {
        document.getElementById("result").innerText = "❌ خطأ: " + data.error;
      } else {
        document.getElementById("result").innerText = `${data.text} ⏱ ${data.time}s`;
      }
    });
  </script>

</body>
</html>
