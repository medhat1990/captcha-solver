import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask
from flask_socketio import SocketIO, emit
from PIL import Image, ImageEnhance, ImageSequence
from io import BytesIO
import base64
import time

# تحميل النموذج
session = ort.InferenceSession("char_cnn_model.onnx")

# ✅ تحميل التصنيفات تلقائياً من أسماء المجلدات داخل مجلد characters
class_labels = sorted([
    name for name in os.listdir("characters")
    if os.path.isdir(os.path.join("characters", name))
])

def preprocess_char(img, size=64):
    img = cv2.resize(img, (size, size))
    img = img.astype("float32") / 255.0
    img = np.stack([img] * 3, axis=-1)
    return np.expand_dims(img, axis=0)

def convert_gif_to_static(image):
    merged_array = np.ones((image.height, image.width), dtype=np.uint8) * 255
    for frame in ImageSequence.Iterator(image):
        frame_array = np.array(frame.convert("L"))
        merged_array = np.minimum(merged_array, frame_array)
    final_image = Image.fromarray(merged_array)
    final_image = ImageEnhance.Contrast(final_image).enhance(2.0)
    final_image = final_image.point(lambda p: 0 if p < 180 else 255)
    return final_image.convert("L")

def predict_captcha_from_pil(image):
    image = np.array(image)
    _, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > 30]
    sorted_cnts = sorted(filtered, key=lambda c: cv2.boundingRect(c)[0])[:5]

    label_predicted = ""
    for cnt in sorted_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y = max(x - 2, 0), max(y - 2, 0)
        w, h = min(w + 4, image.shape[1] - x), min(h + 4, image.shape[0] - y)
        char_img = image[y:y+h, x:x+w]
        processed = preprocess_char(char_img)
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: processed.astype(np.float32)})[0]
        label = class_labels[np.argmax(pred)]
        label_predicted += label

    return label_predicted

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "✅ WebSocket CAPTCHA Solver is running!"

@socketio.on("predict")
def handle_prediction(data):
    try:
        start_time = time.time()
        base64_image = data.get("image", "")
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        image_bytes = base64.b64decode(base64_image + "=" * (-len(base64_image) % 4))
        image = Image.open(BytesIO(image_bytes))
        if getattr(image, "is_animated", False):
            image = convert_gif_to_static(image)
        else:
            image = image.convert("L")

        result = predict_captcha_from_pil(image)
        elapsed = round(time.time() - start_time, 3)
        emit("result", {"text": result, "time": elapsed})
        print(f"✅ Prediction: {result} in {elapsed}s")
    except Exception as e:
        print(f"❌ Error: {e}")
        emit("result", {"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, host="0.0.0.0", port=port)
