import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageSequence
import base64
from io import BytesIO
import time
import uuid

# تحميل نموذج ONNX
onnx_model_path = "char_cnn_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# التصنيفات (31 رمز بدون full_images)
class_labels = ['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                'W', 'X', 'Y', 'Z']

def preprocess_char(img, size=64):
    img = cv2.resize(img, (size, size))
    img = img.astype("float32") / 255.0
    img = np.stack([img] * 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def convert_gif_to_static(image):
    merged_array = np.ones((image.height, image.width), dtype=np.uint8) * 255
    for frame in ImageSequence.Iterator(image):
        frame_array = np.array(frame.convert("L"))
        merged_array = np.minimum(merged_array, frame_array)
    final_image = Image.fromarray(merged_array)
    final_image = ImageEnhance.Contrast(final_image).enhance(2.0)
    threshold = 180
    final_image = final_image.point(lambda p: 0 if p < threshold else 255)
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

# إعداد خادم Flask
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        data = request.json
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data + '=' * (-len(image_data) % 4))
        image = Image.open(BytesIO(image_bytes))
        request_id = str(uuid.uuid4())

        if getattr(image, "is_animated", False):
            image = convert_gif_to_static(image)
        else:
            image = image.convert("L")

        result = predict_captcha_from_pil(image)
        elapsed_time = time.time() - start_time

        print(f"📝 Prediction: {result} | ⏱️ {elapsed_time:.3f}s | 🔑 Request ID: {request_id}")
        return jsonify({"text": result, "time": round(elapsed_time, 3), "request_id": request_id})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "✅ Model is up and running!"

# ⬅️ هذا السطر هو المفتاح للتشغيل على Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
