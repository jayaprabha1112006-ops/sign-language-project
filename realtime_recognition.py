import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import time
from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk
import threading
from collections import deque

MODEL_PATH = 'sign_model.h5'
DATASET_DIR = 'asl_dataset'
IMAGE_SIZE = (224, 224)
CONF_THRESHOLD = 0.9   
DEBOUNCE_INTERVAL = 1.0
SMOOTH_WINDOW = 5

# LOADING MODEL & LABELS 
model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(os.listdir(DATASET_DIR))
label_map = {i: lbl for i, lbl in enumerate(labels)}

# GUI
window = Tk()
window.title("🤟 ASL Predictor — Stable")
window.geometry("820x760")
window.configure(bg="#1f2833")

video_label = Label(window, bg="#0b0f1a")
video_label.pack(pady=8)

info_frame = Frame(window, bg="#1f2833")
info_frame.pack(pady=4, fill="x")

result_text = Label(info_frame, text="Initializing...",
                    font=("Helvetica", 14, "bold"),
                    fg="#66fcf1", bg="#1f2833")
result_text.grid(row=0, column=0, sticky="w", padx=6)

confidence_label = Label(info_frame, text="Confidence: --",
                         font=("Helvetica", 12),
                         fg="#c5c6c7", bg="#1f2833")
confidence_label.grid(row=1, column=0, sticky="w", padx=6)

final_word_label = Label(window, text="Final Word: ",
                         font=("Helvetica", 24, "bold"),
                         fg="#45a29e", bg="#1f2833")
final_word_label.pack(pady=6)

btn_frame = Frame(window, bg="#1f2833")
btn_frame.pack(pady=6)

accumulated_letters = []
pred_buffer = deque(maxlen=SMOOTH_WINDOW)
last_pred_time = 0
prediction_started = False


def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def update_final_word_label():
    final_word_label.config(text="Final Word: " + ''.join(accumulated_letters))

def clear_word():
    accumulated_letters.clear()
    update_final_word_label()
    result_text.config(text="Cleared word", fg="#66fcf1")

def clear_last():
    if accumulated_letters:
        accumulated_letters.pop()
        update_final_word_label()
        result_text.config(text="Removed last letter", fg="#c5c6c7")

def quit_app():
    window.quit()
    window.destroy()
    sys.exit(0)

# ==== PREDICTION LOOP ====
def start_prediction():
    global prediction_started
    if prediction_started:
        return
    prediction_started = True
    threading.Thread(target=update_frame, daemon=True).start()

def update_frame():
    global last_pred_time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        
        x1, y1, x2, y2 = 100, 100, 400, 400
        roi = frame[y1:y2, x1:x2]

        # blurring
        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        display_frame = blurred.copy()
        display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        #ROI box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (32, 255, 178), 3)

        # Preprocess and  Predict
        inp = preprocess_frame(roi)
        preds = model.predict(inp, verbose=0)[0]
        top_conf = float(np.max(preds))
        pred_idx = int(np.argmax(preds))
        predicted_label = label_map[pred_idx]

        # Add to smoothing buffer
        pred_buffer.append(predicted_label)
        stable_pred = max(set(pred_buffer), key=pred_buffer.count)

        if top_conf >= CONF_THRESHOLD:
            current_time = time.time()
            if current_time - last_pred_time > DEBOUNCE_INTERVAL:
                accumulated_letters.append(stable_pred)
                update_final_word_label()
                last_pred_time = current_time
            result_text.config(text=f"Prediction: {stable_pred}", fg="#45a29e")
            confidence_label.config(text=f"Confidence: {top_conf*100:.1f}%")
        else:
            result_text.config(text="Low confidence", fg="#dcdde1")
            confidence_label.config(text=f"Confidence: {top_conf*100:.1f}%")

        # Renders frame
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

    cap.release()
    cv2.destroyAllWindows()

# ==== Buttons ====
Button(btn_frame, text="▶️ Start", font=("Arial", 12),
       bg="#66fcf1", command=start_prediction).grid(row=0, column=0, padx=8)

Button(btn_frame, text="🧹 Clear Word (C)", font=("Arial", 12),
       bg="#ffb86c", command=clear_word).grid(row=0, column=1, padx=8)

Button(btn_frame, text="↩️ Clear Last (L)", font=("Arial", 12),
       bg="#f1a6ff", command=clear_last).grid(row=0, column=2, padx=8)

Button(btn_frame, text="🚪 Quit", font=("Arial", 12),
       bg="#ff5555", command=quit_app).grid(row=0, column=3, padx=8)


# ==== MAIN ====
if __name__ == "__main__":
    start_prediction()
    window.mainloop()
