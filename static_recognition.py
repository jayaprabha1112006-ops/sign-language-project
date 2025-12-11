import tensorflow as tf
import numpy as np
import cv2
import os
from tkinter import Tk, Toplevel, Button, filedialog, Label, Entry, Frame
from PIL import Image, ImageTk

MODEL_PATH = r"C:\Users\jayap\OneDrive\Desktop\mini project\sign-language-project\sign_model.h5"
DATASET_DIR = r"C:\Users\jayap\OneDrive\Desktop\mini project\sign-language-project\asl_dataset"
IMAGE_SIZE = (224, 224)

# Loading model
model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(os.listdir(DATASET_DIR))


#  STATIC IMAGE PREDICTOR 
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img, verbose=0)
    predicted_label = labels[np.argmax(preds)]
    confidence = np.max(preds)

    original = cv2.imread(image_path)
    cv2.putText(original, f"{predicted_label.upper()} ({confidence*100:.2f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def choose_image_and_predict():
    file_path = filedialog.askopenfilename(
        title="Choose a hand gesture image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        predict_image(file_path)



def show_word_visualizer():
    vis_window = Toplevel()   
    vis_window.title("ASL Word Visualizer")

    entry = Entry(vis_window, font=("Arial", 16))
    entry.pack(pady=10)

    frame = Frame(vis_window)
    frame.pack(pady=10)
    image_refs = []

    def show_signs():
        nonlocal image_refs
        for widget in frame.winfo_children():
            widget.destroy()
        image_refs = []

        word = entry.get().lower()
        for letter in word:
            if letter == " ":
                lbl = Label(frame, text=" ", width=5)  
                lbl.pack(side="left", padx=10)
                continue
            letter_path = os.path.join(DATASET_DIR, letter.upper())
            if os.path.isdir(letter_path):
                
                sample_img = os.path.join(letter_path, os.listdir(letter_path)[0])
                img = cv2.imread(sample_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (100, 100))
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img_pil)

                lbl = Label(frame, image=img_tk, borderwidth=2, relief="solid")
                lbl.image = img_tk
                lbl.pack(side="left", padx=5)

                image_refs.append(img_tk)

    btn = Button(vis_window, text="Show Signs", command=show_signs, font=("Arial", 14))
    btn.pack(pady=5)



def main_menu():
    root = Tk()
    root.title("ASL App - Choose Mode")
    root.geometry("400x250")

    Label(root, text="Static input/output", font=("Arial", 18, "bold")).pack(pady=20)

    Button(root, text="🖼 Predict from Hand Sign Image",
           command=choose_image_and_predict, font=("Arial", 14), width=30).pack(pady=10)

    Button(root, text="⌨️ Type Text → Show Signs",
           command=show_word_visualizer, font=("Arial", 14), width=30).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main_menu()