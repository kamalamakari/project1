import tkinter as tk
import webbrowser
from tkinter import filedialog, Label, Button, Canvas, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pandas as pd
import pyttsx3
import text2speach as ts
# --- YOLOv8 Specific Imports ---
from ultralytics import YOLO
df = pd.read_csv('data.csv')
# ------------------------------

# --- YOLOv8 Model Loading ---
MODEL_PATH = 'best.pt'
try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"YOLOv8 model loaded from {MODEL_PATH}")
except Exception as e:
    messagebox.showerror("Model Error",
                         f"Error loading YOLOv8 model: {e}\nPlease ensure '{MODEL_PATH}' is in the same directory and 'ultralytics' is installed.")
    exit()

try:
    label_names_dict = \
        yolo_model.names
    label_names = [label_names_dict[i] for i in sorted(label_names_dict.keys())]
    if not label_names:
        raise ValueError("YOLO model class names could not be loaded or are empty.")
    print("Class names loaded:", label_names)
except Exception as e:
    messagebox.showerror("Label Error", f"Error getting class names from YOLO model: {e}")
    exit()

# Ensure the 'output' directory exists
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")



# ---------------------------------

class ImagePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Predictor (YOLOv8)")
        self.root.geometry("1100x750")
        self.root.config(bg="#f0f0f0")

        self.image_path = None
        self.yolo_model = yolo_model
        self.label_names = label_names

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)

        # --- Header ---
        self.header_frame = tk.Frame(root, bg="#000066", height=60)
        self.header_frame.pack(fill="x", pady=10)
        self.header_label = Label(self.header_frame, text="Fruit Freshness Detection using (YOLOv8)",
                                  font=("Arial", 24, "bold"), bg="#000066", fg="white")
        self.header_label.pack(pady=10)

        # --- Main Content Frame ---
        self.main_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Left side for image display and upload
        self.left_frame = tk.Frame(self.main_frame, bg="#ffffff", width=500)
        self.left_frame.pack(side="left", padx=20, pady=20, fill="y")
        self.left_frame.pack_propagate(False)

        # Image Canvas: 400x400
        self.image_canvas = Canvas(self.left_frame, width=400, height=400, bg="#e0e0e0", bd=2, relief="solid")
        self.image_canvas.pack(pady=10)
        self.placeholder_label = Label(self.image_canvas, text="No Image Selected", font=("Arial", 12), fg="gray",
                                       bg="#e0e0e0")
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")

        # Buttons placed using .place()
        self.upload_button = Button(self.left_frame, text="Upload Image", command=self.upload_image, width="20",
                                    font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
                                    activebackground="#1976D2", relief="raised", bd=3)
        self.upload_button.place(relx=0.10, rely=0.80)

        self.predict_button = Button(self.left_frame, text="Predict", command=self.predict_image, width="20",
                                     font=("Arial", 10, "bold"), bg="#FFC107", fg="black",
                                     activebackground="#FFA000", relief="raised", bd=3, state=tk.DISABLED)
        self.predict_button.place(relx=0.50, rely=0.80)

        self.close_button = Button(self.left_frame, text="Close",
                                   command=self.close_app,
                                   font=("Arial", 14, "bold"),
                                   bg="#F44336", fg="white",
                                   activebackground="#D32F2F",
                                   relief="raised", bd=3)
        self.close_button.pack(side="bottom", pady=20, ipadx=10, ipady=5)

        # Right side for results
        self.right_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

        self.result_label = Label(self.right_frame, text="Prediction Result:",
                                  font=("Arial", 16, "bold"), bg="#ffffff", fg="#333333")
        self.result_label.pack(pady=(0, 10), anchor="w")

        self.confidence_label = Label(self.right_frame, text="Confidence: N/A",
                                      font=("Arial", 14), bg="#ffffff", fg="#555555")
        self.confidence_label.pack(pady=(0, 20), anchor="w")

        # Matplotlib graph canvas: DECREASED SIZE (5, 3.5)
        self.fig, self.ax = plt.subplots(figsize=(5, 3.5))
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        self.ax.set_title("Detected Object Confidence")
        self.ax.set_ylabel("Confidence")
        self.ax.set_xlabel("Classes")
        self.ax.tick_params(axis='x', rotation=45)
        self.fig.tight_layout()



    def botsays(self,message):
        engine = pyttsx3.init()
        engine.say("the prediction is" + message)
        engine.runAndWait()



    def display_info(self, item):
        try:
            # Check if item exists in CSV
            if item in df['name'].values:
                row = df[df['name'] == item].iloc[0]
                infomessage = f"Fruit Name: {row['name']}\n\nFruit Info: {row['info']}"
                messagebox.showinfo("Recommendation", infomessage)
                #----botsay----------
                ts.botsay(infomessage)
            else:
                messagebox.showinfo("Information", f"No information available for {item}")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching information:\n{e}")



    def close_app(self):
        """Gracefully close the application."""
        self.speak_text("Closing the application. Goodbye!")
        self.root.after(1000, self.root.destroy)  # Wait 1 sec before closing

    def webcamlive(self):
        os.system("python liveWebcam_detection.py")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((400, 400), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.placeholder_label.destroy()
            self.predict_button.config(state=tk.NORMAL)
            self.result_label.config(text="Prediction Result:")
            self.confidence_label.config(text="Confidence: N/A")
            self.clear_graph()
            #self.speak_text("Image uploaded. Click Predict to get the result.")

    def clear_graph(self):
        self.ax.clear()
        self.ax.set_title("Detected Object Confidence")
        self.ax.set_ylabel("Confidence")
        self.ax.set_xlabel("Classes")
        self.ax.tick_params(axis='x', rotation=45)
        self.canvas_mpl.draw()



    def predict_image(self):
        if not self.image_path:
            #self.speak_text("Please upload an image first.")
            return

        #self.speak_text("Running YOLOv8 inference...")

        # 1. Run YOLOv8 Inference without saving to default 'runs' directory
        results = self.yolo_model.predict(source=self.image_path, save=False, conf=0.05, iou=0.5, verbose=False)

        # 2. Extract Results from the first image
        result = results[0]
        boxes = result.boxes

        predicted_label = "no image"
        confidence = 0.0
        probabilities = np.zeros(len(self.label_names))
        predicted_class_index = -1

        # Define the path where the annotated image will be saved
        output_image_path = os.path.join(OUTPUT_DIR, "result_img.jpg")

        if not boxes or len(boxes) == 0:
            # --- No objects detected ---
            self.result_label.config(text=f"Prediction Result: {predicted_label}")
            self.confidence_label.config(text=f"Confidence: N/A")
            #self.speak_text(f"No waste was detected in the image.")

            # Display the original image (as no annotations were made)
            img = Image.open(self.image_path)
            img = img.resize((400, 400), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.placeholder_label.destroy()

            # If no object detected, delete previous output image if it exists
            if os.path.exists(output_image_path):
                os.remove(output_image_path)



        else:
            # --- Object(s) detected ---
            # Find the prediction with the highest confidence
            max_conf_idx = boxes.conf.argmax().item()
            predicted_class_index = int(boxes.cls[max_conf_idx].item())
            confidence = boxes.conf[max_conf_idx].item() * 100
            predicted_label = self.label_names[predicted_class_index]
            print("predicted label:"+predicted_label)


            self.result_label.config(text=f"Prediction Result: {predicted_label} (Object Detected)")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            probabilities[predicted_class_index] = confidence / 100.0


            # 3. Get the annotated image data and save it


            try:
                # result.plot() returns a NumPy array (BGR format for OpenCV, usually)
                annotated_img_np = result.plot()

                # Convert BGR (OpenCV default) to RGB for PIL
                annotated_img_pil = Image.fromarray(annotated_img_np[..., ::-1])  # Reverses channels

                # Save the annotated image to the specified output path
                annotated_img_pil.save(output_image_path)
                print(f"Annotated image saved to: {output_image_path}")

                # Now load this saved image for display
                img_to_display = Image.open(output_image_path)
                img_to_display = img_to_display.resize((400, 400), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(img_to_display)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.placeholder_label.destroy()


            except Exception as e:
                print(f"Error saving or displaying annotated image: {e}. Displaying original.")
                # Fallback to original image if there's an issue saving/loading the annotated one
                img = Image.open(self.image_path)
                img = img.resize((400, 400), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)

            #self.speak_text(f"The detected object is {predicted_label} with {confidence:.2f} percent confidence.")

        # 4. Update the graph
        self.ax.clear()
        colors = ['#88CCEE'] * len(self.label_names)

        if predicted_class_index != -1 and predicted_class_index < len(self.label_names) and confidence > 0.0:
            colors[predicted_class_index] = '#CC6677'

        self.ax.bar(self.label_names, probabilities, color=colors)
        self.ax.set_title("Detected Object Confidence")
        self.ax.set_ylabel("Confidence")
        self.ax.set_xlabel("Classes")
        self.ax.tick_params(axis='x', rotation=45, labelsize=10)
        self.ax.set_ylim(0, 1)
        self.fig.tight_layout()
        self.canvas_mpl.draw()
        self.display_info(predicted_label)


    def speak_text(self, text):
        self.root.after(100, lambda: self._speak_thread_safe(text))

    def _speak_thread_safe(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.mainloop()