# OpenCV Vision Playground

![OpenCV Vision PlayGround](assets/open_cv_vision_playground.png)

A Streamlit-based image processing application built with OpenCV and Python. This project demonstrates a range of computer vision techniques such as face detection, object detection, document enhancement, edge detection, cartoon style transfer, pencil sketch conversion, blur analysis, OCR scanning, and facial emotion detection.

## 🚀 Features

- **Face Detection** using Haar cascade classifiers
- **Object Detection** using the MobileNet SSD model
- **Document Enhancer** for improved scanned document clarity
- **Edge Detection** with adjustable Canny thresholds
- **Cartoon Filter** for stylized image rendering
- **Pencil Sketch** transformation
- **Blur Detection** with image sharpness scoring
- **OCR Scanner** using Tesseract OCR
- **Facial Emotion Detection** to recognize emotion labels from detected faces

## 📁 Repository Structure

- `app.py` — Streamlit frontend and feature-routing logic
- `requirements.txt` — Python dependencies
- `models/` — pretrained models used for object detection and face detection
- `src/` — core computer vision modules
- `notebooks/` — example notebooks for additional OpenCV experiments
- `assets/`, `data/`, `outputs/` — supplemental images and outputs

### `src/` modules included

- `blur_detection.py`
- `cartoon_filter.py`
- `document_enhancer.py`
- `edge_detection.py`
- `face_recognition.py`
- `facial_emotion_detection.py`
- `object_detection.py`
- `ocr_scanner.py`
- `pencil_sketch.py`
- `template_matching.py`
- `utils.py`

## ⚙️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- Streamlit
- Pillow
- imutils
- matplotlib
- scikit-image
- pytesseract

## ✅ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ethicalanp/OpencvApplications.git
   cd OpencvApplications
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR for OCR functionality:
   - Download the Windows installer from the Tesseract project site
   - Install and enable `Add to PATH`
   - Restart the terminal or your PC if needed

## ▶️ Run the application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 💡 Usage Notes

- Upload an image or use the webcam input mode.
- Select one of the feature cards in the sidebar.
- Adjust edge detection thresholds using the sidebar sliders.
- OCR requires a local Tesseract installation; the app includes guidance if Tesseract is not detected.
- Output images can be saved using the built-in save button.

## 🧩 Model and Asset Files

- `models/MobileNetSSD_deploy.prototxt`
- `models/MobileNetSSD_deploy.caffemodel`
- `models/haarcascade_frontalface_default.xml`

## 📌 Notes

- `ocr_scanner.py` attempts to auto-detect a Tesseract installation on Windows.
- The object detection module uses a confidence threshold of `0.4`.
- The face detection module uses OpenCV Haar cascades for real-time performance.

## 🙌 Improvements

Suggested future upgrades:
- Add video-stream processing
- Add custom model support for object detection
- Add performance profiling and batch processing options
- Add more advanced document correction and perspective transformation

---

Built with Python, OpenCV, and Streamlit for fast prototyping of computer vision demos.