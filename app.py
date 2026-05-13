import streamlit as st 
from PIL import Image
import numpy as np 
import cv2 
from src.face_recognition import detect_faces
from src.object_detection import detect_objects
from src.document_enhancer import enhance_document
from src.edge_detection import detect_edges
from src.cartoon_filter import cartoon_filter
from src.pencil_sketch import pencil_sketch
from src.blur_detection import detect_blur
from src.ocr_scanner import perform_ocr, draw_text_boxes
from src.facial_emotion_detection import detect_emotions


st.set_page_config(
    page_title='Vision Playgorund',
    layout = "wide"
)

st.title("OpenCV Vision PlayGround")
st.sidebar.title("Features")

#SIDEBAR
feature = st.sidebar.radio(
    "Choose Feature",
    [
        "Face Detection",
        "Object Detection",
        "Document Enhancer",
        "Edge Detection",
        "Cartoon Filter",
        "Pencil Sketch",
        "Blur Detection",
        "OCR Scanner",
        "Facial Emotion Detection"
    ]
)
# IMAGE INPUT OPTIONS
input_option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Webcam"]
)

image = None

#IMAGE UPLOAD
if input_option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)
#webcam input
elif input_option == "Use Webcam":

    camera_image = st.camera_input("Capture an Image")

    if camera_image is not None:

        image = Image.open(camera_image)

        image = np.array(image)

#process image
if image is not None:

    # Create columns
    col1, col2 = st.columns(2)

    # Show original image
    with col1:

        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )
    
    #feature processing
    output = None

    # FACE DETECTION
    if feature == "Face Detection":

        output = detect_faces(image)

    # OBJECT DETECTION
    elif feature == "Object Detection":

        output = detect_objects(image)
    
    # DOCUMENT ENHANCER
    elif feature == "Document Enhancer":

        output = enhance_document(image)

    # EDGE DETECTION
    elif feature == "Edge Detection":

        threshold1 = st.sidebar.slider(
            "Threshold 1",
            0,
            255,
            100
        )
        threshold2 = st.sidebar.slider(
            "Threshold 2",
            0,
            255,
            200
        )
        output = detect_edges(
            image,
            threshold1,
            threshold2
        )

    # CARTOON FILTER
    elif feature == "Cartoon Filter":

        output = cartoon_filter(image)

    # PENCIL SKETCH
    elif feature == "Pencil Sketch":

        output = pencil_sketch(image)

    # BLUR DETECTION
    elif feature == "Blur Detection":

        variance, is_blurry = detect_blur(image)
        status = "BLURRY" if is_blurry else "SHARP"
        st.sidebar.write(f"Blur Score: {variance:.2f}")
        st.sidebar.write(f"Status: {status}")

        # For display, we'll show the original image with blur status
        output = image.copy()
        cv2.putText(output, f"Blur Score: {variance:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(output, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # OCR SCANNER
    elif feature == "OCR Scanner":

        try:
            text, data, processed = perform_ocr(image)
            result_with_boxes = draw_text_boxes(image, data)

            st.sidebar.text_area("Extracted Text", text, height=200)

            output = result_with_boxes
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            st.info("💡 **Quick Fix:**\n"
                   "1. Download: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.4.20240514.exe\n"
                   "2. Install and check 'Add to PATH'\n"
                   "3. Restart the app\n\n"
                   "**Alternative:** Run `conda install -c conda-forge tesseract` in terminal")
            output = image  # Show original image as fallback

    # FACIAL EMOTION DETECTION
    elif feature == "Facial Emotion Detection":

        try:
            result_image, emotion_results = detect_emotions(image)

            # Display emotion results in sidebar
            if emotion_results:
                st.sidebar.write("### Detected Emotions:")
                for i, result in enumerate(emotion_results):
                    st.sidebar.write(f"**Face {i+1}:** {result['emotion']} ({result['confidence']:.2f})")
            else:
                st.sidebar.write("No faces detected in the image.")

            output = result_image
        except Exception as e:
            st.error(f"Emotion Detection Error: {str(e)}")
            output = image  # Show original image as fallback
    if output is not None:
        with col2:
            st.subheader("✨ Processed Output")

            st.image(
                output,
                use_container_width=True
            )

        
        # SAVE OUTPUT BUTTON
        if st.button("💾 Save Output"):

            # Convert RGB -> BGR before saving
            save_image = output

            if len(output.shape) == 3:

                save_image = cv2.cvtColor(
                    output,
                    cv2.COLOR_RGB2BGR
                )

            cv2.imwrite(
                "outputs/result.jpg",
                save_image
            )

            st.success("✅ Output saved in outputs/result.jpg")


# FOOTER
st.markdown("---")

st.markdown(
    """
    ### 🚀 Features Included
    - Face Detection
    - Object Detection
    - Document Enhancement
    - Edge Detection
    - Cartoon Filter
    - Pencil Sketch
    - Blur Detection
    - OCR Scanner
    - Facial Emotion Detection
    """
)