from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from deepface import DeepFace
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
task_list = ["Real time Emotions Detection app","Image emotion detection App"]

with st.sidebar:
    st.title('Task Selection')
    task_name = st.selectbox("Select your tasks:", task_list)

if task_name == task_list[0]:


        class EmotionVideoProcessor:
            def __init__(self):
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            def recv(self, frame):
                frm = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)

                for x, y, w, h in faces:
                    try:
                        result = DeepFace.analyze(frm[y:y + h, x:x + w], actions=['emotion'])
                        dominant_emotion = result[0]['dominant_emotion']
                        emotion_intensity = result[0]['emotion'][dominant_emotion]

                        emotion_text = f"Emotion: {dominant_emotion}"
                        intensity_text = f"Intensity: {emotion_intensity:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        cv2.putText(frm, emotion_text, (x + w + 10, y + h // 2), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frm, intensity_text, (x, y + h + 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    except ValueError as e:
                        error_message = "Face not detected :("
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frm, error_message, (x, y - 20), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(frm, "Please check your position", (x, y + h + 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)

                return av.VideoFrame.from_ndarray(frm, format='bgr24')


        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        st.title("Real time Emotions Detection App")

        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=rtc_configuration,
            # Remove the AudioProcessor
            video_transformer_factory=None,
        )
else:
    st.title("Image emotion detection App")


    def display_image_with_text(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = faces_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # text_x = x-300  # Adjust this value for desired text position
            #
            # text_y = y + h + int(image.shape[0] * fontScale)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # position = (text_x, text_y)
            # color = (0, 0, 255)  # Red color
            # thickness = 1
            # lineType = cv2.LINE_AA  # Anti-aliased line

            img = image.copy()
            # cv2.putText(img, emotion_text, position, font, fontScale, color, thickness, lineType)
            # cv2.putText(img)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img_rgb



    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting emotion..."):
            pred = DeepFace.analyze(image_cv2)
            emotion_text = pred[0]['dominant_emotion']  # Replace with the actual emotion text

        # Display the uploaded image with emotion text below it
        st.header(f"Emotion: {emotion_text}")
        st.image(display_image_with_text(image_cv2), use_column_width=True)
