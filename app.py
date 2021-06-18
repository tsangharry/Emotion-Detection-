import streamlit as st
import cv2
import os
from inference import return_annotated_images
from tensorflow.keras.models import load_model
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


def app_object_detection():
    """Launches video with webrtc, calls annotated images.

    Returns:
        video : Annotated video.
    """    
    class NNVideoTransformer(VideoTransformerBase):

        def __init__(self):
            prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
            weightsPath = os.path.sep.join(['face_detector',
                                            "res10_300x300_ssd_iter_140000.caffemodel"])
            self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            self.emotionsNet = load_model('model/emo.h5')

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            annotated_image = return_annotated_images(image, self.faceNet, self.emotionsNet)

            return annotated_image

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=NNVideoTransformer,
        async_transform=True)

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.confidence_threshold = 0.5


def main():
    """
    Streamlit interface.
    """
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://www.fg-a.com/wallpapers/white-marble-1-2018.jpg");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True) 
    
    st.header("Emotion Detection from Facial Expressions")

    st.subheader('Unable to tell emotions like Sheldon Cooper? Let us help.')
    st.subheader('😀😮😔😡😐')

    app_object_detection()


main()
