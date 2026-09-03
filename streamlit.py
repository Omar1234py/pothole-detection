import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(
    page_title="Pothole Detection",
    layout="wide"
)

st.title("Pothole Detection")
st.write("Detect potholes from images and videos using YOLO.")


@st.cache_resource
def load_model():
    return YOLO(r"pothole2\best.pt")


model = load_model()


conf = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)


image_tab, video_tab = st.tabs([
    "Image Detection",
    "Video Detection"
])



with image_tab:

    st.header("Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        results = model.predict(
            source=image,
            conf=conf,
            verbose=False
        )

        result_image = results[0].plot()


        with col2:

            st.subheader("Detection Result")

            st.image(
                result_image,
                channels="BGR",
                use_container_width=True
            )


        potholes = len(results[0].boxes)

        if potholes > 0:

            st.success(
                f"Detected potholes: {potholes}"
            )

        else:

            st.info(
                "No potholes detected."
            )

with video_tab:

    st.header("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:


        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        input_path = input_file.name


        st.subheader("Original Video")

        st.video(input_path)

        if st.button(
            "Detect Potholes",
            key="video_detection_button"
        ):

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():

                st.error(
                    "Could not open the video."
                )

                st.stop()

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if fps <= 0:
                fps = 30

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_file.close()

            output_path = output_file.name


            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress_bar = st.progress(0)

            status_text = st.empty()

            frame_count = 0
            total_detections = 0


            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=conf,
                    verbose=False
                )

                annotated_frame = results[0].plot()

                detections = len(
                    results[0].boxes
                )

                total_detections += detections

                writer.write(
                    annotated_frame
                )

                frame_count += 1


                if total_frames > 0:

                    progress = (
                        frame_count / total_frames
                    )

                    progress_bar.progress(
                        min(progress, 1.0)
                    )

                    status_text.text(
                        f"Processing video... "
                        f"{frame_count}/{total_frames} frames"
                    )

            cap.release()
            writer.release()

            progress_bar.progress(1.0)

            status_text.success(
                "Video processing completed!"
            )

            if os.path.exists(output_path):

                st.subheader(
                    "Detection Result"
                )

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )
                st.success(
                    f"Total detections across frames: "
                    f"{total_detections}"
                )

                st.download_button(
                    label="Download Result Video",
                    data=video_bytes,
                    file_name="pothole_detection.mp4",
                    mime="video/mp4"
                )
            else:
                st.error(
                    "Could not create output video."
                )