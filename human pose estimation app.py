import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import io

# Setup mediapipe pose estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit app title
st.title("Human Pose Estimation with Images, Videos, and Webcam")

# Instructions for the user
st.write("""
    Use the options below to either upload an image or video, or use your webcam to perform real-time human pose estimation.
    The system will detect your pose and highlight the landmarks on your body.
""")

# Option to select the type of input
input_option = st.selectbox("Choose Input Type", ["Image", "Video", "Webcam"])

# Image Input
if input_option == "Image":
    # File uploader for image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Load and display the uploaded image
        img = Image.open(uploaded_image)
        st.subheader('Uploaded Image')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert the image to a NumPy array
        img_cv = np.array(img)

        # Convert to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Process the image for pose detection
        results = pose.process(img_rgb)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.subheader('Positions Estimated')
        # Display the processed image with pose landmarks
        st.image(img_rgb, caption="Processed Image", channels="BGR", use_container_width=True)

# Video Input
elif input_option == "Video":
    # File uploader for video
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        temp_file.close()  # Close the temp file so OpenCV can access it

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_file.name)

        # Check if video is loaded properly
        if not cap.isOpened():
            st.error("Error: Unable to open video.")
            cap.release()
            

        # Get video metadata (frame count, resolution)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.write(f"Video Loaded: {frame_count} frames, Resolution: {frame_width}x{frame_height}")

        # Prepare video writer to save processed video
        output_path = "output_video.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (frame_width, frame_height))

        # Process each video frame
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for pose estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe pose estimator
            results = pose.process(frame_rgb)

            # Draw landmarks if pose is detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write the processed frame to output video
            out.write(frame)

            # Display the processed frame in Streamlit
            if frame_idx % 10 == 0:  # Display every 10th frame to improve performance
                st.image(frame, caption=f"Processed Video Frame {frame_idx}", channels="BGR", use_container_width=True)

            frame_idx += 1

        cap.release()
        out.release()

        # After processing, show success message and the processed video
        st.success("Pose estimation on video completed.")
        st.video(output_path)

# Webcam Input
elif input_option == "Webcam":
    camera_input = st.camera_input("Capture Image  with Webcam")

    if camera_input is not None:
        # Convert webcam input (in raw byte format) into an image using PIL
        img = Image.open(io.BytesIO(camera_input.getvalue()))
        
        # Convert the image to a NumPy array (OpenCV format)
        img_cv = np.array(img)
        
        # Convert the image from RGB (PIL format) to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Convert the image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the image for pose detection
        results = pose.process(img_rgb)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the processed image with pose landmarks in Streamlit
        st.image(img_bgr, caption="Processed Webcam Frame", channels="BGR", use_container_width=True)

else:
    st.warning("Please select an input type (Image, Video, or Webcam).")
