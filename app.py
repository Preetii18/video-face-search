import streamlit as st
import cv2
import tempfile
import os
from deepface import DeepFace

st.set_page_config(page_title="Face Search in Video", layout="wide")
st.title("🎥 Face Search in Video using DeepFace")

# Upload reference image and video
ref_img = st.file_uploader("📷 Upload Reference Image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("🎞️ Upload Video", type=["mp4", "avi", "mov"])

if ref_img and video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:

        tmp_img.write(ref_img.read())
        tmp_vid.write(video_file.read())
        tmp_img_path = tmp_img.name
        tmp_vid_path = tmp_vid.name

    st.success("✅ Files uploaded successfully. Click below to begin analysis.")
    if st.button("🔍 Search for Face in Video"):
        cap = cv2.VideoCapture(tmp_vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        timestamps = []

        stframe = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        matches_found = 0

        with st.spinner("Analyzing video... this may take a few minutes"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % 5 == 0:
                    try:
                        result = DeepFace.verify(
                            img1_path=tmp_img_path,
                            img2_path=frame,
                            model_name="Facenet",
                            enforce_detection=False
                        )
                        verified = result.get("verified", False)
                        distance = result.get("distance", 1.0)
                        timestamp = frame_num / fps

                        if verified or distance < 0.4:
                            timestamps.append(timestamp)
                            matches_found += 1
                            stframe.image(frame, caption=f"Match at {timestamp:.2f}s | Distance: {distance:.4f}", channels="BGR", use_column_width=True)

                    except Exception as e:
                        st.warning(f"Error on frame {frame_num}: {e}")

                frame_num += 1
                progress.progress(min(frame_num / total_frames, 1.0))

            cap.release()

        if timestamps:
            st.success(f"🎯 Done! Found {matches_found} matches at:")
            st.write(timestamps)
        else:
            st.error("❌ No matches found.")

    # Cleanup
    os.remove(tmp_img_path)
    os.remove(tmp_vid_path)
