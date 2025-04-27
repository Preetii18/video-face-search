from deepface import DeepFace
import cv2

# Load reference image (the person you're searching for)
reference_img_path = "image.jpg"

# Open the video
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_num = 0
match_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 5th frame to save time
    if frame_num % 5 == 0:
        try:
            # Save current frame as temp image
            cv2.imwrite("temp_frame.jpg", frame)

            # Run face verification
            result = DeepFace.verify(img1_path=reference_img_path, img2_path="temp_frame.jpg", enforce_detection=False)

            if result["verified"]:
                timestamp = frame_num / fps
                print(f"Match found at {timestamp:.2f} seconds")
                match_frames.append(timestamp)

        except Exception as e:
            print(f"Error at frame {frame_num}: {e}")

    frame_num += 1

cap.release()
