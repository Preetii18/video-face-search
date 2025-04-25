from deepface import DeepFace
import cv2
import os

# Paths to your reference image and video
reference_img_path = r"C:\Users\vshpr\OneDrive\Pictures\person.png"
video_path = r"C:\Users\vshpr\Videos\Screen Recordings\video1.mp3.mp4"

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = 0
match_timestamps = []

print("🔍 Searching video...")

# Load the reference image as array (better compatibility)
ref_img = cv2.imread(reference_img_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % 5 == 0:  # Process every 5th frame
        try:
            # Compare reference image to current video frame
            result = DeepFace.verify(
                img1_path=ref_img,
                img2_path=frame,
                model_name="Facenet",  # try also "VGG-Face", "ArcFace", etc.
                enforce_detection=False
            )

            distance = result.get("distance", 1.0)
            verified = result.get("verified", False)
            timestamp = frame_num / fps

            print(f"Frame {frame_num} | Time {timestamp:.2f}s | Distance: {distance:.4f} | Match: {verified}")

            # Optional: Add your own threshold for possible matches
            if verified or distance < 0.4:
                print(f"✅ Possible match at {timestamp:.2f} seconds")
                match_timestamps.append(timestamp)

                # Optional: save matched frame for review
                cv2.imwrite(f"match_{frame_num}.jpg", frame)

        except Exception as e:
            print(f"⚠️ Error on frame {frame_num}: {e}")

    frame_num += 1

cap.release()

print("\n🎯 Done! Matching timestamps:")
print(match_timestamps)

