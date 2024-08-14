import cv2
from ultralytics import YOLO
import os

# Step 1: Load the video
video_path = r'C:\Users\shova\PycharmProjects\Web_Scraping\משחק האירופאיות _ המשחק המלא_ מכבי חיפה - הפועל ב_ש 0-3.mp4'


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    return cap


# Step 2: Initialize YOLOv8 model
def initialize_model():
    model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model
    return model

# Step 3: Process the video and filter out non-gameplay segments
def process_video(cap, model):
    frame_number = 0
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Run YOLOv8 model on the frame
        results = model(frame)

        # Extract detected objects from the results
        detected_objects = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])  # Extract the class ID
                if class_id == 0:  # Assuming class 0 is 'person' in COCO dataset
                    detected_objects.append('person')
                elif class_id == 32:  # Assuming class 32 is 'sports ball' in COCO dataset
                    detected_objects.append('sports ball')

        if 'person' in detected_objects or 'sports ball' in detected_objects:
            processed_frames.append((frame_number, frame))
            # You can add additional processing here (e.g., save the frame, extract key events)

    cap.release()
    return processed_frames


# Step 4: Save the filtered frames (optional)
def save_frames(processed_frames, output_dir='processed_frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame_number, frame in processed_frames:
        output_path = os.path.join(output_dir, f'frame_{frame_number}.jpg')
        cv2.imwrite(output_path, frame)


def main():
    # Load video
    cap = load_video(video_path)
    if cap is None:
        return

    # Initialize YOLOv8 model
    model = initialize_model()

    # Process the video
    processed_frames = process_video(cap, model)

    # Save the filtered frames
    save_frames(processed_frames)


if __name__ == "__main__":
    main()
