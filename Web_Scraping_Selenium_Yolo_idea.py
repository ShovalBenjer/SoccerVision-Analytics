import cv2
import torch
from ultralytics import YOLO
import os

# VideoLoader Class: Handles video loading
class VideoLoader:
    def __init__(self, path_to_video):
        self.video_path = path_to_video

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return None
        return cap

# YOLOModel Class: Manages YOLOv8 model initialization and inference
class YOLOModel:
    def __init__(self, model_path='yolov8n.pt', use_mixed_precision=True):
        self.device = "cpu"  # Default to CPU
        self.model = YOLO(model_path).to(self.device)
        self.use_mixed_precision = use_mixed_precision

    def detect_objects(self, frame):
        # Resize the frame to be compatible with the model's expected input size
        input_size = (640, 640)  # Default size for YOLOv8, change according to your model if needed
        frame_resized = cv2.resize(frame, input_size)  # Resize frame

        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(
            self.device)  # Convert frame to tensor
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast(enabled=True):  # Only use if on CUDA device
                    results = self.model(frame_tensor)
            else:
                results = self.model(frame_tensor)

        return results


class FrameProcessor:
    def __init__(self, model):
        self.model = model

    def process_video(self, cap):
        frame_number = 0
        processed_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Skip frames that are less likely to be important
            if frame_number % 5 != 0:
                continue

            # Run YOLOv8 model on the frame
            results = self.model.detect_objects(frame)

            # Extract detected objects from the results
            detected_objects = []
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if class_id in [0, 32]:  # 'person' or 'sports ball'
                        detected_objects.append('key_object')

            if 'key_object' in detected_objects:
                processed_frames.append((frame_number, frame))

        cap.release()
        return processed_frames

# FrameSaver Class: Manages saving of processed frames
class FrameSaver:
    def __init__(self, output_dir='processed_frames'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_frames(self, processed_frames):
        for frame_number, frame in processed_frames:
            output_path = os.path.join(self.output_dir, f'frame_{frame_number}.jpg')
            cv2.imwrite(output_path, frame)

# MetricsTracker Class: Placeholder for advanced football metrics tracking (to be implemented)
class MetricsTracker:
    def __init__(self):
        # Initialize tracking modules (e.g., xG, xA, etc.)
        pass

    def track_metrics(self, frame):
        # Implement metrics tracking logic here
        pass

# Main application logic
class FootballAnalyticsApp:
    def __init__(self, input_video_path):  # Rename the parameter to avoid shadowing
        self.video_loader = VideoLoader(input_video_path)
        self.yolo_model = YOLOModel()
        self.frame_processor = FrameProcessor(self.yolo_model)
        self.frame_saver = FrameSaver()

    def run(self):
        cap = self.video_loader.load_video()
        if cap is None:
            return

        processed_frames = self.frame_processor.process_video(cap)
        self.frame_saver.save_frames(processed_frames)

if __name__ == "__main__":
    video_path = r'C:\Users\shova\PycharmProjects\Web_Scraping\משחק האירופאיות _ המשחק המלא_ מכבי חיפה - הפועל ב_ש 0-3.mp4'
    app = FootballAnalyticsApp(video_path)
    app.run()
