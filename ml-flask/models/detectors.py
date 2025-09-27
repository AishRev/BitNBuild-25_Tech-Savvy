# ml-flask/models/detectors.py

from ultralytics import YOLO
from PIL import Image
import torch

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 object detection model.
        """
        print("Loading Object Detection model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        print("Object Detection model loaded successfully.")

    def detect_objects(self, image: Image.Image) -> list:
        """
        Detects objects in the given image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            list: A list of unique detected object names (e.g., ['person', 'dog', 'car']).
        """
        results = self.model(image, device=self.device)
        
        detected_objects = set() # Use a set to store unique object names
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    object_name = self.model.names[class_id]
                    detected_objects.add(object_name)
                    
        return list(detected_objects)

# Example usage:
if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        test_image = Image.open("test_image.jpg")
        objects = detector.detect_objects(test_image)
        print(f"Detected Objects: {objects}")
    except FileNotFoundError:
        print("Please create a 'test_image.jpg' file in the 'models' directory to test the detector.")
    except Exception as e:
        print(f"An error occurred: {e}")