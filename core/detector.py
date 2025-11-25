import os

os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_HUB'] = 'False'

from ultralytics import YOLO

class PeopleDetector:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        self.confidence = confidence
        self.classes = [0]  # Класс 'person'
        self.model = YOLO(model_path)
    
    def detect(self, frame):

        try:
            results = self.model(frame, conf=self.confidence, 
                                 classes=self.classes, verbose=False,
                                 imgsz=640)
            
            detections = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        detections.append({'bbox': [x1, y1, x2, y2],
                                           'confidence': confidence,
                                           'class_name': 'person'})
            
            return detections
            
        except Exception as e:
            print(f"Error: Ошибка детекции: {e}")
            return self._mock_detections(frame)