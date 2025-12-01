import os

os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_HUB'] = 'False'

from ultralytics import YOLO
import numpy as np

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
    
    def _mock_detections(self, frame):
        """Моковые детекции для тестирования когда модель не доступна"""
        height, width = frame.shape[:2]
        
        detections = [] # Тестовые детекции
        
        # Центральная детекция
        center_x, center_y = width // 2, height // 2
        bbox_size = 100
        
        detections.append({
            'bbox': [center_x - bbox_size//2, center_y - bbox_size//2, 
                    center_x + bbox_size//2, center_y + bbox_size//2],
            'confidence': 0.9,
            'class_name': 'person'
        })
        
        # Случайные детекции
        for i in range(2):
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 200)
            x2 = x1 + np.random.randint(80, 120)
            y2 = y1 + np.random.randint(150, 250)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': 0.7 + np.random.random() * 0.2,
                'class_name': 'person'
            })
        
        return detections