import os
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_HUB'] = 'False'

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from core.detector import PeopleDetector
from core.tracker import PeopleTracker
from utils.logger import EventLogger, EventType
from utils.metrics import MetricsCalculator
import threading
import time

app = Flask(__name__)

detector = None
tracker = None
logger = None
metrics_calc = None
processing = False
current_frame = None
frame_lock = threading.Lock()

def initialize_components():
    """Инициализация компонентов"""
    global detector, tracker, logger, metrics_calc
    
    if detector is None:
        detector = PeopleDetector()
    
    if tracker is None:
        tracker = PeopleTracker()
    
    if logger is None:
        logger = EventLogger()
    
    if metrics_calc is None:
        metrics_calc = MetricsCalculator()


initialize_components()

# Метрики
current_metrics = {'people_count': 0,
                   'violations_count': 0,
                   'density': 0.0,
                   'risk_level': 'Низкий',
                   'active_violations': [],
                   'movement_metrics': {}
                  }


def process_video_stream():
    """Обработка видеопотока в отдельном потоке"""
    global processing, current_frame, current_metrics
    
    # Тестовое видео или камера
    if request.json and request.json.get('video_source') == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = None
    
    frame_count = 0
    while processing:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Тестовый кадр с движущимися объектами
            frame = generate_test_frame(frame_count)
            time.sleep(0.04)  # примерно 25 FPS
        
        # Детекция и трекинг
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        
        # Расчет метрик
        distances, violations = metrics_calc.calculate_pairwise_distances(tracks)
        density = metrics_calc.calculate_density(len(tracks))
        risk_level = metrics_calc.assess_risk_level(len(violations), density)
        movement_metrics = metrics_calc.calculate_movement_metrics(tracks)
        
        # Обновление глобальных метрик
        current_metrics.update({'people_count': len(tracks),
                                'violations_count': len(violations),
                                'density': density,
                                'risk_level': risk_level.value,
                                'active_violations': violations[:10],
                                'movement_metrics': movement_metrics
                               })
        
        # Логирование при нарушениях
        if violations:
            for violation in violations[:3]:
                logger.log_event(EventType.VIOLATION,
                                 f"Нарушение дистанции: {violation['distance']:.2f}м",
                                 violation,
                                 [violation['person1'], violation['person2']]
                                )
        
        # Визуализация
        result_frame = tracker.draw_tracks(frame, tracks)
        
        # Отрисовка нарушений
        for violation in violations[:5]:
            track1 = next((t for t in tracks if t['track_id'] == violation['person1']), None)
            track2 = next((t for t in tracks if t['track_id'] == violation['person2']), None)
            
            if track1 and track2:
                # Линия между нарушителями
                cv2.line(result_frame, track1['center'], track2['center'], (0, 0, 255), 2)

                # Текст с расстоянием
                mid_point = ((track1['center'][0] + track2['center'][0]) // 2,
                             (track1['center'][1] + track2['center'][1]) // 2)
                
                cv2.putText(result_frame, f"{violation['distance']:.1f}m", 
                            mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Обновление текущего кадра
        with frame_lock:
            _, buffer = cv2.imencode('.jpg', result_frame)
            current_frame = buffer.tobytes()
        
        frame_count += 1
    
    if cap:
        cap.release()


def generate_test_frame(frame_count):
    """Генерация тестового кадра с движущимися объектами"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Движущиеся объекты для тестирования
    objects = [{'pos': (100 + frame_count % 200, 100), 'size': (50, 100), 'color': (0, 255, 0)},
               {'pos': (300, 150 + (frame_count * 2) % 100), 'size': (60, 120), 'color': (255, 0, 0)},
               {'pos': (200 + frame_count % 150, 300), 'size': (70, 130), 'color': (0, 0, 255)},
              ]
    
    for obj in objects:
        x, y = obj['pos']
        w, h = obj['size']
        cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], -1)
    
    return frame


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    """Генератор кадров для видеопотока"""
    while True:
        with frame_lock:
            if current_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.04)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    global processing
    if not processing:
        processing = True
        thread = threading.Thread(target=process_video_stream)
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'started', 'message': 'Обработка запущена'})
    return jsonify({'status': 'already_running', 'message': 'Обработка уже запущена'})


@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify({'status': 'stopped', 'message': 'Обработка остановлена'})


@app.route('/api/metrics')
def get_metrics():
    return jsonify(current_metrics)


@app.route('/api/violations')
def get_violations():
    return jsonify(current_metrics.get('active_violations', []))


@app.route('/api/events')
def get_events():
    events = logger.get_recent_events(10)
    return jsonify(events)


@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    if data.get('confidence'):
        detector.confidence = float(data['confidence'])
    return jsonify({'status': 'updated', 'message': 'Настройки обновлены'})


if __name__ == '__main__':
    print("Запуск Flask приложения")
    print("Откройте: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)