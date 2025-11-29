import os
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_HUB'] = 'False'

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
import time
import tempfile
import uuid
from datetime import datetime
import base64
import json

app = Flask(__name__)

# Глобальные переменные
detector = None
tracker = None
logger = None
metrics_calc = None
processing = False
current_frame = None
frame_lock = threading.Lock()
video_source = 'test'
current_video_path = None
cap = None

# Глобальные метрики
current_metrics = {
    'people_count': 0,
    'violations_count': 0,
    'density': 0.0,
    'risk_level': 'Низкий',
    'active_violations': [],
    'movement_metrics': {},
    'zone_type': 'default',
    'timestamp': datetime.now().isoformat()
}

# Импортируем компоненты
try:
    from core.detector import PeopleDetector
    from core.tracker import PeopleTracker
    from utils.logger import EventLogger, EventType
    from utils.metrics import MetricsCalculator, RiskLevel
    components_loaded = True
    print("✅ Все компоненты успешно загружены")
except ImportError as e:
    print(f"❌ Ошибка импорта компонентов: {e}")
    components_loaded = False

def initialize_components():
    global detector, tracker, logger, metrics_calc
    
    if not components_loaded:
        print("❌ Компоненты не загружены")
        return False
    
    try:
        print("🔄 Инициализация детектора...")
        detector = PeopleDetector()
        print("🔄 Инициализация трекера...")
        tracker = PeopleTracker()
        print("🔄 Инициализация логгера...")
        logger = EventLogger()
        print("🔄 Инициализация калькулятора метрик...")
        metrics_calc = MetricsCalculator()
        print("✅ Все компоненты инициализированы")
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации компонентов: {e}")
        import traceback
        traceback.print_exc()
        return False

# Инициализируем компоненты при запуске
components_initialized = initialize_components()

def cleanup_video_capture():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def initialize_video_source(source_type, video_path=None):
    global cap, video_source, current_video_path
    
    cleanup_video_capture()
    
    video_source = source_type
    current_video_path = video_path
    
    if source_type == 'camera':
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Камера найдена (индекс {camera_index})")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 25)
            return True
        else:
            print("❌ Не удалось найти работающую камеру")
            return False
        
    elif source_type == 'video_file' and video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"✅ Видеофайл загружен: {video_path}")
            print(f"   FPS: {fps}, Кадров: {frame_count}")
            return True
        else:
            print(f"❌ Не удалось открыть видеофайл: {video_path}")
            return False
    
    elif source_type == 'test':
        print("✅ Тестовый режим активирован")
        return True
    
    return False

def process_video_stream():
    global processing, current_frame, current_metrics, video_source, cap
    
    print(f"🎥 Запуск обработки видео. Источник: {video_source}")
    
    if not initialize_video_source(video_source, current_video_path):
        print("❌ Не удалось инициализировать источник видео")
        processing = False
        return
    
    frame_count = 0
    last_log_time = time.time()
    
    while processing:
        try:
            frame = None
            
            if video_source in ['camera', 'video_file'] and cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if video_source == 'video_file':
                        # Перезапускаем видеофайл
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        print("🔄 Перезапуск видеофайла")
                        continue
                    else:
                        print("❌ Не удалось получить кадр с камеры")
                        break
                
                frame = cv2.resize(frame, (640, 480))
            else:
                # Генерируем тестовый кадр
                frame = generate_test_frame(frame_count)
                time.sleep(0.04)  # Имитация 25 FPS
            
            if frame is None:
                print("❌ Получен пустой кадр")
                continue
            
            # Обработка кадра
            print(f"🔍 Обработка кадра {frame_count}...")
            detections = detector.detect(frame)
            print(f"   Найдено детекций: {len(detections)}")
            
            tracks = tracker.update(detections, frame)
            print(f"   Активных треков: {len(tracks)}")
            
            # Расчет метрик
            movement_metrics = metrics_calc.calculate_movement_metrics(tracks)
            distances, violations = metrics_calc.calculate_pairwise_distances(
                tracks, frame.shape[:2], movement_metrics
            )
            
            zone_type = metrics_calc.zone_analyzer.detect_zone_type(
                tracks, frame.shape[:2], movement_metrics
            )
            
            density = metrics_calc.calculate_density(len(tracks))
            risk_level = metrics_calc.assess_risk_level(len(violations), density, zone_type.value)
            
            real_violations = [v for v in violations if v.get('is_real_violation', False)]
            
            # Обновление глобальных метрик
            with frame_lock:
                current_metrics.update({
                    'people_count': len(tracks),
                    'violations_count': len(real_violations),
                    'density': density,
                    'risk_level': risk_level.value,
                    'active_violations': real_violations[:5],
                    'movement_metrics': movement_metrics,
                    'zone_type': zone_type.value,
                    'timestamp': datetime.now().isoformat(),
                })
            
            # Визуализация
            result_frame = tracker.draw_tracks(frame, tracks)
            
            # Рисуем нарушения
            for violation in real_violations[:3]:
                track1 = next((t for t in tracks if t['track_id'] == violation['person1']), None)
                track2 = next((t for t in tracks if t['track_id'] == violation['person2']), None)
                
                if track1 and track2:
                    cv2.line(result_frame, track1['center'], track2['center'], (0, 0, 255), 2)
                    mid_point = (
                        (track1['center'][0] + track2['center'][0]) // 2,
                        (track1['center'][1] + track2['center'][1]) // 2
                    )
                    cv2.putText(result_frame, f"{violation['distance']:.1f}m", 
                               mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Рисуем метрики на кадре
            draw_metrics_on_frame(result_frame, current_metrics, zone_type.value)
            
            # Добавляем информацию о режиме
            mode_text = {
                'camera': "РЕЖИМ: ВЕБ-КАМЕРА",
                'video_file': "РЕЖИМ: ВИДЕОФАЙЛ", 
                'test': "РЕЖИМ: ТЕСТОВЫЙ"
            }.get(video_source, "РЕЖИМ: НЕИЗВЕСТЕН")
            
            cv2.putText(result_frame, mode_text, (10, result_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Добавляем отладочную информацию
            debug_text = f"Кадр: {frame_count} | Детекций: {len(detections)} | Треков: {len(tracks)}"
            cv2.putText(result_frame, debug_text, (10, result_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Сохраняем кадр
            with frame_lock:
                success, encoded_image = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    current_frame = encoded_image.tobytes()
                else:
                    print("❌ Ошибка кодирования кадра")
            
            frame_count += 1
            
            # Логируем каждые 5 секунд
            current_time = time.time()
            if current_time - last_log_time >= 5:
                print(f"📊 Статистика: кадров {frame_count}, людей {len(tracks)}, нарушений {len(real_violations)}")
                last_log_time = current_time
                
        except Exception as e:
            print(f"❌ Ошибка обработки кадра: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    cleanup_video_capture()
    print("⏹️ Обработка видео остановлена")

def draw_metrics_on_frame(frame, metrics, zone_type):
    """Рисует метрики на кадре"""
    y_offset = 30
    line_height = 25
    
    metrics_text = [
        f"Люди: {metrics['people_count']}",
        f"Нарушения: {metrics['violations_count']}",
        f"Плотность: {metrics['density']:.1f} чел/м²",
        f"Риск: {metrics['risk_level']}",
        f"Зона: {zone_type}"
    ]
    
    # Фон для текста
    for i, text in enumerate(metrics_text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (5, y_offset + i * line_height - 20), 
                     (text_size[0] + 15, y_offset + i * line_height + 5), 
                     (0, 0, 0), -1)
    
    # Текст метрик
    for i, text in enumerate(metrics_text):
        color = (0, 255, 0) if metrics['risk_level'] == 'Низкий' else (
            (0, 255, 255) if metrics['risk_level'] == 'Средний' else (0, 0, 255)
        )
        cv2.putText(frame, text, (10, y_offset + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def generate_test_frame(frame_count):
    """Генерирует тестовый кадр с движущимися объектами"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Сетка
    for i in range(0, frame.shape[1], 50):
        cv2.line(frame, (i, 0), (i, frame.shape[0]), (50, 50, 50), 1)
    for i in range(0, frame.shape[0], 50):
        cv2.line(frame, (0, i), (frame.shape[1], i), (50, 50, 50), 1)
    
    # Движущиеся объекты (люди) - создаем реалистичные прямоугольники
    objects = [
        {'pos': (100 + int(frame_count * 2) % 400, 100), 'size': (40, 80), 'color': (0, 255, 0), 'id': 1},
        {'pos': (300, 150 + int(frame_count * 1.5) % 200), 'size': (50, 100), 'color': (255, 0, 0), 'id': 2},
        {'pos': (200 + int(frame_count * 1.8) % 300, 300), 'size': (45, 90), 'color': (0, 0, 255), 'id': 3},
        {'pos': (400, 200 + int(frame_count * 1.2) % 150), 'size': (35, 70), 'color': (255, 255, 0), 'id': 4},
    ]
    
    for obj in objects:
        x, y = obj['pos']
        w, h = obj['size']
        
        # Рисуем тело (прямоугольник)
        cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], -1)
        
        # Рисуем контур
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Добавляем ID
        cv2.putText(frame, f"ID:{obj['id']}", (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Добавляем точку центра (для визуализации трекинга)
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
    
    # Информация о тестовом режиме
    cv2.putText(frame, "ТЕСТОВЫЙ РЕЖИМ - РАБОТАЕТ", (150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Система обнаруживает движущихся людей", (150, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Кадр: {frame_count}", (150, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/current_frame')
def get_current_frame():
    """API для получения текущего кадра в base64"""
    try:
        with frame_lock:
            if current_frame is not None:
                frame_base64 = base64.b64encode(current_frame).decode('utf-8')
                return jsonify({
                    'success': True,
                    'frame': frame_base64,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': current_metrics
                })
            else:
                # Генерируем тестовый кадр если нет обработанного
                test_frame = generate_test_frame(int(time.time()))
                success, buffer = cv2.imencode('.jpg', test_frame)
                if success:
                    frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    return jsonify({
                        'success': True,
                        'frame': frame_base64,
                        'timestamp': datetime.now().isoformat(),
                        'metrics': current_metrics,
                        'is_test_frame': True
                    })
        
        return jsonify({'success': False, 'error': 'No frame available'})
    
    except Exception as e:
        print(f"❌ Ошибка в API current_frame: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    global processing
    
    if processing:
        return jsonify({'status': 'error', 'message': 'Обработка уже запущена'})
    
    if not components_initialized:
        return jsonify({'status': 'error', 'message': 'Компоненты системы не инициализированы'})
    
    data = request.get_json() or {}
    source_type = data.get('video_source', 'test')
    video_file_path = data.get('video_file_path')
    
    if source_type == 'video_file' and not video_file_path:
        return jsonify({'status': 'error', 'message': 'Не указан путь к видеофайлу'})
    
    # Обновляем глобальные переменные
    global video_source, current_video_path
    video_source = source_type
    current_video_path = video_file_path
    
    # Запускаем обработку в отдельном потоке
    processing = True
    thread = threading.Thread(target=process_video_stream)
    thread.daemon = True
    thread.start()
    
    source_name = {
        'camera': 'веб-камера',
        'video_file': 'видеофайл', 
        'test': 'тестовый режим'
    }.get(source_type, source_type)
    
    # Логируем событие
    if logger:
        logger.log_event(
            EventType.INFO,
            f"Запущена обработка видео. Источник: {source_type}",
            {'source': source_type, 'file_path': video_file_path}
        )
    
    return jsonify({
        'status': 'success', 
        'message': f'Обработка запущена ({source_name})',
        'video_source': source_type
    })

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    
    processing = False
    cleanup_video_capture()
    
    # Логируем событие
    if logger:
        logger.log_event(
            EventType.INFO,
            "Обработка видео остановлена",
            {}
        )
    
    return jsonify({'status': 'success', 'message': 'Обработка остановлена'})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'Файл не найден'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'status': 'error', 'message': 'Файл не выбран'})
        
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        file_extension = video_file.filename.lower().split('.')[-1]
        if file_extension not in allowed_extensions:
            return jsonify({'status': 'error', 'message': 'Неподдерживаемый формат видео'})
        
        # Сохраняем файл во временную директорию
        temp_dir = tempfile.gettempdir()
        filename = f"uploaded_video_{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(temp_dir, filename)
        video_file.save(file_path)
        
        # Проверяем что файл можно открыть
        test_cap = cv2.VideoCapture(file_path)
        if not test_cap.isOpened():
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': 'Не удалось открыть видеофайл'})
        
        test_cap.release()
        
        # Логируем событие
        if logger:
            logger.log_event(
                EventType.INFO,
                f"Загружен видеофайл: {filename}",
                {'filename': filename, 'file_path': file_path}
            )
        
        return jsonify({
            'status': 'success', 
            'message': 'Видео успешно загружено',
            'file_path': file_path,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Ошибка загрузки: {str(e)}'})

@app.route('/api/metrics')
def get_metrics():
    """API для получения текущих метрик"""
    with frame_lock:
        return jsonify(current_metrics)

@app.route('/api/violations')
def get_violations():
    """API для получения текущих нарушений"""
    with frame_lock:
        violations = current_metrics.get('active_violations', [])
        return jsonify(violations)

@app.route('/api/events')
def get_events():
    """API для получения событий из лога"""
    if logger:
        events = logger.get_recent_events(5)
        return jsonify(events)
    return jsonify([])

@app.route('/api/health')
def health_check():
    """API для проверки статуса системы"""
    return jsonify({
        'status': 'healthy',
        'processing': processing,
        'video_source': video_source,
        'components_initialized': components_initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/debug_info')
def debug_info():
    """API для отладки"""
    debug_info = {
        'processing': processing,
        'video_source': video_source,
        'current_video_path': current_video_path,
        'components_initialized': components_initialized,
        'current_frame_available': current_frame is not None,
        'cap_initialized': cap is not None and cap.isOpened() if cap else False,
        'timestamp': datetime.now().isoformat()
    }
    
    if cap and cap.isOpened():
        debug_info.update({
            'cap_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'cap_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'cap_fps': cap.get(cv2.CAP_PROP_FPS),
            'cap_frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT) if video_source == 'video_file' else 'N/A'
        })
    
    return jsonify(debug_info)

@app.route('/api/test_detection')
def test_detection():
    """API для тестирования детекции"""
    try:
        # Создаем тестовый кадр
        test_frame = generate_test_frame(0)
        
        # Тестируем детектор
        detections = detector.detect(test_frame)
        
        # Тестируем трекер
        tracks = tracker.update(detections, test_frame)
        
        return jsonify({
            'success': True,
            'detections_count': len(detections),
            'tracks_count': len(tracks),
            'detections': [{'bbox': det['bbox'], 'confidence': det['confidence']} for det in detections],
            'tracks': [{'track_id': track['track_id'], 'center': track['center']} for track in tracks]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Запуск системы анализа социальной дистанции")
    print("=" * 50)
    print("📊 Доступные endpoints:")
    print("  http://localhost:5000 - Web интерфейс")
    print("  http://localhost:5000/api/current_frame - Текущий кадр с метриками")
    print("  http://localhost:5000/api/health - Статус системы")
    print("  http://localhost:5000/api/debug_info - Отладочная информация")
    print("  http://localhost:5000/api/test_detection - Тест детекции")
    print("\n🎮 Для тестирования:")
    print("  1. Откройте http://localhost:5000")
    print("  2. Выберите 'Тестовый режим'")
    print("  3. Нажмите 'Запуск обработки'")
    print("=" * 50)
    
    # Создаем начальный тестовый кадр
    initial_frame = generate_test_frame(0)
    success, buffer = cv2.imencode('.jpg', initial_frame)
    if success:
        with frame_lock:
            current_frame = buffer.tobytes()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)