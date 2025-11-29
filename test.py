import cv2
import numpy as np
import base64
from flask import Flask, jsonify, render_template_string
import threading
import time
import random
from datetime import datetime

app = Flask(__name__)

# Глобальные переменные
current_frame = None
current_metrics = {
    'people_count': 0,
    'violations_count': 0, 
    'density': 0.0,
    'risk_level': 'Низкий',
    'zone_type': 'test',
    'timestamp': datetime.now().isoformat()
}
frame_lock = threading.Lock()
processing = False
frame_count = 0

def generate_simple_frame(frame_count):
    """Генерирует простой тестовый кадр с людьми"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    
    # Случайное количество людей (1-6)
    people_count = random.randint(1, 6)
    
    # Цвета для людей
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    people_positions = []
    
    for i in range(people_count):
        color = colors[i % len(colors)]
        
        # Позиции чтобы люди не пересекались
        if i == 0:
            x = 100 + int(frame_count * 2) % 300
            y = 100
        elif i == 1:
            x = 400
            y = 150 + int(frame_count * 1.5) % 200
        elif i == 2:
            x = 200 + int(frame_count * 1.8) % 250
            y = 300
        elif i == 3:
            x = 500
            y = 200 + int(frame_count * 1.2) % 150
        elif i == 4:
            x = 150
            y = 350 + int(frame_count * 0.8) % 100
        else:
            x = 300 + int(frame_count * 1.0) % 200
            y = 400
        
        # Рисуем человека (прямоугольник)
        cv2.rectangle(frame, (x, y), (x + 50, y + 100), color, -1)
        cv2.rectangle(frame, (x, y), (x + 50, y + 100), (255, 255, 255), 2)
        
        # ID человека
        cv2.putText(frame, f"ID:{i+1}", (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        people_positions.append((x + 25, y + 50))  # Центр человека
    
    # Рисуем линии между близкими людьми (нарушения)
    violation_count = 0
    for i in range(len(people_positions)):
        for j in range(i + 1, len(people_positions)):
            pos1 = people_positions[i]
            pos2 = people_positions[j]
            
            # Расстояние между людьми
            distance = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
            
            # Если расстояние меньше 100 пикселей - считаем нарушением
            if distance < 100:
                cv2.line(frame, pos1, pos2, (0, 0, 255), 2)
                violation_count += 1
                
                # Подпись расстояния
                mid_x = (pos1[0] + pos2[0]) // 2
                mid_y = (pos1[1] + pos2[1]) // 2
                cv2.putText(frame, f"{distance:.0f}px", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Обновляем метрики
    with frame_lock:
        current_metrics.update({
            'people_count': people_count,
            'violations_count': violation_count,
            'density': round(people_count * 0.5, 1),
            'risk_level': 'Высокий' if violation_count > 2 else 'Средний' if violation_count > 0 else 'Низкий',
            'zone_type': 'test',
            'timestamp': datetime.now().isoformat()
        })
    
    # Информация на кадре
    cv2.putText(frame, f"ЛЮДИ: {people_count} | НАРУШЕНИЯ: {violation_count}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"КАДР: {frame_count}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "ТЕСТОВЫЙ РЕЖИМ - РАБОТАЕТ", (10, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def simple_processing():
    """Упрощенная обработка видео"""
    global processing, current_frame, frame_count
    
    while processing:
        frame = generate_simple_frame(frame_count)
        
        # Кодируем кадр
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            with frame_lock:
                current_frame = buffer.tobytes()
        
        frame_count += 1
        time.sleep(0.1)  # 10 FPS
    
    print("Обработка остановлена")

# HTML шаблон
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Тест системы</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f2f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
        }
        .controls {
            padding: 20px;
            background: #ecf0f1;
            text-align: center;
        }
        button {
            padding: 12px 24px;
            margin: 0 10px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-start {
            background: #27ae60;
            color: white;
        }
        .btn-stop {
            background: #e74c3c;
            color: white;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 0;
        }
        .video-container {
            padding: 20px;
            background: #1a1a1a;
        }
        .video-container h3 {
            color: white;
            text-align: center;
            margin-bottom: 15px;
        }
        .video-wrapper {
            background: black;
            border-radius: 5px;
            overflow: hidden;
            text-align: center;
        }
        #videoFeed {
            max-width: 100%;
        }
        .metrics-container {
            padding: 20px;
            background: #f8f9fa;
        }
        .metrics-container h3 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card strong {
            color: #2c3e50;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .status {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 10px 0;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background: #27ae60;
            animation: pulse 1s infinite;
        }
        .status-inactive {
            background: #e74c3c;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            z-index: 1000;
        }
        .notification.success { background: #27ae60; }
        .notification.error { background: #e74c3c; }
        .notification.info { background: #3498db; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Тест системы анализа социальной дистанции</h1>
            <p>Упрощенная демонстрация работы</p>
            <div class="status">
                <span class="status-indicator status-inactive" id="statusIndicator"></span>
                <span id="statusText">Система остановлена</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn-start" onclick="startProcessing()">▶️ Запуск обработки</button>
            <button class="btn-stop" onclick="stopProcessing()">⏹️ Остановка</button>
        </div>

        <div class="dashboard">
            <div class="video-container">
                <h3>🎥 Видеопоток</h3>
                <div class="video-wrapper">
                    <img id="videoFeed" src="/api/frame" alt="Видеопоток">
                </div>
            </div>

            <div class="metrics-container">
                <h3>📊 Статистика в реальном времени</h3>
                <div id="metrics">
                    <div class="metric-card">
                        <strong>👥 Людей в кадре:</strong>
                        <div class="metric-value" id="peopleCount">0</div>
                    </div>
                    <div class="metric-card">
                        <strong>🚨 Нарушения дистанции:</strong>
                        <div class="metric-value" id="violationsCount">0</div>
                    </div>
                    <div class="metric-card">
                        <strong>📏 Плотность потока:</strong>
                        <div class="metric-value" id="density">0.0 чел/м²</div>
                    </div>
                    <div class="metric-card">
                        <strong>⚠️ Уровень риска:</strong>
                        <div class="metric-value" id="riskLevel">Низкий</div>
                    </div>
                    <div class="metric-card">
                        <strong>🏷️ Тип зоны:</strong>
                        <div class="metric-value" id="zoneType">test</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let updateInterval;
        let isProcessing = false;

        function showNotification(type, message) {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.innerHTML = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }

        function updateStatusIndicator(active) {
            const indicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            
            if (active) {
                indicator.className = 'status-indicator status-active';
                statusText.textContent = 'Система активна';
            } else {
                indicator.className = 'status-indicator status-inactive';
                statusText.textContent = 'Система остановлена';
            }
            isProcessing = active;
        }

        function startProcessing() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    showNotification('success', '✅ Обработка запущена');
                    updateStatusIndicator(true);
                    startFrameUpdates();
                    startMetricsUpdates();
                })
                .catch(error => {
                    showNotification('error', '❌ Ошибка запуска');
                });
        }

        function stopProcessing() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    showNotification('info', '⏹️ Обработка остановлена');
                    updateStatusIndicator(false);
                    stopFrameUpdates();
                    stopMetricsUpdates();
                })
                .catch(error => {
                    showNotification('error', '❌ Ошибка остановки');
                });
        }

        function startFrameUpdates() {
            // Обновляем кадр каждые 100ms
            setInterval(() => {
                document.getElementById('videoFeed').src = '/api/frame?t=' + Date.now();
            }, 100);
        }

        function stopFrameUpdates() {
            // Очистка интервалов не нужна для этого простого примера
        }

        function startMetricsUpdates() {
            // Обновляем метрики каждую секунду
            updateInterval = setInterval(updateMetrics, 1000);
        }

        function stopMetricsUpdates() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        }

        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(metrics => {
                    document.getElementById('peopleCount').textContent = metrics.people_count;
                    document.getElementById('violationsCount').textContent = metrics.violations_count;
                    document.getElementById('density').textContent = metrics.density + ' чел/м²';
                    document.getElementById('riskLevel').textContent = metrics.risk_level;
                    document.getElementById('zoneType').textContent = metrics.zone_type;
                    
                    // Цвет уровня риска
                    const riskElement = document.getElementById('riskLevel');
                    riskElement.style.color = 
                        metrics.risk_level === 'Высокий' ? '#e74c3c' :
                        metrics.risk_level === 'Средний' ? '#f39c12' : '#27ae60';
                })
                .catch(error => {
                    console.error('Ошибка получения метрик:', error);
                });
        }

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            updateStatusIndicator(false);
            updateMetrics(); // Загружаем начальные метрики
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/frame')
def get_frame():
    """Отдает текущий кадр"""
    if current_frame:
        return current_frame, 200, {'Content-Type': 'image/jpeg'}
    else:
        # Генерируем начальный кадр
        frame = generate_simple_frame(0)
        success, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/api/start', methods=['POST'])
def start_processing():
    global processing
    if not processing:
        processing = True
        threading.Thread(target=simple_processing, daemon=True).start()
        return jsonify({'status': 'started', 'message': 'Обработка запущена'})
    return jsonify({'status': 'already_running', 'message': 'Обработка уже запущена'})

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify({'status': 'stopped', 'message': 'Обработка остановлена'})

@app.route('/api/metrics')
def get_metrics():
    """Отдает текущие метрики"""
    with frame_lock:
        return jsonify(current_metrics)

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'processing': processing,
        'frame_count': frame_count,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Запуск упрощенной тестовой системы")
    print("📊 Откройте: http://localhost:5001")
    print("🎮 Нажмите 'Запуск обработки' для начала")
    app.run(host='0.0.0.0', port=5001, debug=False)