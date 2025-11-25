#!/usr/bin/env python3
"""
Скрипт установки и настройки системы анализа социальной дистанции
"""

import os

def check_requirements():
    """Проверка и установка зависимостей"""
    print("Проверка зависимостей...")
    
    requirements = [
        "streamlit>=1.28.0",
        "opencv-python>=4.8.1.78",
        "ultralytics>=8.0.0",
        "deep-sort-realtime>=1.3.2",
        "torch>=2.0.1",
        "torchvision>=0.15.2", 
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "plotly>=5.15.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0"
    ]
    
    try:

        for package in requirements:

            package_name = package.split('>=')[0]

            if package_name == "opencv-python":
                import cv2
                print(f"Success: opencv-python установлен (версия {cv2.__version__})")
            elif package_name == "Pillow":
                from PIL import Image
                print(f"Success: Pillow установлен (версия {Image.__version__})")
            elif package_name == "deep-sort-realtime":
                try:
                    from deep_sort_realtime.deepsort_tracker import DeepSort
                    print("Success: deep-sort-realtime установлен")
                except ImportError as e:
                    print(f"Проблема с deep-sort-realtime: {e}")
            else:
                __import__(package_name.split('-')[0])
                print(f"Success: {package_name} установлен")
                
    except ImportError as e:
        print(f"Error: Не удалось импортировать {package_name}: {e}")
        return False
    
    return True

def download_yolo_model():
    """Загрузка модели YOLO если отсутствует"""

    print("\nПроверка модели YOLOv8...")
    
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print("Загрузка YOLOv8n модели...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("Success: Модель YOLOv8 загружена")
        except Exception as e:
            print(f"Error: Ошибка загрузки модели: {e}")
            return False
    else:
        print("Success: Модель YOLOv8 найдена")
    
    return True

def create_directories():
    """Создание необходимых директорий"""
    print("\nСоздание структуры проекта...")
    
    directories = [
        "core",
        "utils", 
        "logs",
        "test_data"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Success: Создана директория: {directory}")
        else:
            print(f"Success: Директория существует: {directory}")

def test_system():
    """Тестирование системы после установки"""
    print("\nТестирование системы...")
    
    try:
        from core.detector import PeopleDetector
        from core.tracker import PeopleTracker
        from utils.logger import EventLogger
        from utils.metrics import MetricsCalculator
        
        print("Success: Все модули импортируются корректно")
        
        # Тест инициализации
        detector = PeopleDetector()
        tracker = PeopleTracker()
        logger = EventLogger()
        metrics_calc = MetricsCalculator()
        
        print("Success: Все компоненты инициализируются")
        
        # Тест детектора
        import numpy as np

        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        detections = detector.detect(test_frame)
        print(f"Success: Детектор работает: найдено {len(detections)} объектов")
        
        # Тест трекера
        tracks = tracker.update(detections, test_frame)
        print(f"Success: Трекер работает: создано {len(tracks)} треков")
        
        return True
        
    except Exception as e:
        print(f"Error: Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def main():
    """Основная функция установки"""
    print("Установка системы анализа социальной дистанции")
    print("=" * 50)
    
    create_directories()
    
    if not check_requirements():
        print("\nError: Не все зависимости установлены")
        print("Установите зависимости командой: pip install -r requirements.txt")
        return
    
    if not download_yolo_model():
        print("\nError: Не удалось загрузить модель YOLO")
        return
    
    if not test_system():
        print("\nError: Система не прошла тестирование")
        return
    
    print("\nУстановка завершена успешно!")
    print("\nСледующие шаги:")
    print("1. Запустите приложение: python app.py")
    print("2. Выберите источник видео в боковой панели")
    print("3. Настройте параметры детекции")
    print("4. Нажмите 'Запуск обработки'")

if __name__ == "__main__":
    main()