import json
import datetime
from enum import Enum
import os
import numpy as np
from collections import defaultdict

class EventType(Enum):
    VIOLATION = "violation"
    WARNING = "warning"
    INFO = "info"
    FALSE_POSITIVE = "false_positive"
    LEARNING_UPDATE = "learning_update"

class FeedbackLearner:
    """Компонент для обучения на ошибках оператора"""
    
    def __init__(self, model_path="models/feedback_model.json"):
        self.model_path = model_path
        self.false_positive_patterns = defaultdict(int)
        self.zone_adjustments = defaultdict(float)
        self.duration_adjustments = defaultdict(float)
        self.load_learning_data()
    
    def load_learning_data(self):
        """Загрузка данных обучения"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.false_positive_patterns = defaultdict(int, data.get('false_positive_patterns', {}))
                    self.zone_adjustments = defaultdict(float, data.get('zone_adjustments', {}))
                    self.duration_adjustments = defaultdict(float, data.get('duration_adjustments', {}))
        except Exception as e:
            print(f"Error loading learning data: {e}")
    
    def save_learning_data(self):
        """Сохранение данных обучения"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            data = {
                'false_positive_patterns': dict(self.false_positive_patterns),
                'zone_adjustments': dict(self.zone_adjustments),
                'duration_adjustments': dict(self.duration_adjustments),
                'last_updated': datetime.datetime.now().isoformat()
            }
            with open(self.model_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving learning data: {e}")
    
    def record_false_positive(self, violation_data, operator_feedback):
        """Запись ложного срабатывания и обучение на нем"""
        # Создание паттерна нарушения
        pattern = self._create_violation_pattern(violation_data)
        
        # Увеличение счетчика для этого паттерна
        self.false_positive_patterns[pattern] += 1
        
        # Адаптация параметров зоны
        zone_type = violation_data.get('zone_type', 'default')
        self.zone_adjustments[zone_type] += 0.1  # Увеличиваем порог для этой зоны
        
        # Адаптация длительности
        duration_key = f"{zone_type}_duration"
        self.duration_adjustments[duration_key] += 0.5  # Увеличиваем требуемую длительность
        
        # Сохранение обновленной модели
        self.save_learning_data()
        
        return {
            'pattern': pattern,
            'false_positive_count': self.false_positive_patterns[pattern],
            'zone_adjustment': self.zone_adjustments[zone_type],
            'duration_adjustment': self.duration_adjustments[duration_key]
        }
    
    def _create_violation_pattern(self, violation_data):
        """Создание уникального паттерна нарушения"""
        zone = violation_data.get('zone_type', 'default')
        severity = violation_data.get('violation_severity', 'medium')
        duration = int(violation_data.get('duration', 0))
        is_moving = violation_data.get('is_moving', False)
        
        return f"{zone}_{severity}_{duration}_{is_moving}"
    
    def get_adjusted_parameters(self, zone_type):
        """Получение скорректированных параметров для зоны"""
        base_threshold = 1.5  # Базовый порог
        base_duration = 5.0   # Базовая длительность
        
        zone_adjustment = self.zone_adjustments.get(zone_type, 0)
        duration_key = f"{zone_type}_duration"
        duration_adjustment = self.duration_adjustments.get(duration_key, 0)
        
        return {
            'distance_threshold': base_threshold + zone_adjustment,
            'violation_duration': base_duration + duration_adjustment
        }
    
    def should_suppress_alert(self, violation_data):
        """Проверка, следует ли подавить оповещение на основе истории"""
        pattern = self._create_violation_pattern(violation_data)
        fp_count = self.false_positive_patterns.get(pattern, 0)
        
        # Если этот паттерн часто отмечался как ложное срабатывание
        return fp_count >= 3

class EventLogger:
    def __init__(self, log_file="logs/events_log.json"):
        self.events = []
        self.log_file = log_file
        self.feedback_learner = FeedbackLearner()
        self._ensure_log_directory()
        self._load_existing_logs()
    
    def _ensure_log_directory(self):
        """Создает директорию для логов"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def _load_existing_logs(self):
        """Загрузка существующих логов"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                self.events = [json.loads(line) for line in f if line.strip()]

        except FileNotFoundError:
            print("Warning: Файл логов не найден, создается новый")
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write('')
                print("Success: Создан новый файл логов")
            except Exception as e:
                print(f"Error: Не удалось создать файл логов: {e}")

        except Exception as e:
            print(f"Error: Ошибка загрузки логов: {e}")
            self.events = []
    
    def log_event(self, event_type, message, details=None, track_ids=None):
        event = {'timestamp': datetime.datetime.now().isoformat(),
                 'type': event_type.value,
                 'message': message,
                 'details': details or {},
                 'track_ids': track_ids or []
                }
        
        self.events.append(event)
        
        # Проверка на ложные срабатывания через систему обучения
        if event_type == EventType.VIOLATION:
            should_suppress = self.feedback_learner.should_suppress_alert(details or {})
            if should_suppress:
                event['suppressed'] = True
                event['suppression_reason'] = "Частое ложное срабатывание"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error: Ошибка сохранения лога: {e}")
        
        return event
    
    def mark_false_positive(self, event_id, operator_comment=""):
        """Пометка события как ложного срабатывания"""
        for event in self.events:
            if event.get('timestamp') == event_id or event.get('id') == event_id:
                event['false_positive'] = True
                event['operator_comment'] = operator_comment
                event['corrected_at'] = datetime.datetime.now().isoformat()
                
                # Обучение на этом примере
                learning_result = self.feedback_learner.record_false_positive(
                    event.get('details', {}), 
                    operator_comment
                )
                
                # Логирование обучения
                self.log_event(
                    EventType.LEARNING_UPDATE,
                    "Система обучена на ложном срабатывании",
                    learning_result
                )
                
                # Обновление в файле
                self._rewrite_log_file()
                return True
        
        return False
    
    def _rewrite_log_file(self):
        """Перезапись файла логов"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for event in self.events:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error rewriting log file: {e}")
    
    def get_recent_events(self, count=10, event_type=None):
        events = self.events
        if event_type:
            events = [e for e in events if e['type'] == event_type.value]
        return events[-count:]
    
    def get_false_positive_stats(self):
        """Статистика ложных срабатываний"""
        false_positives = [e for e in self.events if e.get('false_positive')]
        total_events = len(self.events)
        
        return {
            'total_false_positives': len(false_positives),
            'total_events': total_events,
            'false_positive_rate': len(false_positives) / total_events if total_events > 0 else 0,
            'recent_false_positives': false_positives[-10:]  # Последние 10
        }
    
    def get_learning_adjustments(self):
        """Получение корректировок системы"""
        return {
            'zone_adjustments': dict(self.feedback_learner.zone_adjustments),
            'duration_adjustments': dict(self.feedback_learner.duration_adjustments),
            'false_positive_patterns': dict(self.feedback_learner.false_positive_patterns)
        }

# Создание глобального экземпляра логгера
global_logger = None

def get_logger():
    global global_logger
    if global_logger is None:
        global_logger = EventLogger()
    return global_logger

if __name__ == "__main__":
    # Тестирование системы логирования
    logger = EventLogger()
    
    # Тестовые события
    test_violation = {
        'person1': 1,
        'person2': 2,
        'distance': 1.2,
        'zone_type': 'corridor',
        'violation_severity': 'medium',
        'duration': 3.0,
        'is_moving': True
    }
    
    # Логирование тестового события
    event = logger.log_event(
        EventType.VIOLATION,
        "Тестовое нарушение дистанции",
        test_violation,
        [1, 2]
    )
    
    print(f"Создано событие: {event['timestamp']}")
    
    # Пометка как ложного срабатывания
    success = logger.mark_false_positive(
        event['timestamp'],
        "Оператор: это нормальное поведение в коридоре"
    )
    
    print(f"Пометка как ложного срабатывания: {success}")
    
    # Статистика
    stats = logger.get_false_positive_stats()
    print(f"Статистика ложных срабатываний: {stats}")
    
    # Корректировки системы
    adjustments = logger.get_learning_adjustments()
    print(f"Корректировки системы: {adjustments}")