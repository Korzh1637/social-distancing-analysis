import json
import datetime
from enum import Enum

class EventType(Enum):
    VIOLATION = "violation"
    WARNING = "warning"
    INFO = "info"

class EventLogger:
    def __init__(self, log_file="logs/events_log.json"):
        self.events = []
        self.log_file = log_file
        self._ensure_log_directory()
        self._load_existing_logs()
    
    def _ensure_log_directory(self):
        """Создает директорию для логов, если ее не существует"""
        import os
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
        
        # сохранение в файл
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error: Ошибка сохранения лога: {e}")
        
        return event
    
    def get_recent_events(self, count=10, event_type=None):
        events = self.events
        if event_type:
            events = [e for e in events if e['type'] == event_type.value]
        return events[-count:]