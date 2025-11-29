import numpy as np
from scipy.spatial import distance
from enum import Enum
import json
import os
from datetime import datetime, timedelta

class RiskLevel(Enum):
    LOW = "Низкий"
    MEDIUM = "Средний"
    HIGH = "Высокий"

class ZoneType(Enum):
    CORRIDOR = "corridor"
    QUEUE = "queue" 
    OPEN_AREA = "open_area"
    ELEVATOR = "elevator"
    ENTRANCE = "entrance"
    DEFAULT = "default"

class ContextualZoneAnalyzer:
    """Анализатор контекстных зон с адаптивными порогами"""
    
    def __init__(self, config_path="config/zones.json"):
        self.config_path = config_path
        self.zones_config = self._load_zones_config()
        self.zone_detection_history = {}
        
    def _load_zones_config(self):
        """Загрузка конфигурации зон"""
        default_config = {
            "corridor": {
                "distance_threshold": 1.0,
                "density_threshold": 8.0,
                "speed_threshold": 1.5,
                "violation_duration": 3.0,
                "description": "Узкое пространство с направленным движением"
            },
            "queue": {
                "distance_threshold": 0.8,
                "density_threshold": 12.0,
                "speed_threshold": 0.3,
                "violation_duration": 10.0,
                "description": "Очередь с медленным движением"
            },
            "open_area": {
                "distance_threshold": 2.0,
                "density_threshold": 3.0,
                "speed_threshold": 2.0,
                "violation_duration": 2.0,
                "description": "Открытое пространство с свободным движением"
            },
            "elevator": {
                "distance_threshold": 0.6,
                "density_threshold": 15.0,
                "speed_threshold": 0.1,
                "violation_duration": 15.0,
                "description": "Лифт - кратковременное сближение допустимо"
            },
            "entrance": {
                "distance_threshold": 0.9,
                "density_threshold": 10.0,
                "speed_threshold": 1.0,
                "violation_duration": 5.0,
                "description": "Зона входа/выхода с переменным потоком"
            },
            "default": {
                "distance_threshold": 1.5,
                "density_threshold": 5.0,
                "speed_threshold": 1.0,
                "violation_duration": 5.0,
                "description": "Стандартная зона"
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Создаем директорию если не существует
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            print(f"Error loading zones config: {e}, using default")
            return default_config
    
    def detect_zone_type(self, tracks, frame_shape, movement_metrics):
        """
        Автоматическое определение типа зоны на основе анализа поведения
        """
        if not tracks:
            return ZoneType.DEFAULT
        
        people_count = len(tracks)
        frame_area = frame_shape[0] * frame_shape[1]
        density = people_count / (frame_area * 0.0001)  # чел/м²
        
        avg_speed = movement_metrics.get('avg_speed', 0)
        movement_ratio = movement_metrics.get('movement_ratio', 0)
        
        # Анализ траекторий для определения направления движения
        direction_consistency = self._calculate_direction_consistency(tracks)
        
        # Правила определения зон
        if density > 10 and avg_speed < 0.5 and movement_ratio < 0.3:
            return ZoneType.QUEUE
        elif direction_consistency > 0.7 and 0.5 < avg_speed < 2.0:
            return ZoneType.CORRIDOR
        elif density < 4 and avg_speed > 1.0:
            return ZoneType.OPEN_AREA
        elif density > 12 and avg_speed < 0.2:
            return ZoneType.ELEVATOR
        elif 0.3 < movement_ratio < 0.7 and direction_consistency < 0.5:
            return ZoneType.ENTRANCE
        else:
            return ZoneType.DEFAULT
    
    def _calculate_direction_consistency(self, tracks):
        """Расчет согласованности направления движения"""
        if len(tracks) < 2:
            return 0.0
        
        directions = []
        for track in tracks:
            trajectory = track['history']['trajectory']
            if len(trajectory) >= 2:
                # Расчет направления движения
                dx = trajectory[-1][0] - trajectory[0][0]
                dy = trajectory[-1][1] - trajectory[0][1]
                if dx != 0 or dy != 0:
                    direction = np.arctan2(dy, dx)
                    directions.append(direction)
        
        if not directions:
            return 0.0
        
        # Расчет согласованности направлений
        directions_array = np.array(directions)
        consistency = 1.0 - np.std(directions_array) / np.pi
        return max(0.0, consistency)
    
    def get_zone_parameters(self, zone_type):
        """Получение параметров для конкретного типа зоны"""
        zone_name = zone_type.value if isinstance(zone_type, ZoneType) else zone_type
        return self.zones_config.get(zone_name, self.zones_config['default'])

class ViolationDurationAnalyzer:
    """Анализатор длительности нарушений"""
    
    def __init__(self, max_history_seconds=300):
        self.max_history_seconds = max_history_seconds
        self.violation_history = {}
        self.false_positive_feedback = set()
    
    def update_violation_duration(self, violation, timestamp):
        """Обновление длительности нарушения"""
        violation_key = self._get_violation_key(violation)
        
        if violation_key in self.false_positive_feedback:
            return {'is_real_violation': False, 'duration': 0, 'reason': 'Помечено как ложное срабатывание'}
        
        if violation_key not in self.violation_history:
            self.violation_history[violation_key] = {
                'start_time': timestamp,
                'last_seen': timestamp,
                'violation_data': violation
            }
            duration = 0
        else:
            self.violation_history[violation_key]['last_seen'] = timestamp
            duration = (timestamp - self.violation_history[violation_key]['start_time']).total_seconds()
        
        # Очистка старых записей
        self._clean_old_violations(timestamp)
        
        return self._assess_violation_reality(violation, duration, timestamp)
    
    def _get_violation_key(self, violation):
        """Создание уникального ключа для нарушения"""
        persons = sorted([violation['person1'], violation['person2']])
        return f"{persons[0]}_{persons[1]}"
    
    def _clean_old_violations(self, current_time):
        """Очистка старых записей о нарушениях"""
        cutoff_time = current_time - timedelta(seconds=self.max_history_seconds)
        keys_to_remove = []
        
        for key, violation_data in self.violation_history.items():
            if violation_data['last_seen'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.violation_history[key]
    
    def _assess_violation_reality(self, violation, duration, timestamp):
        """Оценка реальности нарушения на основе длительности и контекста"""
        zone_type = violation.get('zone_type', 'default')
        required_duration = violation.get('required_duration', 5.0)
        
        is_moving = violation.get('is_moving', False)
        speed = violation.get('speed', 0)
        
        if duration < required_duration:
            return {
                'is_real_violation': False,
                'duration': duration,
                'reason': f'Кратковременное сближение ({duration:.1f}с < {required_duration}с)'
            }
        elif is_moving and speed > 0.5:
            return {
                'is_real_violation': False,
                'duration': duration,
                'reason': f'Динамическое сближение при движении (скорость: {speed:.1f} м/с)'
            }
        else:
            return {
                'is_real_violation': True,
                'duration': duration,
                'reason': f'Длительное статичное нарушение ({duration:.1f}с)'
            }
    
    def mark_false_positive(self, violation):
        """Пометка нарушения как ложного срабатывания"""
        violation_key = self._get_violation_key(violation)
        self.false_positive_feedback.add(violation_key)
        
        # Удаляем из истории
        if violation_key in self.violation_history:
            del self.violation_history[violation_key]

class TrajectoryAnalyzer:
    """Анализатор траекторий и движения"""
    
    def __init__(self, fps=25, pixel_to_meter=0.05):
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
        self.trajectory_history = {}
    
    def analyze_movement_patterns(self, tracks):
        """Анализ паттернов движения"""
        movement_analysis = {
            'speeds': [],
            'directions': [],
            'movement_patterns': [],
            'group_movements': [],
            'individual_movements': []
        }
        
        for track in tracks:
            trajectory = track['history']['trajectory']
            if len(trajectory) >= 2:
                # Анализ скорости и направления
                speed, direction = self._calculate_speed_direction(trajectory)
                movement_analysis['speeds'].append(speed)
                movement_analysis['directions'].append(direction)
                
                # Анализ паттерна движения
                pattern = self._classify_movement_pattern(trajectory, speed)
                movement_analysis['movement_patterns'].append(pattern)
        
        # Анализ группового движения
        if len(tracks) >= 2:
            movement_analysis['group_movements'] = self._analyze_group_movement(tracks)
        
        return movement_analysis
    
    def _calculate_speed_direction(self, trajectory):
        """Расчет скорости и направления движения"""
        if len(trajectory) < 2:
            return 0.0, 0.0
        
        # Используем последние 5 точек для расчета
        recent_points = trajectory[-5:] if len(trajectory) >= 5 else trajectory
        
        total_distance = 0
        total_dx, total_dy = 0, 0
        
        for i in range(1, len(recent_points)):
            dx = recent_points[i][0] - recent_points[i-1][0]
            dy = recent_points[i][1] - recent_points[i-1][1]
            segment_distance = np.sqrt(dx**2 + dy**2)
            
            total_distance += segment_distance
            total_dx += dx
            total_dy += dy
        
        if len(recent_points) > 1:
            avg_speed_px_per_frame = total_distance / (len(recent_points) - 1)
            avg_speed_m_s = avg_speed_px_per_frame * self.pixel_to_meter * self.fps
            
            # Среднее направление
            avg_dx = total_dx / (len(recent_points) - 1)
            avg_dy = total_dy / (len(recent_points) - 1)
            direction = np.arctan2(avg_dy, avg_dx) if (avg_dx != 0 or avg_dy != 0) else 0.0
            
            return avg_speed_m_s, direction
        
        return 0.0, 0.0
    
    def _classify_movement_pattern(self, trajectory, speed):
        """Классификация паттерна движения"""
        if len(trajectory) < 3:
            return 'unknown'
        
        if speed < 0.1:
            return 'stationary'
        elif speed < 0.5:
            return 'slow_moving'
        elif speed < 1.5:
            return 'walking'
        else:
            return 'fast_moving'
    
    def _analyze_group_movement(self, tracks):
        """Анализ группового движения"""
        group_analysis = []
        
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track1, track2 = tracks[i], tracks[j]
                
                # Проверка синхронности движения
                sync_score = self._calculate_movement_synchronization(
                    track1['history']['trajectory'],
                    track2['history']['trajectory']
                )
                
                if sync_score > 0.7:
                    group_analysis.append({
                        'person1': track1['track_id'],
                        'person2': track2['track_id'],
                        'synchronization': sync_score,
                        'type': 'group_movement'
                    })
        
        return group_analysis
    
    def _calculate_movement_synchronization(self, traj1, traj2):
        """Расчет синхронности движения двух траекторий"""
        min_len = min(len(traj1), len(traj2))
        if min_len < 2:
            return 0.0
        
        # Используем последние точки
        points_to_use = min(10, min_len)
        traj1_recent = traj1[-points_to_use:]
        traj2_recent = traj2[-points_to_use:]
        
        direction_correlations = []
        
        for k in range(1, len(traj1_recent)):
            dx1 = traj1_recent[k][0] - traj1_recent[k-1][0]
            dy1 = traj1_recent[k][1] - traj1_recent[k-1][1]
            
            dx2 = traj2_recent[k][0] - traj2_recent[k-1][0]
            dy2 = traj2_recent[k][1] - traj2_recent[k-1][1]
            
            # Нормализация векторов
            norm1 = np.sqrt(dx1**2 + dy1**2)
            norm2 = np.sqrt(dx2**2 + dy2**2)
            
            if norm1 > 0 and norm2 > 0:
                # Косинусное сходство
                cosine_sim = (dx1*dx2 + dy1*dy2) / (norm1 * norm2)
                direction_correlations.append(cosine_sim)
        
        return np.mean(direction_correlations) if direction_correlations else 0.0

class MetricsCalculator:
    def __init__(self, frame_width=640, frame_height=480, pixel_to_meter=0.05, fps=25):
        """
        Калькулятор метрик для анализа социальной дистанции
        
        Args:
            frame_width: ширина кадра
            frame_height: высота кадра
            pixel_to_meter: коэффициент преобразования пикселей в метры
            fps: кадров в секунду
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pixel_to_meter = pixel_to_meter
        self.fps = fps
        self.frame_area_m2 = (frame_width * pixel_to_meter) * (frame_height * pixel_to_meter)
        
        # Инициализация новых компонентов
        self.zone_analyzer = ContextualZoneAnalyzer()
        self.violation_duration_analyzer = ViolationDurationAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer(fps, pixel_to_meter)
        
        # Пороговые расстояния для разных зон (в метрах)
        self.distance_thresholds = {'default': 1.5,
                                    'corridor': 1.0,
                                    'queue': 0.8,
                                    'open_area': 2.0
                                   }
        
        # Время нарушения для классификации (в секундах)
        self.violation_duration_threshold = 5.0
        
        # История для анализа трендов
        self.metrics_history = []
        self.max_history_size = 100

    def calculate_pairwise_distances(self, tracks, frame_shape=(640, 480), movement_metrics=None):
        """
        Расчет попарных расстояний между всеми объектами с учетом контекста зоны
        
        Args:
            tracks: список треков
            frame_shape: размер кадра (ширина, высота)
            movement_metrics: метрики движения
            
        Returns:
            tuple: (список расстояний, список нарушений)
        """
        distances = []
        violations = []
        
        if len(tracks) < 2:
            return distances, violations
        
        # Автоматическое определение типа зоны
        if movement_metrics is None:
            movement_metrics = self.calculate_movement_metrics(tracks)
        
        zone_type = self.zone_analyzer.detect_zone_type(tracks, frame_shape, movement_metrics)
        zone_params = self.zone_analyzer.get_zone_parameters(zone_type)
        
        # Анализ паттернов движения
        movement_analysis = self.trajectory_analyzer.analyze_movement_patterns(tracks)
        
        # Получение центров всех объектов
        centers = [track['center'] for track in tracks]
        track_ids = [track['track_id'] for track in tracks]
        
        # Расчет попарных расстояний
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                # Евклидово расстояние в пикселях
                dist_pixels = distance.euclidean(centers[i], centers[j])
                
                # Конвертация в метры
                dist_meters = dist_pixels * self.pixel_to_meter
                distances.append(dist_meters)
                
                # Проверка нарушения дистанции с учетом зоны
                threshold = zone_params['distance_threshold']
                
                if dist_meters < threshold:
                    # Анализ движения для этой пары
                    is_moving = self._are_people_moving(tracks[i], tracks[j], movement_analysis)
                    speed = self._get_average_speed([tracks[i], tracks[j]], movement_analysis)
                    
                    violation = {'person1': track_ids[i],
                                 'person2': track_ids[j],
                                 'distance': dist_meters,
                                 'distance_pixels': dist_pixels,
                                 'violation_severity': self._calculate_violation_severity(dist_meters, threshold),
                                 'zone_type': zone_type.value,
                                 'threshold': threshold,
                                 'required_duration': zone_params['violation_duration'],
                                 'is_moving': is_moving,
                                 'speed': speed,
                                 'zone_description': zone_params['description']
                                }
                    
                    # Анализ длительности нарушения
                    current_time = datetime.now()
                    duration_analysis = self.violation_duration_analyzer.update_violation_duration(
                        violation, current_time
                    )
                    
                    violation.update(duration_analysis)
                    
                    violations.append(violation)
        
        return distances, violations

    def _are_people_moving(self, track1, track2, movement_analysis):
        """Проверка, движутся ли люди"""
        patterns = movement_analysis.get('movement_patterns', [])
        if len(patterns) >= 2:
            # Предполагаем, что паттерны соответствуют трекам
            return patterns[0] not in ['stationary', 'unknown'] or patterns[1] not in ['stationary', 'unknown']
        return True

    def _get_average_speed(self, tracks, movement_analysis):
        """Получение средней скорости для набора треков"""
        speeds = movement_analysis.get('speeds', [])
        if speeds and len(speeds) >= len(tracks):
            # Берем среднюю скорость для этих треков
            return np.mean(speeds[:len(tracks)])
        return 0.0

    def calculate_density(self, people_count):
        """
        Расчет плотности потока (человек/м^2)
        
        Args:
            people_count: количество людей
            
        Returns:
            float: плотность потока
        """
        if self.frame_area_m2 > 0:
            return people_count / self.frame_area_m2
        return 0.0

    def assess_risk_level(self, violations_count, density, zone_type='default'):
        """
        Оценка общего уровня риска
        
        Args:
            violations_count: количество нарушений
            density: плотность потока
            zone_type: тип зоны
            
        Returns:
            RiskLevel: уровень риска
        """
        # Получаем параметры зоны
        zone_params = self.zone_analyzer.get_zone_parameters(zone_type)
        
        # Весовые коэффициенты для разных факторов
        violation_weight = 0.6
        density_weight = 0.4
        
        # Нормализация нарушений с учетом зоны
        max_violations = 10  # базовое значение
        norm_violations = min(violations_count / max_violations, 1.0)
        
        # Нормализация плотности с учетом зоны
        density_threshold = zone_params.get('density_threshold', 5.0)
        norm_density = min(density / density_threshold, 1.0)
        
        # Расчет общей оценки уровня риска
        risk_score = (norm_violations * violation_weight + norm_density * density_weight)
        
        if risk_score >= 0.7:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def calculate_movement_metrics(self, tracks, fps=25):
        """
        Расчет метрик движения
        
        Args:
            tracks: список треков
            fps: кадров в секунду
            
        Returns:
            dict: метрики движения
        """
        movement_metrics = {'avg_speed': 0.0,
                            'moving_objects': 0,
                            'stationary_objects': 0,
                            'speeds': [],
                            'movement_ratio': 0.0
                           }
        
        if not tracks:
            return movement_metrics
        
        # Используем TrajectoryAnalyzer для более точного анализа
        movement_analysis = self.trajectory_analyzer.analyze_movement_patterns(tracks)
        
        speeds = movement_analysis.get('speeds', [])
        
        if speeds:
            movement_metrics['avg_speed'] = np.mean(speeds)
            movement_metrics['moving_objects'] = len([s for s in speeds if s > 0.1])  # > 0.1 м/с
            movement_metrics['stationary_objects'] = len(tracks) - movement_metrics['moving_objects']
            movement_metrics['speeds'] = speeds
            movement_metrics['movement_ratio'] = movement_metrics['moving_objects'] / len(tracks)
            
            # Добавляем дополнительную информацию из анализа траекторий
            movement_metrics['movement_patterns'] = movement_analysis.get('movement_patterns', [])
            movement_metrics['group_movements'] = movement_analysis.get('group_movements', [])
        
        return movement_metrics

    def _calculate_violation_severity(self, distance, threshold):
        """
        Расчет серьезности нарушения
        
        Args:
            distance: расстояние между людьми
            threshold: порог для текущей зоны
            
        Returns:
            str: уровень серьезности
        """
        ratio = distance / threshold
        
        if ratio < 0.33:
            return "critical"
        elif ratio < 0.66:
            return "high"
        elif ratio < 1.0:
            return "medium"
        else:
            return "low"

    def get_violation_context_data(self, violation):
        """
        Предоставляет данные о нарушении для генерации объяснений
        
        Args:
            violation: информация о нарушении
            
        Returns:
            dict: структурированные данные для объяснения
        """
        context = {'person1_id': violation['person1'],
                   'person2_id': violation['person2'],
                   'actual_distance': violation['distance'],
                   'required_distance': violation.get('threshold', 1.5),
                   'zone_type': violation.get('zone_type', 'default'),
                   'severity_level': violation['violation_severity'],
                   'is_violation': violation['distance'] < violation.get('threshold', 1.5),
                   'distance_difference': violation.get('threshold', 1.5) - violation['distance'],
                   'violation_duration': violation.get('duration', 0),
                   'is_real_violation': violation.get('is_real_violation', False),
                   'violation_reason': violation.get('reason', ''),
                   'zone_description': violation.get('zone_description', ''),
                   'is_moving': violation.get('is_moving', False),
                   'movement_speed': violation.get('speed', 0)
                  }
        
        # Генерация объяснимого текста
        context['explanation'] = self._generate_violation_explanation(context)
        
        return context
    
    def _generate_violation_explanation(self, context):
        """Генерация объяснимого текста о нарушении"""
        if not context['is_real_violation']:
            if context['violation_duration'] < context['required_distance']:
                return f"Кратковременное сближение ({context['violation_duration']:.1f}с) в {context['zone_description']}. Не является нарушением."
            elif context['is_moving']:
                return f"Динамическое сближение при движении (скорость: {context['movement_speed']:.1f} м/с). Естественное поведение в {context['zone_description']}."
            else:
                return f"Сближение в {context['zone_description']} не превышает допустимую длительность."
        else:
            return f"Реальное нарушение дистанции в {context['zone_description']}. Длительность: {context['violation_duration']:.1f}с. Расстояние: {context['actual_distance']:.2f}м при требуемых {context['required_distance']}м."

    def get_zone_thresholds(self):
        """
        Возвращает пороговые расстояния для всех зон
        
        Returns:
            dict: пороги для разных типов зон
        """
        return self.zone_analyzer.zones_config

    def get_movement_context(self, movement_metrics):
        """
        Предоставляет контекст движения для анализа
        
        Args:
            movement_metrics: метрики движения
            
        Returns:
            dict: структурированные данные о движении
        """
        return {'average_speed': movement_metrics.get('avg_speed', 0),
                'moving_people_count': movement_metrics.get('moving_objects', 0),
                'stationary_people_count': movement_metrics.get('stationary_objects', 0),
                'movement_ratio': movement_metrics.get('movement_ratio', 0),
                'is_crowd_moving': movement_metrics.get('avg_speed', 0) > 0.5,
                'movement_patterns_distribution': self._get_movement_patterns_distribution(movement_metrics),
                'group_movements_count': len(movement_metrics.get('group_movements', []))
               }
    
    def _get_movement_patterns_distribution(self, movement_metrics):
        """Получение распределения паттернов движения"""
        patterns = movement_metrics.get('movement_patterns', [])
        if not patterns:
            return {}
        
        distribution = {}
        for pattern in patterns:
            distribution[pattern] = distribution.get(pattern, 0) + 1
        
        # Нормализация
        total = len(patterns)
        for pattern in distribution:
            distribution[pattern] = distribution[pattern] / total
        
        return distribution

    def mark_false_positive(self, violation_data):
        """Пометка нарушения как ложного срабатывания (для обучения)"""
        self.violation_duration_analyzer.mark_false_positive(violation_data)

def test_metrics():
    """Тестирование калькулятора метрик"""
    calculator = MetricsCalculator()
    
    # Тестовые треки
    test_tracks = [
        {
            'track_id': 1,
            'center': (100, 100),
            'history': {'trajectory': [(100, 100), (101, 100), (102, 100)]}
        },
        {
            'track_id': 2, 
            'center': (150, 100),
            'history': {'trajectory': [(150, 100), (149, 100), (148, 100)]}
        },
        {
            'track_id': 3,
            'center': (300, 300),
            'history': {'trajectory': [(300, 300), (300, 300), (300, 300)]}
        }
    ]
    
    # Расчет метрик для разных зон
    print("Тест метрик:")
    print("=" * 50)
    
    zones_to_test = ['default', 'corridor', 'queue', 'open_area']
    
    for zone in zones_to_test:
        print(f"\n--- Тестирование зоны: {zone} ---")
        
        distances, violations = calculator.calculate_pairwise_distances(test_tracks, (640, 480))
        density = calculator.calculate_density(len(test_tracks))
        risk_level = calculator.assess_risk_level(len(violations), density, zone)
        movement = calculator.calculate_movement_metrics(test_tracks)
        
        print(f"  Людей: {len(test_tracks)}")
        print(f"  Нарушений: {len(violations)}")
        print(f"  Плотность: {density:.2f} чел/м²")
        print(f"  Уровень риска: {risk_level.value}")
        print(f"  Средняя скорость: {movement['avg_speed']:.2f} м/с")
        
        if violations:
            for i, violation in enumerate(violations):
                context_data = calculator.get_violation_context_data(violation)
                print(f"  Нарушение №{i+1}:")
                print(f"    - Люди: {context_data['person1_id']} и {context_data['person2_id']}")
                print(f"    - Дистанция: {context_data['actual_distance']:.2f}м")
                print(f"    - Требуется: {context_data['required_distance']}м")
                print(f"    - Зона: {context_data['zone_type']}")
                print(f"    - Серьезность: {context_data['severity_level']}")
                print(f"    - Объяснение: {context_data['explanation']}")
        
        if distances:
            print(f"  Расстояния между людьми: {[f'{d:.2f}м' for d in distances]}")
    
    # Тестирование контекстных данных
    print("\n--- Контекстные данные ---")
    zone_thresholds = calculator.get_zone_thresholds()
    print(f"Пороги для зон: {zone_thresholds}")
    
    if violations:
        sample_violation = violations[0]
        context = calculator.get_violation_context_data(sample_violation)
        print(f"Данные для объяснения нарушения: {context}")
    
    movement_context = calculator.get_movement_context(movement)
    print(f"Данные о движении: {movement_context}")
    
    return calculator

if __name__ == "__main__":
    test_metrics()