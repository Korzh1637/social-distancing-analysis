import numpy as np
from scipy.spatial import distance
from enum import Enum

class RiskLevel(Enum):
    LOW = "Низкий"
    MEDIUM = "Средний"
    HIGH = "Высокий"


class MetricsCalculator:
    def __init__(self, frame_width=640, frame_height=480, pixel_to_meter=0.05):
        """
        Калькулятор метрик для анализа социальной дистанции
        
        Args:
            frame_width: ширина кадра
            frame_height: высота кадра
            pixel_to_meter: коэффициент преобразования пикселей в метры
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pixel_to_meter = pixel_to_meter
        self.frame_area_m2 = (frame_width * pixel_to_meter) * (frame_height * pixel_to_meter)
        
        # Пороговые расстояния для разных зон (в метрах)
        self.distance_thresholds = {'default': 1.5,
                                    'corridor': 1.0,
                                    'queue': 0.8,
                                    'open_area': 2.0
                                   }
        
        # Время нарушения для классификации (в секундах)
        self.violation_duration_threshold = 5.0
        

    def calculate_pairwise_distances(self, tracks, zone_type='default'):
        """
        Расчет попарных расстояний между всеми объектами
        
        Args:
            tracks: список треков
            zone_type: тип зоны для определения порога
            
        Returns:
            tuple: (список расстояний, список нарушений)
        """
        distances = []
        violations = []
        
        if len(tracks) < 2:
            return distances, violations
        
        # Порог для конкретной зоны
        threshold = self.distance_thresholds.get(zone_type, 1.5)
        
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
                if dist_meters < threshold:
                    violation = {'person1': track_ids[i],
                                 'person2': track_ids[j],
                                 'distance': dist_meters,
                                 'distance_pixels': dist_pixels,
                                 'violation_severity': self._calculate_violation_severity(dist_meters),
                                 'zone_type': zone_type,
                                 'threshold': threshold
                                }
                    violations.append(violation)
        
        return distances, violations
    

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
        # Весовые коэффициенты для разных факторов
        violation_weight = 0.6
        density_weight = 0.4
        
        # Нормализация нарушений (макс кол-во нарушений = 10)
        norm_violations = min(violations_count / 10.0, 1.0)
        
        # Нормализация плотности
        density_thresholds = {'default': 5.0, 'corridor': 8.0, 'queue': 12.0, 'open_area': 3.0}
        max_density = density_thresholds.get(zone_type, 5.0)
        norm_density = min(density / max_density, 1.0)
        
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
                            'speeds': []
                           }
        
        if not tracks:
            return movement_metrics
        
        speeds = []

        for track in tracks:
            trajectory = track['history']['trajectory']

            if len(trajectory) >= 2:
                # Расчет скорости (пикселей/кадр)
                total_distance = 0

                for i in range(1, len(trajectory)):
                    dist = distance.euclidean(trajectory[i-1], trajectory[i])
                    total_distance += dist
                
                if len(trajectory) > 1:
                    avg_speed_px_per_frame = total_distance / (len(trajectory) - 1)
                    # Конвертация в м/с
                    avg_speed_m_s = avg_speed_px_per_frame * self.pixel_to_meter * fps
                    speeds.append(avg_speed_m_s)
        
        if speeds:
            movement_metrics['avg_speed'] = np.mean(speeds)
            movement_metrics['moving_objects'] = len([s for s in speeds if s > 0.1])  # > 0.1 м/с
            movement_metrics['stationary_objects'] = len(tracks) - movement_metrics['moving_objects']
            movement_metrics['speeds'] = speeds
        
        return movement_metrics
    

    def _calculate_violation_severity(self, distance):
        """
        Расчет серьезности нарушения
        
        Args:
            distance: расстояние между людьми
            
        Returns:
            str: уровень серьезности
        """
        if distance < 0.5:
            return "critical"
        elif distance < 1.0:
            return "high"
        elif distance < 1.5:
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
        return {'person1_id': violation['person1'],
                'person2_id': violation['person2'],
                'actual_distance': violation['distance'],
                'required_distance': violation.get('threshold', 1.5),
                'zone_type': violation.get('zone_type', 'default'),
                'severity_level': violation['violation_severity'],
                'is_violation': violation['distance'] < violation.get('threshold', 1.5),
                'distance_difference': violation.get('threshold', 1.5) - violation['distance']
               }
    

    def get_zone_thresholds(self):
        """
        Возвращает пороговые расстояния для всех зон
        
        Returns:
            dict: пороги для разных типов зон
        """
        return self.distance_thresholds.copy()
    

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
                'movement_ratio': movement_metrics.get('moving_objects', 0) / max(len(movement_metrics.get('speeds', [])), 1),
                'is_crowd_moving': movement_metrics.get('avg_speed', 0) > 0.5
               }

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
        
        distances, violations = calculator.calculate_pairwise_distances(test_tracks, zone)
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