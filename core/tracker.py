import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


class PeopleTracker:
    def __init__(self):
        self.tracks_history = {}
        self.next_track_id = 0
        self.tracker = DeepSort(
                max_age=50,
                n_init=3,
                max_cosine_distance=0.2,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True
            )
    

    def update(self, detections, frame):
        if not detections:
            return []
        
        return self._update_deep_sort(detections, frame)
    
    
    def _update_deep_sort(self, detections, frame):
        """Обновление через DeepSORT"""
        ds_detections = []
        for det in detections:
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            ds_detections.append(([bbox[0], bbox[1], width, height], det['confidence'], 'person'))
        
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        current_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            center = ((ltrb[0] + ltrb[2]) // 2, (ltrb[1] + ltrb[3]) // 2)
            
            if track_id not in self.tracks_history:
                self.tracks_history[track_id] = {'trajectory': [],
                                                 'total_frames': 0
                                                }
            
            # Обновление траектории
            self.tracks_history[track_id]['trajectory'].append(center)
            if len(self.tracks_history[track_id]['trajectory']) > 30:
                self.tracks_history[track_id]['trajectory'].pop(0)
            
            self.tracks_history[track_id]['total_frames'] += 1
            
            current_tracks.append({'track_id': track_id,
                                   'bbox': [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                                   'center': center,
                                   'confidence': getattr(track, 'get_det_conf', lambda: 0.9)(),
                                   'history': self.tracks_history[track_id]
                                  })
        
        return current_tracks
    
    
    def draw_tracks(self, frame, tracks):
        display_frame = frame.copy()
        
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            center = track['center']
            
            color = self._get_color(track_id)
            
            # Bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # ID и центр
            cv2.putText(display_frame, f"ID:{track_id}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(display_frame, center, 4, color, -1)
            
            # Траектория
            trajectory = track['history']['trajectory']
            for i in range(1, len(trajectory)):
                cv2.line(display_frame, trajectory[i-1], trajectory[i], color, 1)
        
        return display_frame
    
    def _get_color(self, track_id):
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        return colors[track_id % len(colors)]