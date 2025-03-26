import supervision as sv
from ultralytics import YOLO
import os
import bytetrack as bt
import pickle
import cv2 as cv
import numpy as np
class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)
        self.tracker= bt()
    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
   

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    def draw_box(self, video_frames, tracks):
        out_put_video = []
        for frame_num, frame in enumerate(video_frames):
            current_frame = frame.copy()
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referees_dict = tracks['referees'][frame_num]
            for track_id, player in player_dict.items():
                bbox = player['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv.rectangle(current_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv.putText(current_frame, f"Player {track_id}", (x1, y1-10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            for track_id, referee in referees_dict.items():
                bbox = referee['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv.rectangle(current_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(current_frame, f"Ref {track_id}", (x1, y1-10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for track_id, ball in ball_dict.items():
                bbox = ball['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                size = int(max(x2 - x1, y2 - y1))
            
                p1 = (center_x, center_y - size//2)  
                p2 = (center_x - size//2, center_y + size//2) 
                p3 = (center_x + size//2, center_y + size//2)  
            
            # Draw triangle
                triangle_pts = np.array([p1, p2, p3], np.int32)
                triangle_pts = triangle_pts.reshape((-1, 1, 2))
                cv.polylines(current_frame, [triangle_pts], True, (0, 255, 255), 2)
            
            # Add ball ID = 0
                cv.putText(current_frame, "0", (center_x - 10, center_y - size//2 - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

