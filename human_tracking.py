import numpy as np
import cv2
import torch
import torchvision
from deepsort import DeepSort

class Human_Tracking:
    def __init__(self):
        self.deepsort = DeepSort(max_age=5)

    @classmethod
    def extract_features(cls, frame, detection):
        '''
            :param:
            frame : single frame from the video
            detection : list of dictionaries from frcnn model
        '''
        # extract the bounding boxes, labels and scores
        boxes, labels, scores = detection[0]["boxes"].detach().numpy(), 
        detection[0]["labels"].detach().numpy(), 
        detection[0]["scores"].detach().numpy()

        # extract features from boxes
        features = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            crop = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            feature = cls.deepsort.extract_feature(crop)
            features.append(feature)

        # associate the detections with existing tracks
        detections = np.column_stack((boxes, scores))
        tracked_objects = cls.deepsort.update(detections, features)

        return tracked_objects

    def update_tracker(frame, tracked_objects):
        for obj in tracked_objects:
            bbox = obj[:4]
            track_id = obj[4]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def track_movement(frame, tracked_objects):
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax = obj[:4]
            x = (xmax + xmin) / 2
            y = (ymax + ymin) / 2

            obj_velocity = np.array([x[1] - x[0], y[1] - y[0]])

            # determine if the object is still moving
            if np.linalg.norm(obj_velocity) > 0.1:
                obj_status = 1
            else:
                obj_status = 0
            
            # update the status
            obj.append(obj_status)

        return tracked_objects