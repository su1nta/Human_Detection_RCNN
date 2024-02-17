import cv2
import frcnn as fr
import human_tracking as ht
import store_result as sr
from torchvision import transforms as T

tensor_transform = T.ToTensor()
frcnn = fr.Faster_Rcnn()
human_track = ht.Human_Tracking()

vidcap = cv2.VideoCapture('Crowd.mp4')
frame_count = 0
output_table = []

while vidcap.isOpened():
    success,frame = vidcap.read()
    if not success:
        break
    frame_rs = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_ten = tensor_transform(frame_rs)

    detection = frcnn.human_detection(frame_ten)
    reduced_detection = human_track.extract_features(detection)
    tracked_objects = human_track.track_human(frame_rs, reduced_detection)
    updated_frame = human_track.update_frame(frame_rs, tracked_objects)

    tracking_table = human_track.track_movement(frame_count, tracked_objects)
    output_table.append(tracking_table)

    cv2.imshow('tracking', updated_frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
print(output_table)
sr.store_result(output_table)

vidcap.release()
cv2.destroyAllWindows()