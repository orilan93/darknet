import cv2
import numpy as np
import ffmpeg
import darknet_fish as dn
from overlay import draw_overlay, convert_bbox
import time
from experiments import detection_difference
from sort import *
import csv
from collections import defaultdict

#net = dn.load_net(b"data/yolov3.cfg", b"yolov3/yolov3_50000.weights", 0)
net = dn.load_net(b"data/yolov3_fish_orjan_416.cfg", b"yolov3_tree_416_40000.weights", 0)
meta = dn.load_meta(b"data/fish.data")

VIDEO_URI = "data/FiskKlippet2.mp4"
WIDTH = 1920
HEIGHT = 1080
WINDOW_NAME = "Fish Species Detection"
USE_SORT = False

def smooth(detections):
    xy_dets = np.empty((0, 7))
    for d in detections:
        bbox = d[2]
        class_id = species_names.index(d[0])
        xy_dets = np.concatenate((xy_dets,[[bbox[0],bbox[1],bbox[2],bbox[3],d[1],0,class_id]]))

    smooth_dets = mot_tracker.update(xy_dets)
    result_dets = []
    for e in smooth_dets:
        objects[int(e[4])].append(species_names[int(e[5])])
        # Finds the most common class assigned to this tracked object
        most_common = max(set(objects[int(e[4])]), key=objects[int(e[4])].count)
        result_dets.append([most_common,0,(e[0:4])])

    return result_dets

def fix_detections(detections):
    detections_new = []
    for d in detections:
        new_bbox = convert_bbox(d[2])
        detections_new.append((d[0].decode("ascii"),d[1],new_bbox))
    return detections_new

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)

process1 = (
    ffmpeg
    .input(VIDEO_URI)
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

species_names = []
with open("data/fish.names") as names_file:
    try:
        csv_reader = csv.reader(names_file, delimiter=',')
        species_names.extend(name[0] for name in csv_reader)
    except e:
        print(e)


mot_tracker = Sort(max_age=3, min_hits=4)
objects = defaultdict(lambda: [])
start = time.time()
while True:

    in_bytes = process1.stdout.read(WIDTH * HEIGHT * 3)
    if not in_bytes:
        break
    frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([HEIGHT, WIDTH, 3])
    )

    current = time.time()
    elapsed = current - start
    fps = 1 / elapsed
    start = current

    detections = dn.detect(net, meta, frame)
    detections = fix_detections(detections)

    if(USE_SORT):
        detections = smooth(detections)

    frame = draw_overlay(frame, detections, fps)
    detection_difference(detections)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()