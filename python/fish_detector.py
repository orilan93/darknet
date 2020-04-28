import cv2
import numpy as np
import ffmpeg
import darknet_fish as dn
from overlay import draw_overlay
import time
import experiments as ex
from smooth import smooth
from sort import Sort

net = dn.load_net(b"data/fish.cfg", b"data/fish.weights", 0)
meta = dn.load_meta(b"data/fish.data")

VIDEO_URI = "recording.mp4"
WIDTH = 1920
HEIGHT = 1080
WINDOW_NAME = "Fish Species Detection"
USE_SORT = True

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)

process1 = (
    ffmpeg
    .input(VIDEO_URI)
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

mot_tracker = Sort(max_age=10, min_hits=2)
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

    detections = dn.detect(net, meta, frame, thresh=.5, hier_thresh=.5, nms=.45)
    detections = dn.convert_detections(detections)

    if USE_SORT:
        detections = smooth(detections, mot_tracker)

    frame = draw_overlay(frame, detections, fps)
    ex.detection_difference(detections)
    ex.detections_count(detections)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()