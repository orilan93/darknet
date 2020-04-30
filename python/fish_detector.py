import cv2
import numpy as np
import ffmpeg
import darknet_fish as dn
from overlay import draw_overlay, draw_detections_color
import time
import experiments as ex
from smooth import smooth
from sort import Sort
import matplotlib.pyplot as plt

net = dn.load_net(b"data/fish.cfg", b"data/fish.weights", 0)
meta = dn.load_meta(b"data/fish.data")

VIDEO_URI = "FiskKlippet2.mp4"
WIDTH = 1920
HEIGHT = 1080
WINDOW_NAME = "Fish Species Detection"
USE_SORT = True
USE_OPENCV = False

if USE_OPENCV:
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)

process1 = (
    ffmpeg
        .input(VIDEO_URI, ss="245.6")
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True)
)

frame_num = 0

mot_tracker = Sort(max_age=3, min_hits=2)
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

    frame1 = draw_detections_color(frame, detections, 'red', text=True)  # actually blue
    ex.capture_timeframe(frame1, frame_num, 2)

    frame2 = draw_detections_color(frame, detections, 'red')

    if USE_SORT:
        detections = smooth(detections, mot_tracker)

    frame2 = draw_detections_color(frame2, detections, 'blue', text=True)
    ex.capture_timeframe(frame2, frame_num, 1)

    frame3 = draw_detections_color(frame, detections, 'blue', text=True)  # actually red
    ex.capture_timeframe(frame3, frame_num, 0)

    frame = draw_overlay(frame, detections, fps)
    ex.detection_difference(detections)
    ex.detections_count(detections)

    if frame_num >= 3:
        plt.draw()
        plt.savefig('sort_experiment.png', dpi=300)
        plt.show()
        break

    frame_num += 1

    if USE_OPENCV:
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            break

if USE_OPENCV:
    cv2.destroyAllWindows()
