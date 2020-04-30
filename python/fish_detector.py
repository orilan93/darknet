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
import matplotlib.ticker as ticker

net = dn.load_net(b"data/fish.cfg", b"data/fish.weights", 0)
meta = dn.load_meta(b"data/fish.data")

VIDEO_URI = "recording.mp4"
WIDTH = 1920
HEIGHT = 1080
WINDOW_NAME = "Fish Species Detection"
USE_SORT = True
USE_OPENCV = False
CROP_BOUNDS = (650, 300, 1250, 700)

if USE_OPENCV:
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)

process1 = (
    ffmpeg
    .input(VIDEO_URI, ss="2")
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

frame_num = 0
ax = plt.gca()
ax.set_xlim([0, 4])
ax.set_ylim([0, 3])
ax.get_yaxis().set_visible(False)
ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, 4.5, 1)))
ax.set_xticklabels(['t', 't+1', 't+2', 't+3'])
#ax.set_xticklabels(labels)


def capture_timeframe(frame, col, row):
    frame_fixed = frame[:, :, ::-1]
    frame_cropped = frame_fixed[CROP_BOUNDS[1]:CROP_BOUNDS[3],CROP_BOUNDS[0]:CROP_BOUNDS[2],:]
    ax.imshow(frame_cropped, extent=[col, col+1, row, row+1], origin='upper', aspect='auto')


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
    frame1 = draw_detections_color(frame, detections, 'red')
    capture_timeframe(frame1, frame_num, 0)
    if USE_SORT:
        detections = smooth(detections, mot_tracker)

    frame2 = draw_detections_color(frame1, detections,  'blue')
    capture_timeframe(frame2, frame_num, 1)

    frame3 = draw_detections_color(frame, detections,  'blue')
    capture_timeframe(frame3, frame_num, 2)

    #frame = draw_overlay(frame, detections, fps)
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