from collections import defaultdict
from data import classes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

last = defaultdict(int)
total_frames = 0
diff = 0
total_detections = 0


def detection_difference(detections):
    global total_frames, last, diff

    det_list = defaultdict(int)
    for d in detections:
        det_list[str.encode(d[0])] += 1

    if total_frames > 0:
        for c in classes:
            diff += abs(det_list[str.encode(c)] - last[str.encode(c)])

        diff_total = diff/total_frames
        print("Total diff: {}".format(diff_total))

    last = det_list
    total_frames += 1


def detections_count(detections):
    global total_detections

    total_detections += len(detections)
    average_detections = total_detections/total_frames
    print("Total detections: {}".format(total_detections))
    print("Average detections per frame : {}".format(average_detections))


CROP_BOUNDS = (200, 300, 700, 800)
ax = plt.gca()
ax.set_xlim([0, 4])
ax.set_ylim([0, 3])
ax.get_yaxis().set_visible(False)
ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, 4.5, 1)))
ax.set_xticklabels(['t', 't+1', 't+2', 't+3'])


def capture_timeframe(frame, col, row):
    frame_fixed = frame[:, :, ::-1]
    frame_cropped = frame_fixed[CROP_BOUNDS[1]:CROP_BOUNDS[3],CROP_BOUNDS[0]:CROP_BOUNDS[2],:]
    ax.imshow(frame_cropped, extent=[col, col+1, row, row+1], origin='upper', aspect='auto')