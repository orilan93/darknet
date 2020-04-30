from collections import defaultdict
from data import classes

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
