from collections import defaultdict
from data import classes

last = defaultdict(int)
total_frames = 0
diff = 0


def detection_difference(detections):
    global total_frames, last, diff

    list = defaultdict(int)
    for d in detections:
        list[d[0]] += 1

    if total_frames > 0:
        for c in classes:
            diff += abs(list[str.encode(c)] - last[str.encode(c)])

        diff_total = diff/total_frames
        print("Total diff: {}".format(diff_total))

    last = list
    total_frames += 1