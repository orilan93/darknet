from collections import defaultdict
from data import classes
import numpy as np

objects = defaultdict(lambda: [])


def smooth(detections, mot_tracker):
    xy_dets = np.empty((0, 7))
    for d in detections:
        bbox = d[2]
        class_id = classes.index(d[0])
        xy_dets = np.concatenate((xy_dets, [[bbox[0], bbox[1], bbox[2], bbox[3], d[1], 0, class_id]]))

    smooth_dets = mot_tracker.update(xy_dets)
    result_dets = []
    for e in smooth_dets:
        objects[int(e[4])].append(classes[int(e[5])])
        # Finds the most common class assigned to this tracked object
        most_common = max(set(objects[int(e[4])]), key=objects[int(e[4])].count)
        result_dets.append([most_common, 0, (e[0:4])])

    return result_dets
