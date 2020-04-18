import cv2
import numpy as np
import ffmpeg
import darknet_fish as dn
from overlay import draw_overlay
import time

net = dn.load_net(b"data/fish.cfg", b"data/fish.weights", 0)
meta = dn.load_meta(b"data/fish.data")

VIDEO_URI = "recording.mp4"
WIDTH = 1920
HEIGHT = 1080

cv2.namedWindow("Fish Species Detection", cv2.WND_PROP_AUTOSIZE)

process1 = (
    ffmpeg
    .input(VIDEO_URI)
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

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
    print(detections)
    frame = draw_overlay(frame, detections, fps)

    cv2.imshow("Fish Species Detection", frame)
    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()