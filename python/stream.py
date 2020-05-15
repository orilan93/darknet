import ffmpeg
import numpy as np

stream_params = {
    "format": "flv",
    "c:v": "libx264",
    "preset": "veryfast",
    "b:v": "3000k",
    "maxrate": "3000k",
    "bufsize": "6000k",
    "pix_fmt": "yuv420p",
    "g": 50,
    "c:a": "aac",
    "b:a": "160k",
    "ac": 2,
    "ar": 44100,
    "loglevel": "debug"
}

stream_params = {
    "format": "flv",
    "c:v": "libx264",
    "preset": "veryfast",
    "b:v": "1000k",
    "pix_fmt": "yuv420p",
    "g": 2,
    "s": "1280x720",
    "profile:v": "baseline",
    "loglevel": "debug"
}

STREAM_URI = "rtmp://"


def stream_process(width, height):
    return (ffmpeg
            .input(re=None, filename="pipe:", format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
            .filter("fps", 30)
            #.output('test.flv', **stream_params)
            .output(STREAM_URI, **stream_params)
            .run_async(pipe_stdin=True))


def stream_write(process, frame):
    process.stdin.write(
        frame
            .astype(np.uint8)
            .tobytes()
    )
    process.stdin.flush()


def stream_close(process):
    process.stdin.close()
    process.wait()
