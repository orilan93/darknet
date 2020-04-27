from functools import reduce
from collections import defaultdict
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from data import classes

font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 32)


colors = dict(zip(classes, [
    "blue",
    "cyan",
    "brown",
    "orange",
    "purple",
    "yellow",
    "pink",
    "red",
    "green"
]))


def draw_overlay(image, detections, fps, backend="pil"):
    if backend == "pil":
        return draw_overlay_pil(image, detections, fps)
    if backend == "cairo":
        return draw_overlay_cairo(image, detections, fps)


def draw_overlay_pil(image, detections, fps):
    pil_im = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_im)
    draw_detections_pil(draw, detections)
    draw_hud_pil(draw, detections)
    draw_fps_pil(draw, fps)
    return np.array(pil_im)


def draw_fps_pil(draw, fps):
    draw.text((1746, 0), "FPS: " + "{:.2f}".format(fps), font=font)


def draw_hud_pil(draw, detections):
    list = defaultdict(int)
    for d in detections:
        list[d[0]] += 1
    for index, key in enumerate(list):
        draw.text((8, 32+32*index), "{}: {}".format(key.decode('ascii'), list[key]), font=font, fill=(0, 0, 0))
    return


def draw_detections_pil(draw, detections):
    for d in detections:
        bbox = d[2]
        cx = bbox[0]
        cy = bbox[1]
        bw = bbox[2]
        bh = bbox[3]
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        draw.rectangle(((x1, y1), (x2, y2)), outline=colors[d[0].decode('ascii')], width=2)
        draw.text((x1, y1 - 32), d[0].decode('ascii'), font=font, fill=(0, 0, 0))


def draw_overlay_cairo(image, detections, fps):
    return
