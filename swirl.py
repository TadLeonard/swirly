import argparse
import logging
import sys

from imread import imread, imwrite
from moviepy.editor import VideoClip
import numpy as np

from . import effects


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="input image")
    parser.add_argument("video", type=str, help="output video")
    parser.add_argument("-d", "--duration", type=float, default=3.0)
    parser.add_argument("--compression", default="veryfast",
                        choices=("ultrafast", "veryfast", "fast",))
    parser.add_argument("--fps", type=int, help="frames per second",
                        default=24)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    args = parser.parse_args()
    img, metadata = read_img(args.image, return_metadata=True)

    frames = animations.clump_dark(img)
    #frames = animations.slide_colors(img)

    make_frame = frame_maker(frames)
    animation = VideoClip(make_frame, duration=args.duration)
    animation.write_videofile(args.video, fps=args.fps, audio=False,
                              preset=args.compression,
                              threads=args.ffmpeg_threads)
    imwrite("_{}_last.jpg".format("swirl"), img, metadata=metadata,
            opts={"quality": 100})


def read_img(path, return_metadata=False):
    img = np.squeeze(imread(path, return_metadata=return_metadata))
    if return_metadata:
        img, metadata = img
    if len(img.shape) == 2:
        _img = np.ndarray(img.shape + (3,), dtype=img.dtype)
        _img[:] = img[..., None]
        img = _img
    elif img.shape[-1] == 4:
        # we can't handle an RGBA array
        img = img[..., :3]
    logging.info("Initial image shape: {}".format(img.shape))
    logging.info("Working image shape: {}".format(img.shape))
    if return_metadata:
        return img, metadata
    else:
        return img


