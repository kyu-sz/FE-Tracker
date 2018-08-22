import sys

import cv2

from fent.tracker import Tracker
from utils.frame_reader import SourceType, FrameReader


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Filter evolution tracker')
    parser.add_argument('--img_seq_list',
                        help='input a video which is sliced into a sequence of images by a list of these images')
    parser.add_argument('--video_path',
                        help='input a video by specifying its path')
    parser.add_argument('label_file',
                        help='a label file which contains the bounding boxes of the target in each frame')

    return parser.parse_args()


class LabelReader:
    def __init__(self, label_file: str):
        self._label_file = label_file

    def __iter__(self):
        with open(self._label_file) as f:
            for line in f:
                yield [int(num_str) for num_str in line.split()]


if __name__ == '__main__':
    args = parse_arguments()
    if args.video_path:
        frame_reader = FrameReader(args.video_path, SourceType.VIDEO_FILE)
    elif args.img_seq_list:
        frame_reader = FrameReader(args.img_seq_list, SourceType.IMG_LIST)
    else:
        print('Require video file or image list as input for demo!')
        sys.exit(-1)

    label_reader = LabelReader(args.label_file)
    bundled_reader = zip(frame_reader, label_reader)
    tracker = None

    for frame, gt_bbox in zip(frame_reader, label_reader):
        if tracker is None:
            tracker = Tracker(frame, gt_bbox)
            continue
        bbox = tracker.track(frame)
        cv2.rectangle(frame, bbox[:1], bbox[2:], (255, 0, 0))
        cv2.rectangle(frame, gt_bbox[:1], gt_bbox[2:], (0, 255, 0))
        cv2.imshow(frame)
