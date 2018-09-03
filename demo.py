import sys

import cv2
from torchvision import models

from fent.tracker import Tracker
from utils import draw_bbox
from utils.frame_reader import SourceType, FrameReader


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Filter evolution tracker')
    parser.add_argument('--img_seq_list',
                        default='sequences/bolt1/img_list.txt',
                        help='input a video which is sliced into a sequence of images by a list of these images')
    parser.add_argument('--video_path',
                        help='input a video by specifying its path')
    parser.add_argument('--groundtruth',
                        default='sequences/bolt1/groundtruth.txt',
                        help='a label file which contains the bounding boxes of the target in each frame')

    return parser.parse_args()


class GroundtruthReader:
    def __init__(self, label_file: str):
        self._label_file = label_file

    def __iter__(self):
        with open(self._label_file) as f:
            for line in f:
                corners = [float(num_str) for num_str in line.split(',')]
                left = min(corners[::2])
                right = max(corners[::2])
                top = min(corners[1::2])
                bottom = max(corners[1::2])
                yield [left, top, right - left, bottom - top]


if __name__ == '__main__':
    args = parse_arguments()
    if args.video_path is not None:
        frame_reader = FrameReader(args.video_path, SourceType.VIDEO_FILE)
    elif args.img_seq_list is not None:
        frame_reader = FrameReader(args.img_seq_list, SourceType.IMG_LIST)
    else:
        print('Require video file or image list as input for demo!')
        sys.exit(-1)

    label_reader = GroundtruthReader(args.groundtruth)
    bundled_reader = zip(frame_reader, label_reader)
    tracker = None

    net = models.vgg16_bn(pretrained=True)

    patch_video_writer = None
    feature_video_writer = None
    for frame, gt_bbox in zip(frame_reader, label_reader):
        if tracker is None:
            tracker = Tracker(frame, gt_bbox)
            continue
        bbox = tracker.track(frame)

        draw_bbox(frame, bbox, (255, 0, 0))
        draw_bbox(frame, gt_bbox, (0, 255, 0))
        cv2.imshow("Demo", frame)

        # if patch_video_writer is None:
        #     patch_video_writer = cv2.VideoWriter('patch.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 50,
        #                                          (224, 224))
        #     feature_video_writer = cv2.VideoWriter('features.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 50,
        #                                            (224, 224))
        # offset = int(max(gt_bbox[2], gt_bbox[3]) / 2)
        # x_mid = int(gt_bbox[0] + gt_bbox[2] / 2)
        # y_mid = int(gt_bbox[1] + gt_bbox[3] / 2)
        # patch = cv2.resize(frame[y_mid - offset:y_mid + offset, x_mid - offset: x_mid + offset, :], (224, 224))
        # patch_video_writer.write(patch)
        #
        # features = cv2.resize(
        #     net.features[:-1](torch.stack([transforms.ToTensor()(patch)]))[0, 0, ...].detach().numpy(), (224, 224)
        # )
        # ceil = features.max()
        # feature_vis = np.zeros((224, 224, 3), 'uint8')
        # feature_vis[..., 1] = (ceil / 2 - np.fabs(ceil / 2 - features)) * 128
        # feature_vis[..., 2] = features / ceil * 255
        # feature_video_writer.write(feature_vis)
        # cv2.imshow("features", feature_vis)

        cv2.waitKey(1)

    cv2.destroyAllWindows()
