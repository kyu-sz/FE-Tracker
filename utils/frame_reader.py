from enum import Enum

import cv2

from vot_integration.trax.server import ServerOptions


class SourceType(Enum):
    VIDEO_FILE = 1
    IMG_LIST = 2
    TRAX = 3


class FrameReader:
    def __init__(self, src: str, src_type: SourceType, trax_options: ServerOptions = None):
        self.src = src
        self.src_type = src_type

        if self.src_type is SourceType.TRAX:
            from vot_integration.trax import server
            self.trax_server = server.Server(trax_options)

    def __iter__(self):
        if self.src_type is SourceType.VIDEO_FILE:
            cap = cv2.VideoCapture(self.src)
            while cap.isOpened():
                ret, frame = cap.read()
                yield frame
        elif self.src_type is SourceType.IMG_LIST:
            with open(self.src) as f:
                for line in f:
                    yield cv2.imread(line.strip())
        elif self.src_type is SourceType.TRAX:
            yield self.trax_server.wait()
        else:
            raise NotImplementedError
