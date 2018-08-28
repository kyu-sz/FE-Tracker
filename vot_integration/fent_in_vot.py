#!/usr/bin/python
import sys
import time

from fent.tracker import Tracker
from . import vot

handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

fent = Tracker(imagefile, selection)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    selection, confidence = fent.track(imagefile)

    handle.report(selection, confidence)
    time.sleep(0.01)
