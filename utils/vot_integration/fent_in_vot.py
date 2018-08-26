#!/usr/bin/python
import sys
import time

import vot

handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    handle.report(selection, confidence)
    time.sleep(0.01)
