import numpy as np

STATIC_FEATURES = [
    'conv13',  # remove the pool2 layer
    # 'ColorName',
    # 'HoG',
]

# ratio of the size of search area to the size of the previous target
SEARCH_AREA_SIZE_RATIO = 2

PERCEPTIVE_FIELD_SIZE = 42
# size of sample, corresponding to a search area, in which each 224x224 patch is a candidate patch
INPUT_SAMPLE_SIZE = int(PERCEPTIVE_FIELD_SIZE * SEARCH_AREA_SIZE_RATIO)
# size of static features
STATIC_FEATURE_SIZE = int(INPUT_SAMPLE_SIZE / 2)
# size of the final output maps (cls & bbox reg)
OUTPUT_MAPS_SIZE = STATIC_FEATURE_SIZE

BATCH_SIZE = 2
assert BATCH_SIZE >= 2

MAX_NUM_SAMPLES = 10000

GAUSSIAN_LABEL_SIGMA = 0.36
GAUSSIAN_LABEL_AMPLITUDE = 1

INIT_TRAIN_ITER = 1
TRAIN_ITER_PER_ROUND = 1

MOMENTOM = 0.9
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.001

ANCHORS = np.array([
    (1, 1),
    (0.5, 0.5),
    (1, 0.5),
    (0.5, 1),
]) / SEARCH_AREA_SIZE_RATIO

LOSS_WEIGHTS = {
    'cls': 0.8,
    'bbox_reg': 0.2
}
