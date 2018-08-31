import numpy as np

STATIC_FEATURES = [
    'conv14',
    # 'ColorName',
    # 'HoG',
]

# ratio of the size of search area to the size of the previous target
SEARCH_AREA_SIZE_RATIO = 2

# size of sample, corresponding to a search area, in which each 224x224 patch is a candidate patch
IMPUT_SAMPLE_SIZE = int(224 * SEARCH_AREA_SIZE_RATIO)
# size of static features
STATIC_FEATURE_SIZE = int(IMPUT_SAMPLE_SIZE / 2 / 2)
# size of the final output maps (cls & bbox reg)
OUTPUT_MAPS_SIZE = int(STATIC_FEATURE_SIZE / 2 / 2 / 2)

BATCH_SIZE = 2
assert BATCH_SIZE >= 2

MAX_NUM_SAMPLES = 10000

GAUSSIAN_LABEL_SIGMA = 0.36
GAUSSIAN_LABEL_AMPLITUDE = 1

INIT_TRAIN_ITER = 10
TRAIN_ITER_PER_ROUND = 5

MOMENTOM = 0.9
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01

ANCHORS = np.array([
    [1, 1],
    [2, 0.5],
    [0.5, 2],
    [0.5, 0.5],
    [1, 0.25],
    [0.25, 1],
    [2, 2],
    [4, 1],
    [1, 4],
]) / SEARCH_AREA_SIZE_RATIO

LOSS_WEIGHTS = {
    'cls': 0.8,
    'bbox_reg': 0.2
}
