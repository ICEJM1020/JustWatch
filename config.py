import numpy as np

DATA_DIR = "/Users/timberzhang/Documents/Documents/2024-JustWatch/Data"
# DATA_DIR = "E:\\JustWatchData\\Data"

FILL_NAN = np.nan
CLUSTER_MIN_EPS = 25
CLUSTER_MIN_SAMPLE = 5
MIN_DTW_WINDOW = 2

VR_SCALE = 0.001207812
VR_ZDIST = 1
EYE_SAMPLE_RATE = 50
EYE_SAMPLE_TIME = (1 / EYE_SAMPLE_RATE) * 1000
SCREEN_SIZE = [1280, 720]

VIDEO_SIZE = [1920, 1080]
VIDEO_FPS = 30
