from pathlib import Path

import numpy as np

SPEECHCOMMAND_SR: int = 16000

SPEECHCOMMAND_CLASSES = (
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
    'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
    'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
)

MIN_WINDOW_MS = 60

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATA_SAMPLE_DIR = ROOT_DIR / "data_sample"

EPSILON = np.finfo(np.float64).eps
DEFAULT_BETA = 0.5

LEARNING_RATE = 0.01
TOPK = (1, 2, 5)
