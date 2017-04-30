import numpy as np
import sys
import os

sys.path.append(os.path.abspath('../OpenAI_Gym'))
import learning.tf_util as tf_util

class DensityNet(object):

    def __init__(self):
        return

    def sample(self):
        x = np.random.normal(0, 0.5)
        return x