import numpy as np
from scipy import signal

import pandas as pd
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
##Subfunction: get_percentile
def get_percentile(limit):
    idx = np.argmin(abs(cpf_cum_probability - limit))
    return cpf_magnitude[idx]


import matplotlib.pyplot as plt
