#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os.path
import numpy as np

for data_fn in glob("out/*.dat"):
    op = os.path.splitext(os.path.basename(data_fn))[0]
    frame = pd.read_csv(data_fn, names = ['n', 'opt', 'refr', 'halfopt'], header = None, index_col = 0)
    if np.isnan(frame.iloc[0]['halfopt']):
        frame.drop('halfopt', axis=1, inplace=True)
    ax = frame.plot(kind='line', title=op, xlim=(0, frame.iloc[-1].name))
    ax.set_ylabel("GFLOPS/s")
    plt.savefig(os.path.splitext(data_fn)[0] + '.png')
    plt.close()
