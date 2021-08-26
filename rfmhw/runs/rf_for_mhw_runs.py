""" Module for Ulmo analysis on VIIRS 2013"""
import os
import numpy as np

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse


import h5py

from rfmhw import cleaned_rf_1902


from IPython import embed


def fit(pargs):
    cleaned_rf_1902.fit_me(pargs, file_unbal='./mat_unbalanced.csv',
        ncpu=pargs.ncpu, file_name='./movav_7_19new.csv',
                           outfile='./rf_random_last.pkl')

        
#### ########################## #########################
def main(pargs):

    # UMAP gallery
    if pargs.task == 'fit':
        fit(pargs)

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("SSL Figures")
    parser.add_argument("task", type=str, help="task to execute: 'fit'")
    parser.add_argument('--ncpu', type=int, default=10, help='Number of CPUs to run with')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)
