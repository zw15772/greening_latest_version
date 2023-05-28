# coding='utf-8'
import sys

import pingouin
import pingouin as pg


version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import re
import to_raster
from osgeo import ogr
from osgeo import osr
from tqdm import tqdm
import datetime
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random
# import h5py
from netCDF4 import Dataset
import shutil
import requests
from lytools import *
from osgeo import gdal

from osgeo import gdal
# T=lytools.Tools()
# Tools=lytools.Tools


import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
T=Tools()



# project_root='/Volumes/SSD_sumsang/project_greening/'
# data_root=project_root+'Data/'
# result_root=project_root+'Result/new_result/'


this_root = 'D:/Greening/'
data_root = 'D:/Greening/Data/'
results_root = 'D:/Greening/Result_update/'
T.mk_dir(results_root,force=True)

def check_len():
    fdir = data_root + 'MODIS_LAI_MOD15A2H\TIFF\\'

    year_list = [str(i) for i in range(2000, 2023)]

    dic = {}


    for year in year_list:
        flag= 0

        for f in os.listdir(fdir):
            if f.endswith('.tif'):
                if f.split('.')[0][0:4] == str(year):
                    flag += 1
        dic[year] = flag
        print(year, flag)

def trend():
    fdir = data_root + 'MODIS_LAI_MOD15A2H\TIFF\\'

    year_list = [str(i) for i in range(2000, 2023)]


    average_all_year = []
    for year in year_list:
        arrs = []

        for f in os.listdir(fdir):
            if f.endswith('.tif'):
                f_year = f.split('.')[0][0:4]
                if not f_year == str(year):
                    continue
                f_month = f.split('_')[1]
                # if not f_month in [ '05', '06', '07', '08', '09', '10']:
                #     continue
                if not f_month in ['08','09', '10',]:
                    continue

                array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                array = np.array(array, dtype=np.float)
                array = array[:360]  # PAR是361*720

                array[array < -999] = np.nan
                array[array > 10] = np.nan
                arrs.append(array)
        average_arr = np.nanmean(arrs, axis=0)
        average_all_year.append(average_arr)

    # calculate trend
    dic_trend = {}
    dic_spatial = {}

    for r in tqdm(range(len(average_all_year[0])), desc=''):  # 构造time series
        for c in range(len(average_all_year[0][0])):

            r=int(r)
            c=int(c)

            val_list = []
            for arr in average_all_year:

                value = arr[r][c]
                val_list.append(value)
            if np.isnan(val_list).all():
                continue

            isnan = np.isnan(val_list)
            isnan_number=np.sum(isnan)
            if isnan_number>10:
                continue


            xaxis = list(range(len(val_list)))

            # print(val_list)
            # calculate trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(xaxis, val_list)

            # print(slope)
            dic_trend[(r,c)] = slope
            dic_spatial[(r,c)] = len(val_list)
    correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_trend)
    correlation_arr = np.array(correlation_arr)
    plt.imshow(correlation_arr, vmin=-0.01, vmax=0.01, cmap='jet', interpolation='nearest',)
    plt.colorbar()
    plt.show()













def main():
    # check_len()
    trend()

    pass


if __name__ == '__main__':
    main()






