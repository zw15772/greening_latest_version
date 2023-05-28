# coding='utf-8'
import sys
from HANTS import HANTS
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
results_root = 'D:/Greening/Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)


def tif2dict():
    fdir=rf'D:\Greening\Data\MODIS_LAI_MOD15A2H\\TIFF\\'
    outdir='D:\Greening\Data\MODIS_LAI_MOD15A2H\\DIC\\'


    NDVI_mask_f='C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
    array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
    array_mask[array_mask<0]=np.nan

    T.mk_dir(outdir,force=True)
    flist=os.listdir(fdir)
    all_array=[]
    year_list=list(range(2000,2023))  # 作为筛选条件
    for f in tqdm(sorted(flist),desc='loading...'):
        if f.startswith('.'):
            continue
        if not f.endswith('.tif'):
            continue
        print(f)
        # print(f.split('.')[0].split('_')[2][0:4])
        # print(f.split('.')[0][0:4])

        # if int(f.split('.')[0].split('_')[2][0:4]) not in year_list:  #
        #     continue

        if int(f.split('.')[0][0:4]) not in year_list:  #
            continue
        # if int(f.split('_')[2][0:4]) not in year_list:  #
        #     continue
        # print(f.split('.')[0][0:4])
        # if int(f.split('.')[0][0:4]) not in year_list:  #
        #     continue

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
        array = np.array(array, dtype=np.float)
        array=array[:360]  # PAR是361*720

        array[array<-999]=np.nan
        # array[array ==0] = np.nan
        # array[array < 0] = np.nan # 当变量是LAI 的时候，<0!!
        # plt.imshow(array)
        # plt.show()
        array_mask=np.array(array_mask,dtype=np.float)
        # plt.imshow(array_mask)
        # plt.show()
        array=array * array_mask
        # plt.imshow(array)
        # plt.show()

        # print(np.shape(array))
        # exit()
        all_array.append(array)
    # exit()

    row=len(all_array[0])
    col = len(all_array[0][0])
    key_list=[]
    dic={}

    for r in tqdm(range(row),desc='构造key'): #构造字典的键值，并且字典的键：值初始化
        for c in range(col):
            dic[(r,c)]=[]
            key_list.append((r,c))
    #print(dic_key_list)


    for r in tqdm(range(row),desc='构造time series'): # 构造time series
        for c in range(col):
            for arr in all_array:
                value=arr[r][c]
                dic[(r,c)].append(value)
            # print(dic)
    time_series=[]
    flag=0
    temp_dic={}
    for key in tqdm(key_list,desc='output...'): #存数据
        flag=flag+1
        time_series=dic[key]
        time_series=np.array(time_series)
        temp_dic[key]=time_series
        if flag %10000 == 0:
            # print(flag)
            np.save(outdir +'per_pix_dic_%03d' % (flag/10000),temp_dic)
            temp_dic={}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)