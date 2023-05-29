# coding='utf-8'
import sys
from HANTS import HANTS
import lytools
import pingouin
import pingouin as pg
# from green_driver_trend_contribution import *

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




this_root = 'D:/Greening/'
data_root = 'D:/Greening/Data/'
results_root = 'D:/Greening/Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class nctotif():

    def __init__(self):
        data_root = 'D:/Greening/Data/'

    def run (self):
        # self.nctotif_CRU()
        # self.nc_to_tif_GLEAM()
        self.tif2dict()

        # self.per_pixel_annual()

    def nctotif_CRU(self):  # 降雨需要乘以30天，温度不需要。

        fpath = data_root+'Climate\monthly\\Et_1980-2022_GLEAM_v3.7a_MO.nc'
        outdir = data_root + 'Climate\monthly\ET\\'

        T.mk_dir(outdir, force=True)
        nc = Dataset(fpath)
        print(nc)
        print(nc.variables.keys())
        variable = 'Et'
        t = nc['time']
        print(t)
        lat = nc['lat'][::-1]
        lon = nc['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]

        time = nc.variables['time']

        # print(time)
        # exit()
        # time_bounds = ncin.variables['time_bounds']
        # print(time_bounds)
        start = datetime.datetime(1900, 1, 1)
        # a = start + datetime.timedelta(days=5459)
        # print(a)
        # print(len(time_bounds))
        # print(len(time))
        # for i in time:
        #     print(i)
        # exit()
        # nc_dic = {}
        flag = 0

        for i in tqdm(range(len(time))):
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time[i]))
            year = str(date.year)
            year_int = int(year)
            month = '%02d' % date.month
            month_int = date.month
            days_number_of_one_month = T.number_of_days_in_month(year_int, month_int)
            # day = '%02d'%date.day
            date_str = year + month
            # print(date_str)
            # arr = ncin.variables[f'{variable}'][i][::-1]
            arr = nc.variables[f'{variable}'][i][::-1]
            arr = np.array(arr)
            # arr_monthly = arr * days_number_of_one_month  #only for precipitation *30
            arr_monthly = arr
            grid = arr < 99999
            arr_monthly[np.logical_not(grid)] = -999999
            newRasterfn = outdir + date_str + '.tif'
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight,
                                    arr_monthly)

    def nc_to_tif_GLEAM(self):
        fpath = data_root + 'Climate\monthly\\Et_1980-2022_GLEAM_v3.7a_MO.nc'
        outdir = data_root + 'Climate\monthly\ET\\'

        Tools().mk_dir(outdir, True)

        nc = Dataset(fpath)

        print(nc)
        print(nc.variables.keys())
        t = nc['time']
        print(t)
        start_year = int(t.units.split(' ')[-1].split('-')[0])


        basetime = datetime.datetime(start_year, 1, 1)  # 告诉起始时间
        lat_list = nc['lat']
        lon_list = nc['lon']
        # lat_list=lat_list[::-1]  #取反
        print(lat_list[:])
        print(lon_list[:])

        origin_x = lon_list[0]  # 要为负数-180
        origin_y = lat_list[0]  # 要为正数90
        pix_width = lon_list[1] - lon_list[0]  # 经度0.5
        pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
        print(origin_x)
        print(origin_y)
        print(pix_width)
        print(pix_height)
        # SIF_arr_list = nc['SIF']
        SPEI_arr_list = nc['Et']
        print(SPEI_arr_list.shape)
        print(SPEI_arr_list[0])
        # plt.imshow(SPEI_arr_list[5])
        # # plt.imshow(SPEI_arr_list[::])
        # plt.show()

        # date_list=list(range(1982,2021))
        # print(date_list)
        for i in range(len(SPEI_arr_list)):
            date_delta_i = t[i]
            print(date_delta_i)
            date_delta_i = datetime.timedelta(int(date_delta_i))
            # print(date_delta_i)
            date_i = basetime + date_delta_i
            print(date_i)
            # if date_i.split('-')[0] not in date_list :
            #     continue
            # print(date_i)
            year = date_i.year
            month = date_i.month
            date = date_i.day
            fname = ' {}{:02d}{:02d}.tif'.format(year, month, date)
            print(fname)
            newRasterfn = outdir + fname
            print(newRasterfn)
            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height
            # array = val
            array = SPEI_arr_list[i]
            array = np.array(array)
            # method 2
            #     array = array.T
            array[array < -10] = np.nan
            # plt.imshow(array)
            # plt.colorbar()
            # plt.show()
            to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array,
                                   ndv=-999999)
class resample():
    def __init__(self):
        data_root = 'D:/Greening/Data/'
        pass
    def run(self):
        pass

    def resample(self):
        fdir = data_root + 'Climate\Original_monthly\ET\\'
        outdir = data_root + 'Climate\resample\ET\\'

        T.mk_dir(outdir, force=True)
        year = list(range(2000, 2023))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir), ):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue

            # year_selection=f.split('.')[1].split('_')[1]
            # print(year_selection)
            # if not int(year_selection) in year:  ##一定注意格式
            #     continue
            # fcheck=f.split('.')[0]+f.split('.')[1]+f.split('.')[2]+'.'+f.split('.')[3]
            # if os.path.isfile(outdir+'resample_'+fcheck):  # 文件已经存在的时候跳过
            #     continue
            # date = f[0:4] + f[5:7] + f[8:10] MODIS
            print(f)
            # exit()
            date = f.split('.')[0]
            date_2 = date.split('_')[-1]
            print(date_2)

            # print(date)
            # exit()
            dataset = gdal.Open(fdir + f)
            # band = dataset.GetRasterBand(1)
            # newRows = dataset.YSize * 2
            # newCols = dataset.XSize * 2
            try:
                gdal.Warp(outdir + '{}.tif'.format(date_2), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass



    def tif2dict(self):
        fdir=data_root + 'Climate\monthly\ET\\'
        outdir=data_root+'\Climate\\DIC\\'



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



            if int(f.split('.')[0][0:5]) not in year_list:  #
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


    def per_pixel_annual(self):
        fdir=data_root+'\MODIS_LAI_MOD15A2H\\DIC\\'
        outdir=data_root+'\MODIS_LAI_MOD15A2H\\\per_pix_annual\\'


        Tools().mk_dir(outdir)
        dic = {}

        for f in tqdm(os.listdir(fdir)):
            # print(fi
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        # result_dic = DIC_and_TIF().void_spatial_dic()
        for y in range(2000, 2023):
            outf = outdir + '{}.npy'.format(y)
            result_dic = {}
            for pix in tqdm(dic, desc='{}'.format(y)):
                r,c = pix

                if r>120:
                    continue
                time_series = dic[pix]
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.nanmean(time_series) == 0:
                    continue
                # if np.nanmean(time_series) <= 0:
                #     continue
                time_series_reshape = np.reshape(time_series, (23, -1))
                year_series=time_series_reshape[y-2000]
                # print(len(year_series))
                # print(year_series)
                result_dic[pix] = year_series
                # plt.plot(year_series)
                # plt.show()

            np.save(outf, result_dic)



class Phenology():

    def __init__(self):

        self.datadir_all = data_root

    def run(self):
        self.hants()

    def hants(self):

        outdir=self.datadir_all+'Hants_annually_smooth/'
        T.mkdir(outdir)
        fdir=self.datadir_all+'MODIS_LAI_MOD15A2H/per_pix_annual/'

        params = []
        for y in T.listdir(fdir):
            params.append([outdir, y, fdir])
            self.kernel_hants([outdir, y, fdir])
        MULTIPROCESS(self.kernel_hants, params).run(process=4)

    def kernel_hants(self, params):
        outdir, y, fdir = params
        outf = join(outdir, y)
        dic = dict(np.load(fdir + y, allow_pickle=True, ).item())
        hants_dic = {}
        spatial_dic = {}
        for pix in tqdm(dic, desc=y):
            r, c = pix
            if r > 120:
                continue
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            if T.is_all_nan(vals):
                continue
            vals = T.interp_nan(vals)
            if len(vals) == 1:
                continue
            spatial_dic[pix] = 1
            xnew, ynew = self.__interp__(vals)
            std = np.nanstd(ynew)
            std = float(std)
            ynew = np.array([ynew])
            plt.plot(ynew[0])
            plt.show()
            results =HANTS(sample_count=365, inputs=ynew,
                            frequencies_considered_count=3,
                            outliers_to_reject='Hi',
                            low=-1, high=7,
                            fit_error_tolerance=np.std(ynew),
                            delta=0.1)
            result = results[0]
            plt.plot(result)
            plt.show()
            if T.is_all_nan(result):
                continue
            hants_dic[pix] = result
        T.save_npy(hants_dic, outf)

    def __interp__(self, vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new, y_new




def main():
    nctotif().run()
    resample().run()
    # Phenology().run()

    pass

if __name__ == '__main__':
    main()