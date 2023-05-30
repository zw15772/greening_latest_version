# coding='utf-8'
import sys

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
        self.nc_to_tif_Trendy()


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

            # plt.imshow(array)
            # plt.colorbar()
            # plt.show()
            to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array,
                                   ndv=-999999)


    def nc_to_tif_Trendy(self):
        fdir = data_root + 'Trendy/nc/'

        for f in os.listdir(fdir):
            if f.startswith('.'):
                continue

            outdir_name = f.split('.')[0]
            print(outdir_name)

            outdir = data_root+rf'/Trendy_TIFF/{outdir_name}//'
            Tools().mk_dir(outdir, force=True)
            yearlist = list(range(2000, 2023))

            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir + f, var_name='lai', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue


    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

        except:
            raise UserWarning('File not supported: ' + fname)
        # lon,lat = np.nan,np.nan
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year not in yearlist:
                continue
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = []
            lat_list = []
            value_list = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    lon_i = xx[i][j]
                    if lon_i > 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)


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
class Resample():
    def __init__(self):
        data_root = 'D:/Greening/Data/'
        pass
    def run(self):
        self.resample()

    def resample(self):
        fdir = data_root + 'Climate\Original_monthly\SMsurf\\'
        outdir = data_root + rf'Climate\resample\SMsurf\\'

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


class TIFtoDIC():
    def __init__(self):
        data_root = 'D:/Greening/Data/'
        pass
    def run(self):
        self.tif2dict()

    def tif2dict(self):
        fdir=data_root + rf'Climate\resample\SMroot\\'
        outdir=data_root+'\Climate\\DIC\\SMroot\\'


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


class interpolation():
    def __init__(self):
        data_root = 'D:/Greening/Data/'
        pass
    def run(self):
        self.interpolation_climate_data()

    def interpolation_climate_data(self):

        pass


class Phenology():

    def __init__(self):

        self.datadir_all = data_root

    def run(self):
        self.hants()
        # self.check_hants()
        # self.per_pixel_annual()
        # self.annual_phenology()

    def hants(self):

        outdir=self.datadir_all+'Hants_annually_smooth/'
        T.mkdir(outdir)
        fdir=self.datadir_all+'MODIS_LAI_MOD15A2H/DIC/'
        spatial_dic=T.load_npy_dir(fdir)
        tif_dir=self.datadir_all+'MODIS_LAI_MOD15A2H/TIFF/'
        date_list=[]
        for f in os.listdir(tif_dir):
            if f.endswith('.tif'):
                date=f.split('.')[0]
                y,m,d=date.split('_')
                y=int(y)
                m=int(m)
                d=int(d)
                date_obj=datetime.datetime(y,m,d)
                date_list.append(date_obj)

        hants365={}

        for pix in tqdm(spatial_dic,desc='hants'):
            r,c=pix
            if r>120:
                continue
            vals=spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            try:
                results=HANTS().hants_interpolate( values_list=vals, dates_list=date_list, valid_range=[0.001,10],nan_value=0)
            except:
                continue

            hants365[pix]=results
        np.save(outdir+'hants365.npy',hants365)

            # for year in results:
            #     result=results[year]
            #     plt.plot(result)
            #     plt.title(str(year))
            #     plt.show()

    def per_pixel_annual(self):
        fdir=data_root+'\\MODIS_LAI_MOD15A2H\\Hants_annually_smooth\\'
        outdir=data_root+'\MODIS_LAI_MOD15A2H\\\per_pix_annual\\'

        Tools().mk_dir(outdir)


        hants365_dic = dict(np.load(fdir+'hants365.npy', allow_pickle=True, ).item())

        for y in range(2000, 2023):
            outf = outdir + '{}.npy'.format(y)
            result_dic = {}
            for pix in hants365_dic:
                r,c=pix
                if r>120:
                    continue
                result = hants365_dic[pix]
                for year in result:
                    if year != y:
                        continue
                    result_i = result[year]

                if np.isnan(np.nanmean(result_i)):
                    continue
                if np.nanmean(result_i) == 0:
                    continue

                result_dic[pix] = result_i

            np.save(outf, result_dic)

    def check_hants(self):

        hants365=np.load('D:\Greening\Result\phenology\Hants_annually_smooth\LAI4g/2000.npy',allow_pickle=True).item()
        # for pix in hants365:
            # result=hants365[pix]
            # for year in result:
            #     result_i=result[year]
            #     print(len(result_i))
            #
            #     plt.plot(result_i)
            #     plt.title(pix)
            #     plt.show()
        for pix in hants365:
            result=hants365[pix]

            print(len(result))

            plt.plot(result)
            plt.title(pix)
            plt.show()


    def annual_phenology(self, threshold_i=0.2, ):
        fdir = data_root+'MODIS_LAI_MOD15A2H\phenology\per_pix_annual\\'

        out_dir =data_root+'MODIS_LAI_MOD15A2H\phenology\\annual_phenology\\'
        T.mkdir(out_dir, force=True)
        for f in T.listdir(fdir):
            year = int(f.split('.')[0])
            outf_i = join(out_dir, f'{year}.df')
            hants_smooth_f = join(fdir, f)
            hants_dic = T.load_npy(hants_smooth_f)
            result_dic = {}
            for pix in tqdm(hants_dic, desc=str(year)):
                vals = hants_dic[pix]

                result = self.pick_phenology(vals, threshold_i)
                result_dic[pix] = result
            df = T.dic_to_df(result_dic, 'pix')
            T.save_df(df, outf_i)
            T.df_to_excel(df, outf_i)
            np.save(outf_i, result_dic)

    def pick_phenology(self, vals, threshold_i):

        peak = np.nanargmax(vals)
        # if peak == 0 or peak == (len(vals) - 1):
        #     return {}
        # plt.plot(vals)
        # plt.show()
        # print(peak)
        # print(np.max(vals))
        # test=vals[peak]
        # print(test)

        if peak == 0 or peak == (len(vals) - 1):
            return {}
        try:
            early_start = self.__search_left(vals, peak, threshold_i)
            late_end = self.__search_right(vals, peak, threshold_i)
        except:
            early_start = 60
            late_end = 130
            print(vals)
            plt.plot(vals)
            plt.show()
        # method 1
        # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
        # method 2
        early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

        early_period = early_end - early_start
        peak_period = late_start - early_end
        late_period = late_end - late_start
        dormant_period = 365 - (late_end - early_start)

        result = {
            'early_length': early_period,
            'mid_length': peak_period,
            'late_length': late_period,
            'dormant_length': dormant_period,
            'early_start': early_start,
            'early_start_mon': self.__day_to_month(early_start),

            'early_end': early_end,
            'early_end_mon': self.__day_to_month(early_end),

            'peak': peak,
            'peak_mon': self.__day_to_month(peak),

            'late_start': late_start,
            'late_start_mon': self.__day_to_month(late_start),

            'late_end': late_end,
            'late_end_mon': self.__day_to_month(late_end),
        }
        return result
        pass

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.nanmin(left_vals)
        max_v = vals[maxind]
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.nanmin(right_vals)
        max_v = vals[maxind]
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __median_early_late(self, vals, sos, eos, peak):
        # 2 使用sos-peak peak-eos中位数作为sos和eos的结束和开始

        median_left = int((peak - sos) / 2.)
        median_right = int((eos - peak) / 2)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind

    def __day_to_month(self, doy):
        base = datetime.datetime(2000, 1, 1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

class Check_plot():

    def run(self):
        self.foo()

    def foo(self):

        # f='/Volumes/SSD_sumsang/project_greening/Result/detrend/extraction_during_late_growing_season_static/during_late_CSIF_par/per_pix_dic_008.npy'
        f = rf'D:\Greening\Data\Climate\DIC\SMsurf\per_pix_dic_014.npy'
        # f='/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_window/1982-2015_during_early/during_early_CO2.npy'
        result_dic = {}
        spatial_dic = {}
        # array = np.load(f)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)
        dic = dict(np.load(f, allow_pickle=True, encoding='latin1').item())
        # ///////check 字典是否不缺值////
        for pix in tqdm(dic, desc='interpolate'):
            r, c = pix
            # china_r=list(range(150,150))
            # china_c=list(range(550,620))

            # china_r = list(range(140, 570))
            # china_c = list(range(550, 620))
            # if r not in china_r:
            #     continue
            # if c not in china_c:
            #     continue
            print(len(dic[pix]))
            # exit()
            if len(dic[pix]) == 0:
                continue
            time_series = dic[pix]
            print(time_series)
            if len(time_series) == 0:
                continue
            # print(time_series)
            # time_series_reshape=time_series.reshape(12,-1)
            # time_series=np.array(time_series)
            plt.plot(time_series)
            # plt.imshow(time_series,aspect='auto')
            # plt.imshow(time_series)
            # plt.title(str(pix))
            plt.show()
            # spatial_dic[pix]=len(time_series)
        #     spatial_dic[pix] = time_series[0]
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.title(str(pix))
        # plt.show()

    pass

def main():
    nctotif().run()
    # Resample().run()
    # TIFtoDIC().run()
    # Check_plot().run()
    # Phenology().run()

    pass

if __name__ == '__main__':
    main()