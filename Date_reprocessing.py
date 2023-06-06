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
result_root = 'D:/Greening/Result/'

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
        fdir = data_root + 'D:\Greening\Data\LAI\\'

        for f in os.listdir(fdir):
            if not f.endswith('OCN_S2_nbp'):
                continue


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
        # self.resample()
        # self.resample_trendy()
        self.unify_TIFF()

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

    def resample_trendy(self):
        fdir_all = data_root + '\Trendy\Trendy_TIFF\\'
        for fdir in tqdm(os.listdir(fdir_all)):

            outdir = data_root + rf'Trendy\Trendy_resample\\{fdir}\\'

            T.mk_dir(outdir, force=True)
            year = list(range(2000, 2023))
            # print(year)
            # exit()
            for f in tqdm(os.listdir(fdir_all+fdir+'\\'), ):
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
                dataset = gdal.Open(fdir_all + fdir+'\\'+ f)
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]


                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(outdir + '{}.tif'.format(date_2), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass
    def unify_TIFF(self):
        fdir_all=data_root + rf'Trendy\Trendy_resample\\'


        for fdir in tqdm(os.listdir(fdir_all)):
            outdir = data_root + rf'Trendy\Trendy_unify\\{fdir}\\'
            T.mk_dir(outdir, force=True)
            for f in tqdm(os.listdir(fdir_all+fdir+'\\')):
                fpath=fdir_all+fdir+'\\'+f
                outpath=outdir+f
                if not f.endswith('.tif'):
                    continue
                if f.startswith('._'):
                    continue
                unify_tiff=DIC_and_TIF().unify_raster(fpath,outpath)



class TIFtoDIC():
    def __init__(self):
        data_root = 'D:/Greening/Data/'
        pass
    def run(self):
        self.tif2dict()
        # self.tif2dict_trendy()

    def tif2dict(self):
        fdir=data_root + rf'LAI\MCD_15A3H_TIFF\\'
        outdir=data_root+'LAI\MCD_15A3H_DIC\\'


        NDVI_mask_f='C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
        array_mask[array_mask<0]=np.nan

        T.mk_dir(outdir,force=True)
        flist=os.listdir(fdir)
        all_array=[]
        year_list=list(range(2003,2023))  # 作为筛选条件
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

    def tif2dict_trendy(self):
        fdir_all=data_root + rf'Trendy\Trendy_unify\\'

        NDVI_mask_f='C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
        array_mask[array_mask<0]=np.nan


        year_list=list(range(2000,2022))  # 作为筛选条件
        for fdir in tqdm(os.listdir(fdir_all),desc='loading...'):
            all_array = []

            outdir = data_root + '\Trendy\\DIC\\{}\\'.format(fdir)
            # if isdir(outdir):
            #     continue
            T.mk_dir(outdir, force=True)
            for f in tqdm(sorted(os.listdir(fdir_all+fdir)),desc='loading...'):
                if f.startswith('.'):
                    continue
                if not f.endswith('.tif'):
                    continue
                if isfile(outdir):
                    continue
                print(f)



                if int(f.split('.')[0][0:4]) not in year_list:  #
                    continue

                array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir_all+fdir +'\\' +f)
                array = np.array(array, dtype=np.float)
                #extract 360 and 720
                array_unify=array[:360][:360,:720] # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify<-999]=np.nan
                # array[array ==0] = np.nan
                array_unify[array_unify < 0] = np.nan # 当变量是LAI 的时候，<0!!
                # plt.imshow(array)
                # plt.show()
                array_mask=np.array(array_mask,dtype=np.float)
                # plt.imshow(array_mask)
                # plt.show()
                array_unify=array_unify * array_mask
                # plt.imshow(array)
                # plt.show()

                # print(np.shape(array))
                # exit()
                all_array.append(array_unify)
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
        # self.hants()
        # self.hants_trendy()
        # self.check_hants()
        # self.per_pixel_annual()
        # self.annual_phenology()
        # self.compose_annual_phenology()
        # self.data_clean()
        # self.average_phenology()
        self.pick_daily_phenology()
        # self.plot_phenology()

    def hants(self):

        outdir=self.datadir_all+'LAI//Hants_annually_smooth/MCD//'
        T.mkdir(outdir)
        fdir=self.datadir_all+'LAI\MCD_15A3H_DIC\\'
        spatial_dic=T.load_npy_dir(fdir)
        tif_dir=self.datadir_all+'LAI/MCD_15A3H_TIFF/'
        date_list=[]
        for f in os.listdir(tif_dir):
            if f.endswith('.tif'):
                date=f.split('.')[0]
                # y,m,d=date.split('_')
                y=date[0:4]
                m=date[4:6]
                d=date[6:8]
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
                # plt.plot(vals)
                # plt.show()

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
    def hants_trendy(self):


        fdir_all=self.datadir_all+'Trendy/DIC/'
        tif_dir_all = self.datadir_all + 'Trendy/trendy_unify/'
        date_list = []

        for fdir in os.listdir(fdir_all):
            tif_dir=tif_dir_all+fdir+'/'
            for f_tiff in os.listdir(tif_dir):
                if f_tiff.endswith('.tif'):
                    date=f_tiff.split('.')[0]

                    y=int(date[:4])
                    m=int(date[4:6])
                    d=int(date[6:8])
                    # format = '%Y%m%d'
                    date_obj = datetime.datetime.strptime(date, '%Y%m%d')
                    # date_obj=datetime.datetime(y,m,d)
                    date_list.append(date_obj)

            outdir = self.datadir_all + rf'Trendy/Hants_annually_smooth/{fdir}/'
            T.mkdir(outdir,force=True)

            hants365={}

            spatial_dic = T.load_npy_dir(fdir_all + fdir+ '/')

            for pix in tqdm(spatial_dic,desc='hants'):
                r,c=pix
                if r>120:
                    continue
                vals=spatial_dic[pix]

                if T.is_all_nan(vals):
                    continue
                try:
                    # plt.plot(vals)
                    # plt.show()
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
        fdir_all=data_root+'\LAI\Hants_annually_smooth\\'

        for fdir in os.listdir(fdir_all):
            if not fdir.startswith('MCD'):
                continue
            outdir = data_root + '\LAI\\\per_pix_annual\\' + fdir + '\\'
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all+fdir):
                if f.endswith('.npy'):
                    hants365_dic=np.load(fdir_all+fdir+'/'+f,allow_pickle=True).item()


            for y in range(2000, 2023):
                outf = outdir + '{}.npy'.format(y)
                result_dic = {}
                for pix in hants365_dic:
                    r,c=pix
                    if r>120:
                        continue
                    result = hants365_dic[pix]
                    # plt.plot(result)
                    # plt.show()
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

        hants365=np.load(rf'D:\Greening\Data\LAI\Hants_annually_smooth\IBIS_S2_lai\hants365.npy',allow_pickle=True).item()
        for pix in hants365:
            result=hants365[pix]
            print(len(result))
            exit()

            for year in result:

                result_i=result[year]
                print(len(result_i))

                plt.plot(result_i)
                plt.title(pix)
                plt.show()
        # for pix in hants365:
        #     result=hants365[pix]
        #
        #     print(len(result))
        #
        #     plt.plot(result)
        #     plt.title(pix)
        #     plt.show()


    def annual_phenology(self, threshold_i=0.2, ):
        fdir_all = data_root+'\LAI\\\per_pix_annual\\'
        for fdir in os.listdir(fdir_all):
            if not fdir=='MCD':
                continue

            out_dir =data_root+rf'\LAI\phenology\\annual_phenology\\{fdir}\\'
            T.mkdir(out_dir, force=True)

            for f in T.listdir(fdir_all + fdir):
                year = int(f.split('.')[0])

                outf_i = join(out_dir, f'{year}.df')
                hants_smooth_f = join(fdir_all, fdir, f)
                hants_dic = T.load_npy(hants_smooth_f)
                result_dic = {}
                for pix in tqdm(hants_dic, desc=str(year)):
                    r,c=pix
                    if r>120:
                        continue

                    vals = hants_dic[pix]
                    # plt.plot(vals)
                    # plt.show()
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

    def compose_annual_phenology(self, ):
        fdir_all = data_root+rf'LAI\phenology\annual_phenology\\'
        for fdir in os.listdir(fdir_all):
            if not fdir.startswith('MCD'):
                continue

            outdir = data_root + f'LAI/phenology/compose_annual_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'phenology_dataframe.df')

            all_result_dic = {}
            pix_list_all = []
            col_list = None
            for f in os.listdir(fdir_all + fdir):
                f = join(fdir_all, fdir, f)
                if not f.endswith('.df'):
                    continue
                df = T.load_df(f)
                pix_list = T.get_df_unique_val_list(df, 'pix')
                pix_list_all.append(pix_list)
                col_list = df.columns
            all_pix = []
            for pix_list in pix_list_all:
                for pix in pix_list:
                    all_pix.append(pix)
            pix_list = T.drop_repeat_val_from_list(all_pix)

            col_list = col_list.to_list()
            col_list.remove('pix')
            for pix in pix_list:
                dic_i = {}
                for col in col_list:
                    dic_i[col] = {}
                all_result_dic[pix] = dic_i
            # print(len(T.listdir(f_dir)))
            # exit()
            for f in tqdm(T.listdir(fdir_all + fdir)):
                if not f.endswith('.df'):
                    continue
                year = int(f.split('.')[0])
                df = T.load_df(join(fdir_all, fdir, f))
                dic = T.df_to_dic(df, 'pix')
                for pix in dic:
                    for col in dic[pix]:
                        if col == 'pix':
                            continue
                        all_result_dic[pix][col][year] = dic[pix][col]
            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)

    def data_clean(self,):  # 盖帽法

        f_dir = data_root + rf'LAI/phenology/compose_annual_phenology/MODIS//'
        outdir = data_root + rf'LAI/phenology/compose_annual_phenology_clean/MODIS//'
        T.mkdir(outdir, force=True)
        outf = join(outdir, 'phenology_dataframe.df')
        all_result_dic = {}
        pix_list_all = []

        for f in T.listdir(f_dir):
            if not f.endswith('.df'):
                continue
            df = T.load_df(join(f_dir, f))
            columns = df.columns
            column_list = []
            for col in columns:
                if col == 'pix':
                    continue
                column_list.append(col)
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
                # if lat > 70:
                #     continue
                # address = Tools().lonlat_to_address(lon, lat)
                # print(address)
                for col in column_list:
                    dic_i = row[col]

                    dic_clean = {}
                    # print(dic_i)
                    # values_list=dic_i.values()
                    # values_list=list(values_list)
                    series = pd.Series(dic_i)
                    cap_series = self.cap(series)
                    # print(series)
                    # plt.plot(series)
                    # # plt.title(address)
                    # plt.plot(cap_series)
                    #
                    # plt.show()
                    for year in dic_i:
                        dic_clean[year] = cap_series[year]
                    if pix not in all_result_dic:
                        all_result_dic[pix] = {}

                    all_result_dic[pix][col] = dic_clean
            # convert to df
            df_all = T.dic_to_df(all_result_dic, 'pix')
            #save
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)



    def cap(self, x, quantile=(0.05, 0.95)):

        """盖帽法处理异常值
        Args：
            x：pd.Series列，连续变量
            quantile：指定盖帽法的上下分位数范围
        """

        # 生成分位数
        Q01, Q99 = x.quantile(quantile).values.tolist()

        # 替换异常值为指定的分位数
        if Q01 > x.min():
            x = x.copy()
            x.loc[x < Q01] = Q01

        if Q99 < x.max():
            x = x.copy()
            x.loc[x > Q99] = Q99

        return (x)

    def average_phenology(self, ):   #将多年物候期平均
        fdir_all = data_root + f'LAI/phenology/compose_annual_phenology/'
        for fdir in os.listdir(fdir_all):
            if not 'MCD' in fdir:
                continue

            outdir = data_root + f'LAI/phenology/average_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'phenology_dataframe.df')

            all_result_dic = {}

            for f in T.listdir(fdir_all + fdir):
                if not f.endswith('.df'):
                    continue
                df = T.load_df(join(fdir_all,fdir, f))
                columns = df.columns
                column_list = []
                for col in columns:
                    if col == 'pix':
                        continue
                    column_list.append(col)

                pix_list = T.get_df_unique_val_list(df, 'pix')

                ########################################build dic##############################################################
                for pix in pix_list:
                    dic_i = {}
                    for col in column_list:
                        dic_i[col] = {}
                    all_result_dic[pix] = dic_i

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    pix = row['pix']
                    lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
                    # address=Tools().lonlat_to_address(lon,lat)
                    # print(address)
                    for col in column_list:
                        dic_i = row[col]
                        # print(dic_i)
                        values = dic_i.values()
                        values = list(values)
                        value_mean = np.mean(values)
                        value_ = round(value_mean, 0)
                        value_std = np.std(values)
                        all_result_dic[pix][col] = value_

            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)

    def pick_daily_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]
        fdir_all = data_root + f'LAI/phenology/average_phenology/'
        for fdir in os.listdir(fdir_all):
            if not fdir.endswith('MCD'):
                continue
            outdir = data_root + f'LAI/phenology/pick_daily_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'pick_daily_phenology.df')

            phenology_df = T.load_df(
                f'{fdir_all}/{fdir}/phenology_dataframe.df')

            early_dic = {}
            peak_dic = {}
            late_dic = {}
            all_result_dic = {}

            for i, row in tqdm(phenology_df.iterrows(), total=len(phenology_df)):
                pix = row['pix']
                all_result_dic[pix] = {}
                early_start = row['early_start']
                early_end = row['early_end']
                peak_start = row['early_end']
                peak_end = row['late_start']
                late_start = row['late_start']
                late_end = row['late_end']
                early_period = np.arange(int(early_start), int(early_end), 1)
                # print(early_period)
                peak_period = np.arange(int(early_end), int(late_start), 1)
                # print(peak_period)
                late_period = np.arange(int(late_start), int(late_end), 1)
                # print(late_period)
                all_result_dic[pix]['early'] = early_period
                all_result_dic[pix]['peak'] = peak_period
                all_result_dic[pix]['late'] = late_period
                all_result_dic[pix]['early_peak'] = np.concatenate((early_period, peak_period))
                all_result_dic[pix]['early_peak_late'] = np.concatenate((early_period, peak_period, late_period))
                # print(all_result_dic[pix])


            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)


    def plot_phenology(self):
        # df_f=data_root+r'MODIS_LAI_MOD15A2H\phenology\annual_phenology\\2015.df'
        df_f=data_root+r'Trendy\phenology\annual_phenology\MODIS\\2021.df'

        df=T.load_df(df_f)
        colunms='early_start'
        spatial_dic=T.df_to_spatial_dic(df,colunms)
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,vmin=100,vmax=160,cmap='jet')
        plt.colorbar()
        plt.show()





class Check_plot():

    def run(self):
        self.foo()

    def foo(self):

        # f='/Volumes/SSD_sumsang/project_greening/Result/detrend/extraction_during_late_growing_season_static/during_late_CSIF_par/per_pix_dic_008.npy'
        f = rf'D:\Greening\Data\LAI\Hants_annually_smooth\ISAM_S2_lai\hants365.npy'
        # f = rf'D:\Greening\Data\Trendy\DIC\\CABLE-POP_S2_lai\per_pix_dic_014.npy'
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

class process_LAI():

    def __init__(self):
        data_root = 'D:/Greening/Data/'
        result_root='D:/Greening/Result/'
        pass

    def run(self):

        # self.data_transform()
        self.extraction_variables_static_during_daily()


    def data_transform(self):  # 将hants365 数据转换成一个pix 23 年数据

        fdir_all = data_root + f'LAI\\Hants_annually_smooth\\'
        for fdir in os.listdir(fdir_all):
            if not fdir.startswith('MCD'):
                continue

            outdir = data_root + f'original_dataset/{fdir}/'
            T.mkdir(outdir, force=True)

            annual_pheno_dict = {}
            for f in T.listdir(fdir_all + fdir):
                if not f.endswith('.npy'):
                    continue
                fpath = os.path.join(fdir_all, fdir, f)
                hants365_dic = T.load_npy(fpath)


                result_dic = {}

                for pix in tqdm(hants365_dic):
                    result_all = []
                    r, c = pix
                    if r > 120:
                        continue
                    result = hants365_dic[pix]

                    for year in result:

                        result_i = result[year]

                        if np.isnan(np.nanmean(result_i)):
                            continue
                        if np.nanmean(result_i) == 0:
                            continue
                        result_all.append(result_i)

                    result_dic[pix] = result_all
                #save data

                time_series = []
                temp_dic = {}
                flag = 0
                for key in tqdm(result_dic, desc='output...'):  # 存数据
                    flag = flag + 1
                    time_series = result_dic[key]
                    time_series = np.array(time_series)
                    temp_dic[key] = time_series

                    if flag % 10000 == 0:
                        # print(flag)
                        np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                        temp_dic = {}
                np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)


                # np.save(outf, result_dic)

    def extraction_variables_static_during_daily(self):  # 静态提取during multiyear


        variables_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',  'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai', 'JULES_S2_lai',
                          'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai','SDGVM_S2_lai',
                          'ORCHIDEE_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
                          'Trendy_ensemble',]

        variables_list=['MCD']

        for variable in variables_list:
            phenology_df = T.load_df(data_root + rf'LAI/phenology/pick_daily_phenology/{variable}/pick_daily_phenology.df')

            fdir=data_root+rf'original_dataset\{variable}\\'

            dic_variables = {}

            # 加载变量数据
            for f in os.listdir(fdir):
                if not f.endswith('.npy'):
                    continue
                dic_i=dict(np.load(fdir+f, allow_pickle=True, encoding='latin1' ).item())
                dic_variables.update(dic_i)

            period_list=['early','peak','late','early_peak','early_peak_late']


            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root +rf'extraction_original_val/LAI/{variable}/'

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):

                    if pix not in dic_period:
                        continue
                    picked_daily = dic_period[pix]   #  修改这里
                    if picked_daily[0]<=0:
                        continue
                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]
                    time_series_flatten=time_series.flatten()
                    # print(time_series_flatten)
                    time_series_flatten=np.array(time_series_flatten)
                    if len(time_series_flatten)!=20*365:
                        continue
                    print(len(time_series_flatten))


                    picked_daily = np.array(picked_daily, dtype=int)


                    for year in range(20):  # 修改


                        during_time_series = time_series[year][picked_daily]
                        # print(picked_month)#!!!!!

                        during_time_series=np.array(during_time_series, dtype=float)

                        during_time_series[during_time_series < -99.] = np.nan

                        # if np.isnan(np.nanmean(during_time_series)):  # 修改
                        #     continue

                        # variable_sum = np.nansum(during_time_series)
                        # dic_during_variables[pix].append(variable_sum)
                        variable_mean = np.nanmean(during_time_series)  # !!! 降雨需要是sum  # 其他变量是平均值 nanmean
                        dic_during_variables[pix].append(variable_mean)

                    dic_spatial_count[pix] = len(dic_during_variables[pix])
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                # plt.imshow(arr, cmap='jet')
                # plt.colorbar()
                # plt.title('')
                # plt.show()
                np.save(outdir + 'during_{}_{}'.format(period,variable), dic_during_variables)  # 修改

class statistic_analysis():
    def __init__(self):
        pass
    def run(self):
        # self.trend_analysis()
        self.detrend_zscore()


    def trend_analysis(self):

        dic_mask_lc_file = 'C:/Users/pcadmin/Desktop/Data/Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = 'C:/Users/pcadmin/Desktop/Data/Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)


        outdir=result_root+'trend_analysis\\'
        Tools().mk_dir(outdir, True)

        fdir = result_root + rf'extraction_original_val/LAI/MCD/'
        for f in os.listdir(fdir):
            outf=outdir+f.split('.')[0]
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            trend_dic = {}
            p_value_dic = {}
            for pix in dic_mask_lc:
                if pix not in dic:
                    continue
                time_series = dic[pix]
                if len(time_series) != 20:
                    continue
                if dic_mask_lc[pix] == 'Crop':
                    continue
                val_lc_change = array_mask_landcover_change[pix]
                if val_lc_change == np.nan:
                    continue
                time_series = np.array(time_series)

                time_series[time_series < -99.] = np.nan
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                trend_dic[pix] = slope
                p_value_dic[pix] = p_value

            arr_trend = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
            p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(p_value_dic)

            DIC_and_TIF().arr_to_tif(arr_trend, outf + '_trend.tif')
            DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

            # plt.imshow(arr_trend, cmap='jet', vmin=-0.03, vmax=0.03)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

    pass

    def detrend_zscore(self): #

        dic_mask_lc_file = 'C:/Users/pcadmin/Desktop/Data/Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = 'C:/Users/pcadmin/Desktop/Data/Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        product_list=  ['MCD']

        for period in ['early', 'peak', 'late', 'early_peak', 'early_peak_late']:
            outdir = result_root + rf'detrend_zscore\\{period}\\'
            for product in product_list:
                outf=outdir+product+'.npy'
                # print(outf)
                # exit()
                f = result_root + rf'extraction_original_val\LAI\{product}\\during_{period}_MCD.npy'

                Tools().mk_dir(outdir,force=True)
                dic = {}


                dic = dict(np.load( f, allow_pickle=True, ).item())


                detrend_zscore_dic={}

                for pix in tqdm(dic):

                    val_lc_change = array_mask_landcover_change[pix]
                    if val_lc_change < -9999:
                        continue
                    if pix not in dic_mask_lc:
                        continue
                    if dic_mask_lc[pix] == 'Crop':
                        continue
                    if array_mask_landcover_change[pix] == np.nan:
                        continue
                    time_series = dic[pix]
                    time_series=np.array(time_series)
                    # plt.plot(time_series)
                    # plt.show()

                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue

                    delta_time_series = []
                    mean = np.nanmean(time_series)
                    std=np.nanstd(time_series)
                    delta_time_series = (time_series - mean) / std

                    detrend_delta_time_series = signal.detrend(delta_time_series)

                    # plt.plot(detrend_delta_time_series)
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series

                np.save(outf, detrend_zscore_dic)

    def pick_greening_year_extraction_corresponding_variables_threshold(self): # 通过greening qu

        f_early_peak_LAI = results_root + rf'\detrend\detrend_zscore\1982_2018\Y\detrend_early_peak_MODIS_LAI_zscore.npy'
        f_late_LAI = results_root + rf'\detrend\detrend_zscore\1982_2018\Y\detrend_late_MODIS_LAI_zscore.npy'
        outdir = results_root + rf'Dataframe/pick_greening_year_extraction_corresponding_variables_threshold//'
        T.mk_dir(outdir, force=1)
        dic_early_peak_LAI = dict(np.load(f_early_peak_LAI, allow_pickle=True, ).item())
        dic_late_LAI = dict(np.load(f_late_LAI, allow_pickle=True, ).item())

        fdir_all = results_root+rf'detrend\detrend_zscore_monthly_1982_2018\\'
        picked_year_list =list(range(2000,2018+1))
        result_all_dic={}
        threshold = 0.5
        outf=outdir+rf'{threshold}.df'

        for f in T.listdir(fdir_all):
            print(f)

            dic_daily = dict(np.load(fdir_all + f, allow_pickle=True, ).item())

            spatial_dic_amplifying={}
            spatial_dic_weak_stabilizing={}
            spatial_dic_strong_stabilizing={}




            for pix in dic_early_peak_LAI:

                early_peak_LAI = dic_early_peak_LAI[pix]
                if not pix in dic_late_LAI:
                    continue
                late_LAI=dic_late_LAI[pix]
                amplifying_list = []
                weak_stabilizing_list = []
                strong_stabilizing_list = []

                for i in range(len(early_peak_LAI)):
                    early_peak_LAI_i = early_peak_LAI[i]
                    late_LAI_i = late_LAI[i]
                    if early_peak_LAI_i <threshold:
                        continue
                    if late_LAI_i > early_peak_LAI_i:
                        classification = 'amplifying'
                        amplifying_list.append(i)
                    elif late_LAI_i > 0:
                        classification = 'weak stabilizing'
                        weak_stabilizing_list.append(i)
                    else:
                        classification = 'strong stabilizing'
                        strong_stabilizing_list.append(i)


                amplifying_list=np.array(amplifying_list)+2000
                weak_stabilizing_list=np.array(weak_stabilizing_list)+2000
                strong_stabilizing_list=np.array(strong_stabilizing_list)+2000

                if not pix in dic_daily:
                    continue

                time_series_reshape=dic_daily[pix]
                daterange_dict=self.variable_daterange()

                daterange = daterange_dict[f.split('.')[0]]
                # fname=f.split('_')[-2]
                # daterange = daterange_dict[daterange]
                yearrange_list=list(range(daterange[0],daterange[1]+1))
                time_series_reshape_dict=T.dict_zip(yearrange_list,time_series_reshape)


                selection_list_amplifying = []

                # selection_dic={}
                for yr in amplifying_list:
                    if yr<2000:
                        continue
                    selection = time_series_reshape_dict[yr]
                    # selection_dic[key]=selection
                    selection_list_amplifying.append([yr, selection])
                # plt.plot(selection_list)
                # plt.show()
                spatial_dic_amplifying[pix] = selection_list_amplifying

                selection_list_weak_stabilizing = []
                for yr in weak_stabilizing_list:
                    if yr<2000:
                        continue
                    selection = time_series_reshape_dict[yr]
                    # selection_dic[key]=selection
                    selection_list_weak_stabilizing.append([yr, selection])
                # plt.plot(selection_list)
                # plt.show()
                spatial_dic_weak_stabilizing[pix] = selection_list_weak_stabilizing

                selection_list_strong_stabilizing = []
                for yr in strong_stabilizing_list:
                    if yr<2000:
                        continue
                    selection = time_series_reshape_dict[yr]
                    # selection_dic[key]=selection
                    selection_list_strong_stabilizing.append([yr, selection])
                # plt.plot(selection_list)
                # plt.show()
                spatial_dic_strong_stabilizing[pix] = selection_list_strong_stabilizing

            result_all_dic[f'amplifying_{f.split(".")[0]}']=spatial_dic_amplifying
            result_all_dic[f'weak_stabilizing_{f.split(".")[0]}']=spatial_dic_weak_stabilizing
            result_all_dic[f'strong_stabilizing_{f.split(".")[0]}']=spatial_dic_strong_stabilizing

        df=T.spatial_dics_to_df(result_all_dic)
        T.save_df(df,outf)
        T.df_to_excel(df,outf+'.xlsx')


def main():
    # nctotif().run()
    # Resample().run()
    # TIFtoDIC().run()
    # Check_plot().run()
    # Phenology().run()
    # process_LAI().run()
    statistic_analysis().run()

    pass

if __name__ == '__main__':
    main()