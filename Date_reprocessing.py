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
import semopy

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
        # self.nc_to_tif_Trendy()
        self.nc_to_tif_CO2()


        # self.per_pixel_annual()

    def nctotif_CRU(self):  # 降雨需要乘以30天，温度不需要。

        fpath = data_root+'Climate\monthly\\SMroot_1980-2022_GLEAM_v3.7a_MO.nc'
        outdir = data_root + 'Climate\monthly\SMroot\\'

        T.mk_dir(outdir, force=True)
        nc = Dataset(fpath)
        print(nc)
        print(nc.variables.keys())
        variable = 'SMroot'
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
        fdir = data_root + 'TRENDY_SM\\TRENDY_nc\\'

        for f in os.listdir(fdir):
            if not 'DLEM' in f:
                continue

            if f.startswith('.'):
                continue

            outdir_name = f.split('.')[0]
            print(outdir_name)


            outdir = data_root+rf'/Trendy_TIFF/{outdir_name}//'
            Tools().mk_dir(outdir, force=True)
            yearlist = list(range(2003, 2023))

            # # check nc variables
            # print(nc.variables.keys())
            # exit()

            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir + f, var_name='mrso', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue

    def nc_to_tif_CO2(self):
        fdir = rf'E:\Project3\Data\CO2\\'

        for f in os.listdir(fdir):
            if not 'CO2_SSP245_2015_2150.nc' in f:
                continue

            if f.startswith('.'):
                continue



            outdir = rf'E:\Project3\Data\CO2_TIFF//SSP245//'
            Tools().mk_dir(outdir, force=True)
            yearlist = list(range(1982, 2021))

            # # check nc variables
            # print(nc.variables.keys())
            # exit()

            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir + f, var_name='CO2', outdir=outdir, yearlist=yearlist)
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
                    try:
                        lat = ncin.variables['Latitude'][:]
                        lon = ncin.variables['Latitude'][:]
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
        year_list = [i for i in range(2003, 2023)]
        for year in year_list:
            fpath = data_root + rf'\Climate\NC_monthly\\SMsurf_2003-2022_GLEAM_v3.7b_MO.nc'
            outdir = data_root + rf'Climate\TIFF_monthly\SMsurf\\'

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
            SPEI_arr_list = nc['SMsurf']
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
        fdir_all = data_root + 'Trendy_SM\Trendy_TIFF\\'

        year = list(range(2000, 2023))
        # print(year)
        # exit()
        for fdir in tqdm(os.listdir(fdir_all)):
            outdir = data_root + rf'Trendy_SM\resample_SM\\{fdir}\\'

            T.mk_dir(outdir, force=True)

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
        fdir_all=data_root + rf'Trendy_SM\resample_SM\\'


        for fdir in tqdm(os.listdir(fdir_all)):
            outdir = data_root + rf'Trendy_SM\Trendy_SM_unify\\{fdir}\\'
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
        # self.trendy_ensemble_calculation()
        self.tif2dict()
        # self.tif2dict_trendy()
    def trendy_ensemble_calculation(self):
        fdir_all = data_root + rf'Trendy_SM\Trendy_SM_unify\\'
        outdir = data_root +'LAI\Trendy_unify\Trendy_enemble2\\'
        T.mk_dir(outdir, force=True)
        year_list = list(range(2000, 2023))
        month_list= list(range(1, 13))

        for year in tqdm(year_list):
            for month in tqdm(month_list):
                data_list = []
                for fdir in tqdm(os.listdir(fdir_all)):
                    if 'MCD' in fdir:
                        continue
                    if 'MOD' in fdir:
                        continue
                    for f in tqdm(os.listdir(fdir_all + fdir + '\\')):
                        if not f.endswith('.tif'):
                            continue
                        if f.startswith('._'):
                            continue
                        # print(f)
                        # exit()
                        data_year = f.split('.')[0][0:4]
                        data_month = f.split('.')[0][4:6]


                        if not int(data_year) == year:
                            continue
                        if not int(data_month) == month:
                            continue
                        arr=to_raster.raster2array(fdir_all + fdir + '\\' + f)[0]
                        arr_unify=arr[:360][:360,:720]
                        arr_unify = np.array(arr_unify, dtype=np.float)
                        arr_unify[arr_unify < 0] = np.nan
                        arr_unify[arr_unify > 10] = np.nan
                        data_list.append(arr_unify)

                ##define arr_average and calculate arr_average

                arr_average=np.nanmean(data_list,axis=0)
                arr_average=np.array(arr_average,dtype=np.float)
                arr_average[arr_average<0]=np.nan
                arr_average[arr_average>10]=np.nan
                # save

                DIC_and_TIF().arr_to_tif(arr_average,outdir+'{}{:02d}{:02d}.tif'.format(year,month,11))


                #############3

                # average_matrix=[]

                # for i in range(len(data_list[0])):
                #     temp=[]
                #     for j in range(len(data_list[0][0])):
                #         values_list=[]
                #         for arr in data_list:
                #             val=arr[i][j]
                #             values_list.append(val)
                #         values_list=np.array(values_list)
                #         avarage=np.nanmean(values_list)
                #         temp.append(avarage)
                #     average_matrix.append(temp)
                # average_matrix=np.array(average_matrix)
                # DIC_and_TIF().arr_to_tif(average_matrix,outdir+'{}{}.tif'.format(year,month))










    def tif2dict(self):
        fdir=data_root + rf'\resample_monthly\\Temp\\'
        outdir=data_root+'Climate\DIC\\Temp\\'


        NDVI_mask_f='D:\Greening\Data\Base_data/NDVI_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
        array_mask[array_mask<0]=np.nan

        T.mk_dir(outdir,force=True)
        flist=os.listdir(fdir)
        all_array=[]
        year_list=list(range(2003,2022))  # 作为筛选条件
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
            array = np.array(array)
            array=array[:360]  # PAR是361*720

            array[array<-999]=np.nan
            # array[array ==0] = np.nan
            # array[array < 0] = np.nan # 当变量是LAI 的时候，<0!!
            # plt.imshow(array)
            # plt.show()
            array_mask=np.array(array_mask)
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
        fdir_all=data_root + rf'Trendy_SM\Trendy_SM_unify\\'

        NDVI_mask_f=data_root+rf'/Base_data/NDVI_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
        array_mask[array_mask<0]=np.nan


        year_list=list(range(2003,2022))  # 作为筛选条件
        for fdir in tqdm(os.listdir(fdir_all),desc='loading...'):


            all_array = []

            outdir = data_root + 'Trendy_SM\\DIC\\{}\\'.format(fdir)

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
                array=np.array(array,dtype=float)
                #extract 360 and 720
                array_unify=array[:360][:360,:720] # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify<-999]=np.nan
                # array[array ==0] = np.nan
                array_unify[array_unify < 0] = np.nan # 当变量是LAI 的时候，<0!!
                # plt.imshow(array)
                # plt.show()
                array_mask=np.array(array_mask,dtype=float)
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
        self.hants_trendy()
        # self.check_hants()
        # self.per_pixel_annual()
        # self.annual_phenology()
        # self.compose_annual_phenology()
        # self.data_clean()
        # self.average_phenology()
        # self.pick_daily_phenology()
        # self.pick_monthly_phenology()
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


        fdir_all=self.datadir_all+'LAI/DIC/'

        tif_dir_all = self.datadir_all + rf'\LAI\Trendy_Yang\\'


        for fdir in os.listdir(fdir_all):
            if not 'Trendy' in fdir:
                continue

            date_list = []

            print(fdir)
            outdir = self.datadir_all + rf'LAI/Hants_annually_smooth/{fdir}/'
            if isdir(outdir):
                continue
            T.mkdir(outdir, force=True)

            tif_dir=tif_dir_all+fdir+'/'
            for f_tiff in os.listdir(tif_dir):
                if f_tiff.endswith('.tif'):
                    date=f_tiff.split('.')[0]

                    y=int(date[:4])
                    m=int(date[4:6])
                    d=int(date[6:8])
                    # format = '%Y%m%d'
                    if y<2003:
                        continue
                    date_obj = datetime.datetime.strptime(date, '%Y%m%d')
                    # date_obj = datetime.datetime.strptime(date, '%Y%m')
                    # date_obj=datetime.datetime(y,m,d)
                    date_list.append(date_obj)
            # print(len(date_list))

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
            if isdir(fdir_all+fdir):
                continue

            outdir = data_root + '\LAI\\\per_pix_annual\\' + fdir + '\\'
            Tools().mk_dir(outdir, force=True)

            for f in tqdm (os.listdir(fdir_all+fdir)):
                if f.endswith('.npy'):
                    hants365_dic=np.load(fdir_all+fdir+'/'+f,allow_pickle=True).item()


            for y in range(2003, 2023):
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

        hants365=np.load(rf'D:\Greening\Data\FLUXNET_2015\screening_sites_hants\\hants365_DE-Obe.npy',allow_pickle=True).item()
        for pix in hants365:
            result=hants365[pix]
            print(len(result))
            # exit()

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

    def pick_monthly_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]
        fdir_all = data_root + f'LAI/phenology/average_phenology/'
        for fdir in os.listdir(fdir_all):
            if not fdir.endswith('MCD'):
                continue
            outdir = data_root + f'LAI/phenology/pick_monthly_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'pick_monthly_phenology.df')

            phenology_df = T.load_df(
                f'{fdir_all}/{fdir}/phenology_dataframe.df')

            early_dic = {}
            peak_dic = {}
            late_dic = {}
            all_result_dic = {}

            for i, row in tqdm(phenology_df.iterrows(), total=len(phenology_df)):
                pix = row['pix']
                all_result_dic[pix] = {}

                early_start = row['early_start_mon']
                early_end = row['early_end_mon']

                late_start = row['late_start_mon']
                late_end = row['late_end_mon']
                early_period= np.arange(int(early_start), int(early_end))
                early_peak_period = list(range(int(early_start), int(late_start)))
                peak_period=list(range(int(early_end), int(late_start)))
                late_period = list(range(int(late_start), int(late_end)+1))
                print(early_peak_period)
                print(peak_period)
                print(early_period)
                print(late_period)
                print('-------------------')
                # exit()
                all_result_dic[pix]['early_peak'] = early_peak_period
                all_result_dic[pix]['early'] = early_period
                all_result_dic[pix]['peak'] = peak_period
                all_result_dic[pix]['late'] = late_period

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
        f = rf'D:\Greening\Result\zscore\LAI\late\MCD.npy'
        # f = rf'D:\Greening\Data\Trendy\DIC\\CABLE-POP_S2_lai\per_pix_dic_014.npy'
        # f='/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_window/1982-2015_during_early/during_early_CO2.npy'

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
        # self.extraction_variables_static_during_daily()
        # self.extraction_variables_static_during_monthly()
        self.extraction_variables_static_during_monthly_TRENDY_SM()
        # self.average_all_trendy()


    def data_transform(self):  # 将hants365 数据转换成一个pix 23 年数据

        fdir_all = data_root + f'LAI\\Hants_annually_smooth\\'
        for fdir in os.listdir(fdir_all):

            outdir = data_root + f'original_dataset/{fdir}/'
            if isdir(outdir):
                continue

            T.mkdir(outdir, force=True)



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
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'DLEM_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai','Trendy_ensemble']



        for variable in variables_list:
            phenology_df = T.load_df(data_root + '/phenology/pick_daily_phenology/MCD/pick_daily_phenology.df')

            fdir=data_root+rf'original_dataset\\{variable}\\'

            dic_variables = {}

            # 加载变量数据
            for f in os.listdir(fdir):
                if not f.endswith('.npy'):
                    continue
                dic_i=dict(np.load(fdir+f, allow_pickle=True, encoding='latin1' ).item())
                dic_variables.update(dic_i)

            period_list=['early','peak','late','early_peak','early_peak_late']
            # period_list = [ 'late', 'early_peak', ]


            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root +rf'extraction_original_val/Trendy/{variable}/'

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):
                    r, c = pix
                    if r > 120:
                        continue

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
                    # if len(time_series_flatten)!=20*365:
                    #     continue
                    # print(len(time_series_flatten))


                    picked_daily = np.array(picked_daily, dtype=int)
                    print(len(time_series))


                    for year in range(len(time_series)):

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

    def extraction_variables_static_during_monthly(self):  # 静态提取during multiyear


        climate_variales = ['precip', 'Temp', 'SMroot', 'SMsurf', 'Et']

        for variable in climate_variales:
            phenology_df = T.load_df(data_root + 'LAI/phenology/pick_monthly_phenology/MCD/pick_monthly_phenology.df')

            fdir = data_root + rf'Climate\DIC_monthly\\{variable}\\'

            dic_variables = {}

            # 加载变量数据
            for f in os.listdir(fdir):
                if not f.endswith('.npy'):
                    continue
                dic_i = dict(np.load(fdir + f, allow_pickle=True, encoding='latin1').item())
                dic_variables.update(dic_i)

            period_list=['early','peak','late','early_peak',]


            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root + rf'extraction_original_val/Climate_monthly/'

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):
                    r, c = pix
                    if r > 120:
                        continue

                    if pix not in dic_period:
                        continue
                    picked_daily = dic_period[pix]  # 修改这里
                    print(picked_daily)
                    if np.nanmean(picked_daily) == 0:
                        continue
                    ## if nan continue

                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]
                    time_series_reshape = time_series.reshape((len(time_series) // 12, 12))
                    if len(time_series_reshape) != 20:
                        continue


                    picked_daily = np.array(picked_daily, dtype=int)
                    # print(len(time_series))
                    # print(len(time_series_reshape))


                    for year in range(len(time_series_reshape)):

                        during_time_series = time_series_reshape[year][picked_daily-1]
                        # print(picked_month)#!!!!!

                        during_time_series = np.array(during_time_series, dtype=float)

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
                np.save(outdir + 'during_{}_{}'.format(period, variable), dic_during_variables)  # 修改

    def extraction_variables_static_during_monthly_TRENDY_SM(self):  # 提取TRENDY SM


        fdir_all=data_root+rf'Trendy_SM\DIC\\'

        for fdir in os.listdir(fdir_all):
            phenology_df = T.load_df(data_root + 'pick_monthly_phenology\MCD/pick_monthly_phenology.df')



            dic_variables = {}

            # 加载变量数据
            dic_variables=T.load_npy_dir(fdir_all+fdir)
            period_list=['early','peak','late','early_peak',]


            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root + rf'extraction_Trendy_SM/{fdir}/'

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):
                    r, c = pix
                    if r > 120:
                        continue

                    if pix not in dic_period:
                        continue
                    picked_daily = dic_period[pix]  # 修改这里
                    print(picked_daily)
                    if np.nanmean(picked_daily) == 0:
                        continue
                    ## if nan continue

                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]

                    time_series_reshape = time_series.reshape((len(time_series) // 12, 12))
                    # if len(time_series_reshape) != 20:
                    #     continue


                    picked_daily = np.array(picked_daily, dtype=int)
                    # print(len(time_series))
                    # print(len(time_series_reshape))


                    for year in range(len(time_series_reshape)):

                        during_time_series = time_series_reshape[year][picked_daily-1]
                        # print(picked_month)#!!!!!

                        during_time_series = np.array(during_time_series, dtype=float)

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
                np.save(outdir + 'during_{}_{}'.format(period, fdir), dic_during_variables)  # 修改

class statistic_analysis():
    def __init__(self):
        pass
    def run(self):
        # self.trend_analysis()
        # self.trend_analysis_plot()
        # self.detrend_zscore()
        self.trend_analysis_stat()

        # self.detrend_zscore_seasonality()
        # self.detrend_zscore_monthly()
        # self.zscore()


    def trend_analysis(self):

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)



        period_list = ['early', 'peak', 'late', 'early_peak', 'early_peak_late']



        for period in period_list:

            fdir = result_root + rf'zscore\LAI\{period}\\'
            outdir = result_root + rf'trend_analysis\\zscore\\{period}\\'
            Tools().mk_dir(outdir, force=True)
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
    def trend_analysis_stat(self):

        fdir=result_root+rf'trend_analysis\\zscore\\late\\'

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'MCD' in f:
                continue
            if 'p_value' in f:
                continue
            f_trend = f.split('.')[0] + '.npy'
            print(f_trend)
            f_pvalue = f.split('.')[0].replace('trend', 'p_value') + '.npy'
            # print(f_pvalue)
            # exit()

            dic_trend=DIC_and_TIF().spatial_arr_to_dic(np.load(fdir+f_trend))
            dic_pvalue=DIC_and_TIF().spatial_arr_to_dic(np.load(fdir+f_pvalue))

            count=0
            pix_number = 0
            for pix in dic_trend:
                if pix not in dic_pvalue:
                    continue
                if dic_pvalue[pix]<-99:
                    continue
                if dic_trend[pix]<-99:
                    continue


                if np.isnan(dic_trend[pix]):
                    continue
                if np.isnan(dic_pvalue[pix]):
                    continue
                pix_number+=1
                trend=dic_trend[pix]

                pvalue=dic_pvalue[pix]
                if pvalue<0.05 and trend>0:
                    count+=1
                # if trend>0:
                #     count+=1


                ## calculate percentage
            percentage = count / pix_number
            print(percentage)


        pass


    def get_trend_analysis_flist(self):

        period_list = ['late', 'early_peak', 'early_peak_late']
        flist_all_product = ['MCD_trend.tif','CABLE-POP_S2_lai_trend.tif', 'CLASSIC_S2_lai_trend.tif',
                 'CLM5_trend.tif', 'IBIS_S2_lai_trend.tif',
                 'ISAM_S2_LAI_trend.tif', 'ISBA-CTRIP_S2_lai_trend.tif',
                 'JSBACH_S2_lai_trend.tif', 'LPX-Bern_S2_lai_trend.tif',
                  'VISIT_S2_lai_trend.tif']
        for period in period_list:
            count = 1
            plt.figure(figsize=(11, 7))
            fdir = result_root + rf'trend_analysis\\zscore\\{period}\\'
            flist = []
            for f in tqdm(T.listdir(fdir)):
                if not f.endswith('.tif'):
                    continue
                if '_p_value' in f:
                    continue

                if 'MOD' in f:
                    continue
                if 'ensemble' in f:
                    continue
                if 'ORCHIDEE' in f:
                    continue
                if 'JULES' in f:
                    continue
                if 'LPJ' in f:
                    continue
                if 'OCN' in f:
                    continue
                if 'SDGVM' in f:
                    continue
                if 'YIBs' in f:
                    continue
                flist.append(f)
            print(flist);exit()

    def detrend_zscore(self): #

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'//Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        product_list=  ['MCD','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'DLEM_S2_lai' 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',''
                           'LPX-Bern_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai','Trendy_ensemble']

        product_list=  ['CABLE-POP_S2_mrso', 'CLASSIC_S2_mrso',  'CLM5', 'IBIS_S2_mrso',
                          'ISAM_S2_mrso', 'ISBA-CTRIP_S2_mrso', 'JSBACH_S2_mrso',''
                           'LPX-Bern_S2_mrso', 'VISIT_S2_mrso', ]


        for period in ['early', 'peak', 'late', 'early_peak', ]:

            for product in product_list:
                outdir = result_root + rf'detrend_zscore\Trendy_SM\\{period}\\'
                outf=outdir+product+'.npy'
                # print(outf)
                # exit()
                f = result_root + rf'extraction_Trendy_SM\{product}\\during_{period}_{product}.npy'

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
                    print(len(dic[pix]))
                    time_series = dic[pix]
                    print(len(time_series))

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
                    if std == 0:
                        continue
                    delta_time_series = (time_series - mean) / std

                    detrend_delta_time_series = signal.detrend(delta_time_series)

                    # plt.plot(detrend_delta_time_series)
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series

                np.save(outf, detrend_zscore_dic)

    def detrend_zscore_monthly(self): #
        ## here we use monthly data to detrend the zscore of climate

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'/Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)


        product_list = ['Precipitation','Temp','SMroot','SMsurf','Et']

        for product in product_list:
            outdir = result_root + rf'detrend_zscore\\climate_monthly\annual\\'
            outf=outdir+product+'.npy'
            fdir= data_root + rf'\Climate\DIC\\{product}\\'

            Tools().mk_dir(outdir,force=True)

            dic = T.load_npy_dir(fdir)

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
                print(len(dic[pix]))
                time_series_reshape = dic[pix].reshape(-1, 12)

                ## detrend the zscore
                detrend_time_series_list = []
                time_series_T=time_series_reshape.T

                for time_series in time_series_T:

                    time_series_mean = np.nanmean(time_series)
                    time_series_std = np.nanstd(time_series)
                    # plt.plot(time_series)

                    time_series_zscore = (time_series - time_series_mean) / time_series_std
                    # plt.plot(time_series_zscore,c='b')
                    if np.isnan(np.nanmean(time_series_zscore)):
                        continue
                    detrend_time_series = signal.detrend(time_series_zscore)

                    # plt.plot(detrend_time_series,c='r')
                    # plt.show()
                    detrend_time_series_list.append(detrend_time_series)
                detrend_time_series_list = np.array(detrend_time_series_list).T
                detrend_time_series_list = detrend_time_series_list.flatten()
                # plt.plot(detrend_time_series_list)
                # plt.show()


                detrend_zscore_dic[pix] = detrend_time_series_list


            np.save(outf, detrend_zscore_dic)

    def detrend_zscore_seasonality(self): #

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'/Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)


        product_list = ['precip','Temp','SMroot','SMsurf','Et']


        for period in ['early', 'peak', 'late', 'early_peak',]:

            for product in product_list:
                outdir = result_root + rf'detrend_zscore\\climate_monthly\{period}\\'
                outf=outdir+product+'.npy'
                # print(outf)
                # exit()
                f = result_root + rf'\extraction_original_val\Climate_monthly\\during_{period}_{product}.npy'

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
                    print(len(dic[pix]))
                    time_series = dic[pix][:19]
                    print(len(time_series))

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
                    if std == 0:
                        continue
                    delta_time_series = (time_series - mean) / std

                    detrend_delta_time_series = signal.detrend(delta_time_series)

                    # plt.plot(detrend_delta_time_series)
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series

                np.save(outf, detrend_zscore_dic)

    def zscore(self):

        dic_mask_lc_file = data_root+'/Base_data/LC_reclass2.npy'

        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        product_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'DLEM_S2_lai','IBIS_S2_lai',
                        'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                         'LPX-Bern_S2_lai',  'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'Trendy_ensemble',]
        product_list=['MCD']


        for period in ['early', 'peak', 'late', 'early_peak', 'early_peak_late']:
            outdir = result_root + rf'zscore2\\LAI\\{period}\\'
            for product in product_list:
                outf = outdir + product + '.npy'
                # print(outf)
                # exit()
                f = result_root + rf'extraction_original_val\LAI\{product}\\during_{period}_{product}.npy'

                Tools().mk_dir(outdir, force=True)
                dic = {}

                dic = dict(np.load(f, allow_pickle=True, ).item())

                zscore_dic = {}

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
                    # print(len(dic[pix]))
                    time_series = dic[pix][:19]
                    # print(time_series)

                    time_series = np.array(time_series)
                    # plt.plot(time_series)
                    # plt.show()

                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue

                    delta_time_series = []
                    mean = np.nanmean(time_series)
                    std = np.nanstd(time_series)
                    if std == 0:
                        continue
                    delta_time_series = (time_series - mean) / std

                    # plt.plot(delta_time_series)
                    # plt.title(len(delta_time_series))
                    # plt.show()

                    zscore_dic[pix] = delta_time_series

                T.save_npy(zscore_dic, outf)

        pass


class frequency_analysis():
    def __init__(self):

        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + 'Data_frame\\Frequency\\'

        self.dff = self.this_class_arr + 'frequency_dataframe.df'

        Tools().mk_dir(self.this_class_arr, force=True)


        pass

    def run(self):

        ### 1.create frequency dataframe

        # df=self.pick_greening_year_frequency_heatmap()

        ## 2. add landcover and trend, row, and some attributes

        # call the function in the class of 'build_dataframe'

        # 3. plot frequency heatmap

        # df, dff = self.__load_df()
        # df_clean = self.clean_df(df)
        # self.frenquency_heatmap(df_clean)

        ### 3. plot heatmap differences between trendy and obs


        self.frenquency_heatmap_differences()


        ## 4. plot frequency bar
        # self.frenquency_heatmap_bar(df_clean)


        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df, dff
        # return df_early,dff

    def clean_df(self, df):
        df = df[df['row'] < 120]
        # df = df[df['HI_class'] == 'Humid']
        # df = df[df['HI_class'] == 'Dryland']
        df = df[df['max_trend'] < 10]

        df = df[df['landcover_GLC'] != 'Crop']

        return df


    def pick_greening_year_frequency_heatmap(self): # 通过pick years and calculate frequency

        f_early_peak_LAI = result_root + rf'\\detrend_zscore\\early_peak\\Trendy_ensemble.npy'
        f_late_LAI = result_root + rf'\detrend_zscore\\late\\Trendy_ensemble.npy'
        outdir = result_root + rf'Data_frame/Frequency/\\Trendy_ensemble///'
        outf = outdir + f'frequency_dataframe.df'
        T.mk_dir(outdir, force=1)
        dic_early_peak_LAI = dict(np.load(f_early_peak_LAI, allow_pickle=True, ).item())
        dic_late_LAI = dict(np.load(f_late_LAI, allow_pickle=True, ).item())
        # threshold_early_list=np.linspace(-2, 2, 21)
        #
        # threshold_late_list=np.linspace(-2, 2, 21)

        threshold_early_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]

        threshold_late_list = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

        all_result_dic={}


        for i in tqdm(range(len(threshold_early_list))):
            if i >= len(threshold_early_list) - 1:
                break

            for j in range(len(threshold_late_list)):
                if j >= len(threshold_late_list) - 1:
                    break

                early_threshold=threshold_early_list[i]
                late_threshold=threshold_late_list[j]

                spatial_dic={}
                for pix in dic_early_peak_LAI:

                    early_peak_LAI = dic_early_peak_LAI[pix]

                    # print(len(early_peak_LAI))
                    if not pix in dic_late_LAI:
                        continue
                    late_LAI=dic_late_LAI[pix]

                    early_condition1=early_peak_LAI > threshold_early_list[i]
                    early_condition2=early_peak_LAI < threshold_early_list[i + 1]
                    early_condition_intersect_index=np.logical_and(early_condition1, early_condition2)

                    index_early_peak_LAI = np.where(early_condition_intersect_index)
                    index_early_peak_LAI=np.array(index_early_peak_LAI)
                    index_early_peak_LAI=index_early_peak_LAI.flatten()

                    if threshold_late_list[j]>=0:
                        factor=1
                    else:
                        factor=-1
                    late_condition1=late_LAI > threshold_late_list[j]
                    late_condition2= late_LAI < threshold_late_list[j+1]


                    late_LAI_condition_intersect_index = np.logical_and(late_condition1, late_condition2)

                    index_late_LAI = np.where(late_LAI_condition_intersect_index)

                    intersect_index = np.intersect1d(index_early_peak_LAI, index_late_LAI)
                    if len(index_early_peak_LAI)==0:
                        continue

                    frequency=len(intersect_index)/len(index_early_peak_LAI)*100*factor



                    spatial_dic[pix] = frequency
                    column_name=f'{early_threshold:0.5f}-{late_threshold:0.5f}'
                    all_result_dic[column_name]=spatial_dic

        df=T.spatial_dics_to_df(all_result_dic)

        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        return df

    def build_frequency_heatmap_dataframe(self,df, P_PET_fdir):


        build_dataframe().add_row(df)
        build_dataframe().add_NDVI_mask(df)
        build_dataframe().add_GLC_landcover_data_to_df(df)
        build_dataframe().add_max_trend_to_df(df)
        P_PET_dic =build_dataframe().P_PET_ratio(P_PET_fdir)
        P_PET_reclass_dic = build_dataframe(P_PET_fdir)
        df = T.add_spatial_dic_to_df(df, P_PET_reclass_dic, 'HI_class')


        pass

    def frenquency_heatmap(self,df):
        T.print_head_n(df, 10)

        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        regions = ['Humid', 'Dryland']
        cm=1/2.54

        for region in regions:
            plt.figure(figsize=(15*cm, 7*cm))

            df_temp=df[df['HI_class']==region]

            threshold_early_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]

            threshold_late_list = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

            threshold_late_list_reverse = threshold_late_list
            threshold_early_list_reverse = threshold_early_list[::-1]

            threshold_early_list_str = [f'{i:.5f}' for i in threshold_early_list]
            threshold_late_list_str = [f'{i:.5f}' for i in threshold_late_list]


            x_list = []
            y_list = []
            z_list = []

            for x in threshold_early_list_str:
                for y in threshold_late_list_str:
                    key=f'{x}-{y}'
                    if key not in df_temp.keys():
                        continue
                    vals = df_temp[key].to_list()
                    # plt.hist(vals,bins=20)
                    # plt.show()

                    vals = np.array(vals)
                    # vals[vals==0]=np.nan
                    vals_mean = np.nanmean(vals)
                    z_list.append(vals_mean)
                    x_list.append(x)
                    y_list.append(y)

            z_list=np.array(z_list)
            z_list=z_list.reshape(len(threshold_early_list_str)-1,len(threshold_late_list_str)-1)
            # z_list_T = z_list.T
            z_list_T = z_list
            z_list_T = z_list_T[::-1]
            label_matrix=abs(z_list_T)
            label_matrix=np.round(label_matrix,2)
            print(label_matrix)
            # np.save(result_root + rf'Data_frame\\Frequency\\LAI3g_MCD_2003_2018\\MCD_{region}.npy', label_matrix)
            # exit()


            ax=sns.heatmap(z_list_T, annot=label_matrix, linewidths=0.75,yticklabels=threshold_early_list_str,
                           xticklabels=threshold_late_list_str,cmap='RdBu',vmin=-15,vmax=15,
                           annot_kws={'fontsize': 8},
                           cbar_kws={'label': 'Frenquency (%)','ticks':[-15, -10,-5, 0, 5, 10,15]},fmt='.1f')
            threshold_early_list_str_format = [f'{i:.2f}' for i in threshold_early_list_reverse]
            threshold_late_list_str_format = [f'{i:.2f}' for i in threshold_late_list_reverse]

            ax.set_xticklabels(threshold_late_list_str_format, rotation=45, horizontalalignment='right')

            ax.set_yticklabels(threshold_early_list_str_format, rotation=0, horizontalalignment='right')
            cbar=ax.collections[0].colorbar
            cbar.ax.set_yticklabels([15, 10, 5, 0, 5, 10,15])
            # plt.tight_layout()


            plt.title(f'MCD_{region}')

        plt.show()
        plt.close()
        #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
        #     plt.close()

    def frenquency_heatmap_differences(self): ## to address reviewer's question
        df_obs = T.load_df(self.this_class_arr + '\MCD\\frequency_dataframe.df')
        df_trendy = T.load_df(self.this_class_arr + '\Trendy_ensemble\\frequency_dataframe.df')

        df_obs_clean=self.clean_df(df_obs)
        df_trendy_clean = self.clean_df(df_trendy)

        regions = ['Humid', 'Dryland']
        cm=1/2.54

        for region in regions:
            plt.figure(figsize=(15*cm, 7*cm))

            df_obs_temp=df_obs_clean[df_obs_clean['HI_class']==region]
            df_trendy_temp=df_trendy_clean[df_trendy_clean['HI_class']==region]

            threshold_early_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]

            threshold_late_list = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

            threshold_late_list_reverse = threshold_late_list
            threshold_early_list_reverse = threshold_early_list[::-1]

            threshold_early_list_str = [f'{i:.5f}' for i in threshold_early_list]
            threshold_late_list_str = [f'{i:.5f}' for i in threshold_late_list]


            x_list = []
            y_list = []
            z_list = []

            for x in threshold_early_list_str:

                for y in threshold_late_list_str:
                    key=f'{x}-{y}'
                    if key not in df_obs_temp.keys():
                        continue

                    if key not in df_trendy_temp.keys():
                        continue
                    ##
                    vals_obs = df_obs_temp[key].to_list()
                    vals_trendy = df_trendy_temp[key].to_list()
                    # plt.hist(vals,bins=20)
                    # plt.show()

                    vals_obs=np.array(vals_obs)
                    vals_trendy=np.array(vals_trendy)
                    # vals[vals==0]=np.nan
                    vals_obs_mean =abs(np.nanmean(vals_obs))
                    vals_trendy_mean = abs(np.nanmean(vals_trendy))
                    vals_diff = vals_trendy_mean-vals_obs_mean
                    z_list.append(vals_diff)
                    x_list.append(x)
                    y_list.append(y)

            z_list=np.array(z_list)
            z_list=z_list.reshape(len(threshold_early_list_str)-1,len(threshold_late_list_str)-1)
            # z_list_T = z_list.T
            z_list_T = z_list
            z_list_T = z_list_T[::-1]
            # plt.imshow(z_list_T,vmin=-10,vmax=10,cmap='RdBu')
            # plt.show()
            label_matrix=z_list_T
            label_matrix=np.round(label_matrix,2)
            print(label_matrix)
            # np.save(result_root + rf'Data_frame\\Frequency\\LAI3g_MCD_2003_2018\\MCD_{region}.npy', label_matrix)
            # exit()


            ax=sns.heatmap(z_list_T, annot=label_matrix, linewidths=0.75,yticklabels=threshold_early_list_str,
                           xticklabels=threshold_late_list_str,cmap='RdBu',vmin=-15,vmax=15,
                           annot_kws={'fontsize': 8},
                           cbar_kws={'label': 'Frenquency (%)','ticks':[-15, -10,-5, 0, 5, 10,15]},fmt='.1f')
            threshold_early_list_str_format = [f'{i:.2f}' for i in threshold_early_list_reverse]
            threshold_late_list_str_format = [f'{i:.2f}' for i in threshold_late_list_reverse]

            ax.set_xticklabels(threshold_late_list_str_format, rotation=45, horizontalalignment='right')

            ax.set_yticklabels(threshold_early_list_str_format, rotation=0, horizontalalignment='right')
            cbar=ax.collections[0].colorbar
            cbar.ax.set_yticklabels([-15, -10, -5, 0, 5, 10,15])
            # plt.tight_layout()


            plt.title(f'Diff_{region}')

        plt.show()
        plt.close()
        #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
        #     plt.close()

    def creat_data_source(self):
        pass

    def spatial_frequency(self):
        import xycmap
        Ter = xymap.Ternary_plot(
            top_color=(10, 94, 0),
            left_color=(0, 30, 210),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )
        rgb_arr = Ter.rgb_arr

        # plt.imshow(rgb_arr)
        # plt.show()
        outdir = join(self.this_class_tif,'MCD')
        fdir_early = join(data_root,'detrended_zscore_LAI/early_peak')
        fdir_late = join(data_root,'detrended_zscore_LAI/late')
        for f in T.listdir(fdir_early):
            if not 'MCD' in f:
                continue
            outdir_i = join(outdir,f.replace('.npy',''))
            T.mk_dir(outdir_i,force=True)
            fpath_early = join(fdir_early,f)
            fpath_late = join(fdir_late,f)
            early_dict = T.load_npy(fpath_early)
            late_dict = T.load_npy(fpath_late)
            result_dict = {}
            for pix in tqdm(early_dict,desc=f):
                early_vals = early_dict[pix]
                if not pix in late_dict:
                    continue
                late_vals = late_dict[pix]
                father = 0
                son_amplification = 0
                son_weak_stabilization = 0
                son_strong_stabilization = 0
                for i in range(len(early_vals)):
                    early_val = early_vals[i]
                    late_val = late_vals[i]
                    if early_val < 0:
                        continue
                    father += 1
                    if late_val > early_val:
                        son_amplification += 1
                    elif late_val < early_val and late_val > 0:
                        son_weak_stabilization += 1
                    else:
                        son_strong_stabilization += 1
                if father == 0:
                    continue
                ratio_amplification = son_amplification/father
                ratio_weak_stabilization = son_weak_stabilization/father
                ratio_strong_stabilization = son_strong_stabilization/father
                result_dict[pix] = {
                    'ratio_amplification':ratio_amplification,
                    'ratio_weak_stabilization':ratio_weak_stabilization,
                    'ratio_strong_stabilization':ratio_strong_stabilization,
                }
            if len(result_dict) == 0:
                continue
            df = T.dic_to_df(result_dict,'pix')
            cols = ['ratio_amplification','ratio_weak_stabilization','ratio_strong_stabilization']
            # cols = []
            arr_list = []
            for col in cols:
                spatial_dict = T.df_to_spatial_dic(df,col)
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
                arr = np.array(arr)
                ##### interpolation
                # window = 3
                # new_arr = []
                # for i in tqdm(range(len(arr))):
                #     new_arr_i = []
                #     for j in range(len(arr[0])):
                #         val = arr[i, j]
                #         if np.isnan(val):
                #             val = np.nanmean(arr[i - window:i + window + 1, j - window:j + window + 1])
                #         new_arr_i.append(val)
                #     new_arr.append(new_arr_i)
                # new_arr = np.array(new_arr)
                # arr_list.append(new_arr)

                arr_list.append(arr)

            color_matrix = []

            s1_matrix = []
            s2_matrix = []
            s3_matrix = []

            for i in range(len(arr_list[0])):
                color_matrix_i = []
                s1_matrix_i = []
                s2_matrix_i = []
                s3_matrix_i = []
                for j in range(len(arr_list[0][0])):
                    s1 = arr_list[0][i,j]
                    s2 = arr_list[1][i,j]
                    s3 = arr_list[2][i,j]

                    s1_matrix_i.append(s1)
                    s2_matrix_i.append(s2)
                    s3_matrix_i.append(s3)


                    if np.isnan(s1) or np.isnan(s2) or np.isnan(s3):
                        color = (0, 0, 0, 0)
                        color_matrix_i.append(color)
                        continue

                    # print(s1,s2,s3)

                    color = Ter.get_color(s1,s2,s3)
                        # r, g, b, w = 0, 0, 0, 0
                        # color_matrix_i.append(np.array([r, g, b, w]))
                        # continue
                    # color = Ter.get_color(s1,s2,s3)
                    r,g,b = color
                    r,g,b = int(r*255),int(g*255),int(b*255)
                    w = 255
                    if r==255 and g==255 and b==255:
                        print(s1,s2,s3)
                    color_matrix_i.append(np.array([r,g,b,w]))
                    # color_matrix_i.append(np.array([r3, g3, b3,255]))
                    # color_matrix_i.append(np.array([r1,g1,b1,255]))
                # exit()
                s1_matrix.append(s1_matrix_i)
                s2_matrix.append(s2_matrix_i)
                s3_matrix.append(s3_matrix_i)
                color_matrix.append(color_matrix_i)
            s1_matrix = np.array(s1_matrix,dtype=float)
            s2_matrix = np.array(s2_matrix,dtype=float)
            s3_matrix = np.array(s3_matrix,dtype=float)
            color_matrix = np.array(color_matrix)
            outf = join(outdir_i,'frequency.tif')
            outf_s1 = join(outdir_i,'s1.tif')
            outf_s2 = join(outdir_i,'s2.tif')
            outf_s3 = join(outdir_i,'s3.tif')
            DIC_and_TIF().arr_to_tif(s1_matrix,outf_s1)
            DIC_and_TIF().arr_to_tif(s2_matrix,outf_s2)
            DIC_and_TIF().arr_to_tif(s3_matrix,outf_s3)
            # T.open_path_and_file(outdir_i)
            legend_arr = Ter.rgb_arr
            plt.imshow(legend_arr)
            plt.axis('off')
            plt.axis('equal')
            legend_f = join(outdir_i,'legend.pdf')
            # plt.savefig(legend_f)
            plt.figure()

            img = Image.fromarray(color_matrix.astype('uint8'), 'RGBA')
            plt.imshow(img)
            plt.show()
        #     img.save(outf)
        #     raster = gdal.Open(outf)
        #     geotransform = raster.GetGeoTransform()
        #     originX = DIC_and_TIF().originX
        #     originY = DIC_and_TIF().originY
        #     pixelWidth = DIC_and_TIF().pixelWidth
        #     pixelHeight = DIC_and_TIF().pixelHeight
        #
        #     raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        #     outRasterSRS = osr.SpatialReference()
        #
        #     # outRasterSRS.ImportFromEPSG(4326)
        #     # outRasterSRS.ImportFromEPSG(projection)
        #     # raster.SetProjection(outRasterSRS.ExportToWkt())
        #     raster.SetProjection('EPSG:4326')
        #     # T.open_path_and_file(outdir_i)
        #     # exit()
        # T.open_path_and_file(outdir)
        # exit()

    def get_raster_projections(self, rasterfn):
        '''
        get raster projection
        Agrs:
            rasterfn: tiff file path
        Returns:
            projection: projection of the raster
        '''
        raster = gdal.Open(rasterfn)
        projection = raster.GetProjection()
        return projection

    def frenquency_heatmap_bar(self):

        fdir = join(data_root, 'Frequency/LAI3g')
        outdir = join(self.this_class_png, 'LAI3g')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            arr = np.load(join(fdir, f))
            flag = 0
            strong_stabilization_sum_list = []
            weak_stabilization_sum_list = []
            amplification_sum_list = []
            for arr_i in arr:
                strong_stabilization = arr_i[:6]
                weak_stabilization = arr_i[6:12 - flag]
                amplification = arr_i[12 - flag:]
                flag += 1
                strong_stabilization_sum = np.sum(strong_stabilization)
                weak_stabilization_sum = np.sum(weak_stabilization)
                amplification_sum = np.sum(amplification)

                strong_stabilization_sum_list.append(strong_stabilization_sum)
                weak_stabilization_sum_list.append(weak_stabilization_sum)
                amplification_sum_list.append(amplification_sum)
            strong_stabilization_sum_list = np.array(strong_stabilization_sum_list)
            weak_stabilization_sum_list = np.array(weak_stabilization_sum_list)
            amplification_sum_list = np.array(amplification_sum_list)

            plt.bar(np.arange(len(arr)), strong_stabilization_sum_list, label='strong_stabilization')
            plt.bar(np.arange(len(arr)), weak_stabilization_sum_list, bottom=strong_stabilization_sum_list,
                    label='weak_stabilization')
            plt.bar(np.arange(len(arr)), amplification_sum_list,
                    bottom=strong_stabilization_sum_list + weak_stabilization_sum_list, label='amplification')
            # plt.legend()
            plt.title(f)
            outf = join(outdir, f.replace('.npy', '.pdf'))
            plt.savefig(outf)
            plt.close()
            # plt.show()
        T.open_path_and_file(outdir)

class Trend_analysis_plot(): ## this one is creating spatial map of all products at the same layout

    def __init__(self):
        pass
    def trend_analysis_plot(self):
        outdir = result_root + rf'trend_analysis\\trend_analysis_plot\\'

        period_list = [ 'late', 'early_peak', 'early_peak_late']
        # flist = self.get_trend_analysis_flist()
        flist_all_product = ['MCD_trend.tif', 'CABLE-POP_S2_lai_trend.tif', 'CLASSIC_S2_lai_trend.tif',
                             'CLM5_trend.tif', 'IBIS_S2_lai_trend.tif',
                             'ISAM_S2_LAI_trend.tif', 'ISBA-CTRIP_S2_lai_trend.tif',
                             'JSBACH_S2_lai_trend.tif', 'LPX-Bern_S2_lai_trend.tif',
                             'VISIT_S2_lai_trend.tif']
        period_rename_dict = {
            'late': 'Late season',
            'early_peak': 'Early- and peak season',
            'early_peak_late': 'Growing season',
        }

        # plt.show()
        for period in period_list:
            count = 1
            plt.figure(figsize=(11, 7))
            fdir = result_root + rf'trend_analysis\\zscore\\{period}\\'
            outdir_i = join(outdir, period)
            T.mk_dir(outdir_i, force=True)
            for f in tqdm(flist_all_product):
                if not f.endswith('.tif'):
                    continue
                if '_p_value' in f:
                    continue

                if 'MOD' in f:
                    continue
                if 'ensemble' in f:
                    continue
                if 'ORCHIDEE' in f:
                    continue
                if 'JULES' in f:
                    continue
                if 'LPJ' in f:
                    continue
                if 'OCN' in f:
                    continue
                if 'SDGVM' in f:
                    continue
                if 'YIBs' in f:
                    continue
                ax = plt.subplot(5,2,count)
                # print(count, f)
                count = count + 1
                # if not count == 9:
                #     continue

                fpath = join(fdir,f)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr = Tools().mask_999999_arr(arr, warning=False)
                arr_m = ma.masked_where(np.isnan(arr), arr)
                lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
                lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
                lon_list, lat_list = np.meshgrid(lon_list, lat_list)
                m = Basemap(projection='cyl', llcrnrlat=30, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='i',ax=ax)
                ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap='RdBu', zorder=-1, vmin=-0.1, vmax=0.1)
                coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
                plt.title(f.replace('_trend.tif','').replace('_S2','').replace('_lai','').replace('_LAI','').replace('_S2_lai','').replace('MCD','MODIS'))
                # plt.show()
            cax = plt.axes([0.5-0.15, 0.05, 0.3, 0.02])
            cb = plt.colorbar(ret, ax=ax,cax=cax, orientation='horizontal')
            cb.set_label(f'{period_rename_dict[period]} zscore trend (/yr)',labelpad=-40)
            outf = join(outdir_i, f'{period}.pdf')
            plt.subplots_adjust(wspace=0.029, hspace=0.)

            plt.savefig(outf, dpi=600)
            plt.close()
            # plt.show()
            # T.open_path_and_file(outdir)
            # exit()


class trends_seasonal_feedback():  ### 暂时不用 之前的最喜欢的图删掉了
    def __init__(self):
        self.product_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                        'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                         'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                        'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'Trendy_ensemble', 'MCD',]


    def run(self):
        # self.calculate_long_trends_seasonal_feedback()
        # self.calculate_long_trends_seasonal_feedback_modeling()
        # self.plot_statistic_long_trends_seasonal_feedback_from_tif()
        # df=self.long_trends_seasonal_feedback_from_tif_seperately()
        # self.plot_statistic_long_trends_seasonal_feedback_from_df_SI(df)
        self.plot_statistic_long_trends_seasonal_feedback_from_df_main()

    def calculate_long_trends_seasonal_feedback(self):
        fdir_early_peak=result_root+rf'\trend_analysis\original\\early_peak\\'
        fdir_late = result_root + rf'\trend_analysis\original\\late\\'
        outdir=result_root+r'\\long_trends_seasonal_feedback\\original\\'
        T.mk_dir(outdir,force=True)
        class_label_dict = {'stablilizing': -1, 'weak amplifying': 0, 'amplifying': 1, 'other': -2, }
        for model in tqdm(self.product_list):

            early_peak_f = join(fdir_early_peak, f'{model}_trend.tif')
            late_f = join(fdir_late, f'{model}_trend.tif')
            early_peak_dict = DIC_and_TIF().spatial_tif_to_dic(early_peak_f)
            late_dict = DIC_and_TIF().spatial_tif_to_dic(late_f)
            class_dict = {}
            class_dict_num = {}
            for pix in early_peak_dict:
                early_peak = early_peak_dict[pix]
                late = late_dict[pix]
                if np.isnan(early_peak):
                    continue
                if not early_peak > 0:
                    class_i = 'other'
                else:
                    if late <= 0:
                        class_i = 'stablilizing'
                    else:
                        if early_peak >= late:
                            class_i = 'weak amplifying'
                        else:
                            class_i = 'amplifying'
                class_dict[pix] = class_i
                class_dict_num[pix] = class_label_dict[class_i]
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(class_dict_num)
            plt.imshow(arr)
            plt.colorbar()
            plt.show()
            # outf = join(outdir, f'{model}.tif')
            # DIC_and_TIF().arr_to_tif(arr, outf)

    def calculate_long_trends_seasonal_feedback_modeling(self):
        fdir_early_peak=result_root+rf'\\trend_anaysis\original\\early_peak\\\\'
        fdir_late = result_root + rf'\\trend_anaysis\original\\late\\\\'
        outdir=result_root+r'\\long_trends_seasonal_feedback\\LAI\\'
        T.mk_dir(outdir,force=True)
        class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1, 'other': -2, }
        for model in tqdm(self.product_list):

            early_peak_f = join(fdir_early_peak, f'{model}_trend.tif')
            late_f = join(fdir_late, f'{model}_trend.tif')
            early_peak_dict = DIC_and_TIF().spatial_tif_to_dic(early_peak_f)
            late_dict = DIC_and_TIF().spatial_tif_to_dic(late_f)
            class_dict = {}
            class_dict_num = {}
            for pix in early_peak_dict:
                early_peak = early_peak_dict[pix]
                late = late_dict[pix]
                if np.isnan(early_peak):
                    continue
                if not early_peak > 0:
                    class_i = 'other'
                else:
                    if late <= 0:
                        class_i = 'strong stablilizing'
                    else:
                        if early_peak >= late:
                            class_i = 'weak stablilizing'
                        else:
                            class_i = 'amplifying'
                class_dict[pix] = class_i
                class_dict_num[pix] = class_label_dict[class_i]
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(class_dict_num)
            # plt.imshow(arr, cmap='jet_r', vmin=-2, vmax=1,interpolation='nearest')
            # plt.colorbar()
            # plt.title(model)
            # plt.show()
            outf = join(outdir, f'{model}.tif')
            DIC_and_TIF().arr_to_tif(arr, outf)

    def long_trends_seasonal_feedback_from_tif_seperately(self):  ## create humid and arid dataframe seperately
        fdir = result_root+rf'long_trends_seasonal_feedback\\LAI\\'


        product_list = ['MCD','Trendy_ensemble','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                         'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]


        # class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1, 'other': -2, }
        class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1, }

        trend_mark_dict_reverse = T.reverse_dic(class_label_dict)
        all_spatial_dic = {}
        for product in product_list:

            fpath = join(fdir, f'{product}.tif')

            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            all_spatial_dic[product] = spatial_dict
        df= T.spatial_dics_to_df(all_spatial_dic)
        df=Dataframe_func(df).df       # add region
        T.print_head_n(df)
        return df


    def plot_statistic_long_trends_seasonal_feedback_from_tif(self):  #### 生成整体all
        fdir = result_root+rf'long_trends_seasonal_feedback\\LAI\\'


        product_list = ['MCD','Trendy_ensemble','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                         'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',]

        color_list = ['#F7B36B', '#F9E29F', ]
        color_list.extend(['lavender'] * 14)

        class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1, }

        trend_mark_dict_reverse = T.reverse_dic(class_label_dict)
        result_dict = {}
        for product in product_list:

            fpath = join(fdir, f'{product}.tif')

            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            dict_i = {}

            for pix in spatial_dict:
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                val = int(val)
                if val == -2:
                    continue
                mark_i = trend_mark_dict_reverse[val][0]
                mark_i_class = class_label_dict[mark_i]
                if not mark_i_class in dict_i:
                    dict_i[mark_i_class] = []
                dict_i[mark_i_class].append(pix)
            total=0
            for mark_i_class in dict_i:
                total= total + len(dict_i[mark_i_class])
            result_dict_i={}
            for mark_i_class in dict_i:
                ratio=len(dict_i[mark_i_class])/total
                result_dict_i[mark_i_class]=ratio*100
            result_dict[product]=result_dict_i

        df = pd.DataFrame(result_dict)

        T.print_head_n(df)
        T.save_df(df, result_root + f'\\long_trends_seasonal_feedback\\df\\all.df')
        T.df_to_excel(df, result_root + f'\\long_trends_seasonal_feedback\\df\\all.xlsx')


    def plot_statistic_long_trends_seasonal_feedback_from_df_main(self):
        fdir=result_root+rf'long_trends_seasonal_feedback\df\\'


        for df in os.listdir(fdir):
            if not df.endswith('.df'):
                continue
            dff=T.load_df(fdir+df)
            region=df.split('.')[0]

            product_list = ['MCD','Trendy_ensemble']

            class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1,  }

            trend_mark_dict_reverse = T.reverse_dic(class_label_dict)


            for product in product_list:
                cm = 1 / 2.54

                plt.figure(figsize=(5 * cm, 6 * cm))

                mark_i_list = []
                column_name = f'{product}'
                ###extract the column
                dff_temp = dff[[column_name]]
                ## define label as trend_mark_dict_reverse
                for i in range(len(dff_temp.index)):
                    print(dff_temp.index[i])

                    mark_i=trend_mark_dict_reverse[dff_temp.index[i]][0]
                    mark_i_list.append(mark_i)
                plt.bar(dff_temp.index, dff_temp[column_name], color='lavender')
                x_ticks_labels = mark_i_list
                plt.xticks(dff.index, x_ticks_labels, rotation=45, horizontalalignment='right')

                plt.ylabel('Percentage (%)')
                plt.tight_layout()

                plt.yticks([0, 20, 40, 60])
                plt.title(f'{product}_{region}')
                plt.ylim(0, 60)
                plt.show()





    def plot_statistic_long_trends_seasonal_feedback_from_df_SI(self,df):

        product_list = ['MCD','Trendy_ensemble','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai',
                         'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]


        color_list = ['#F7B36B',  ]
        color_list.extend(['lavender'] * 14)

        # class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1, 'other': -2, }
        class_label_dict = {'strong stablilizing': -1, 'weak stablilizing': 0, 'amplifying': 1,  }

        trend_mark_dict_reverse = T.reverse_dic(class_label_dict)
        result_dict = {}

        for region in ['Dryland','Humid']:
            df_region=df[df['HI_class']==region]

            for product in product_list:
                spatial_dict=T.df_to_spatial_dic(df_region,product)
                dict_i = {}

                for pix in spatial_dict:
                    val = spatial_dict[pix]
                    if np.isnan(val):
                        continue
                    val = int(val)
                    if val == -2:
                        continue
                    mark_i = trend_mark_dict_reverse[val][0]
                    mark_i_class = class_label_dict[mark_i]
                    if not mark_i_class in dict_i:
                        dict_i[mark_i_class] = []
                    dict_i[mark_i_class].append(pix)
                total=0
                for mark_i_class in dict_i:
                    total= total + len(dict_i[mark_i_class])
                result_dict_i={}
                for mark_i_class in dict_i:
                    ratio=len(dict_i[mark_i_class])/total
                    result_dict_i[mark_i_class]=ratio*100
                result_dict[product]=result_dict_i

            df_new = pd.DataFrame(result_dict)

##############start to plot figure

            T.save_df(df_new, result_root + f'\\long_trends_seasonal_feedback\\df\\{region}.df')
            T.df_to_excel(df_new, result_root + f'\\long_trends_seasonal_feedback\\df\\{region}.xlsx')
            df_new = df_new.T
            T.print_head_n(df_new)

            cm = 1 / 2.54




            for column in df_new:
                plt.figure(figsize=(7 * cm, 6 * cm))

                print(trend_mark_dict_reverse[column][0])

                df_new[column] = df_new[column].astype(float)
                plt.bar(df_new.index, df_new[column], color=color_list)
                plt.xticks([''])
                plt.ylabel('Percentage (%)')
                plt.yticks([0, 20, 40, 60])
                plt.title(f'{trend_mark_dict_reverse[column][0]}_{region}')
                plt.ylim(0, 60)
                # set MCD as a standard
                plt.axhline(y=df_new.loc['MCD', column], color='k', linestyle='--', linewidth=0.5)

                plt.tight_layout()
                # plt.show()
                plt.savefig(result_root + f'\\long_trends_seasonal_feedback\\SI_bar\\{region}_{trend_mark_dict_reverse[column][0]}.pdf', dpi=300)


class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)

        print('add NDVI mask')
        df = self.add_NDVI_mask(df)

        # if is_clean_df == True:
        #     df = self.clean_df(df)

        # print('add landcover')
        # df = self.add_GLC_landcover_data_to_df(df)

        print('add Aridity Index')
        df = self.add_AI_to_df(df)


        print('add AI_reclass')
        df = self.AI_reclass(df)


        self.df = df

    def clean_df(self,df):

        df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]
        # df = df[df['ELI_significance'] == 1]
        return df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic.npy')
        val_dic=T.load_npy(f)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['landcover_GLC'] = val_list
        return df

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'Base_data', 'NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):
        f = join(data_root, 'Base_data/Aridity_Index/aridity_index.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'HI_class')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df


    def AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            AI = row['HI_class']
            if AI < 0.65:
                AI_class.append('Dryland')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['HI_class'] = AI_class
        return df




class long_term_seasonal_feedbacks_window_anaysis():  ##这个函数没有用
    def run(self):
        self.trend_anaysis()
        # self.window_extraction_trend()
        # self.trend_window_trend()
        # self.bivariate_trend()

        pass
    def trend_anaysis(self):  ###这个求得是全球平均 average and std

        dic_mask_lc_file = data_root + 'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root + 'Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        product_list=['Trendy_ensemble', 'MCD',]

        period = 'early_peak'

        fdir_all = result_root + rf'zscore\LAI\{period}\\'
        outdir = result_root + rf'trend_anaysis\\zscore\\{period}\\'
        T.mk_dir(outdir, force=True)
        result_dic = {}
        for f in tqdm(os.listdir(fdir_all)):
            fname=f.split('.')[0]
            if not fname in product_list:
                continue
            if not f.endswith('.npy'):
                continue
            fpath=join(fdir_all,f)


            # fpath=join(fdir_all,fdir)
            print(fpath)

            outf_i = join(outdir, f)
            # if os.path.isfile(outf_i):
            #     continue
            dic = T.load_npy(fpath)
            trend_dic={}
            for pix in dic:

                if pix not in dic_mask_lc:
                    continue
                if dic_mask_lc[pix] == 'Crop':
                    continue
                val_lc_change = array_mask_landcover_change[pix]
                if val_lc_change == np.nan:
                    continue

                time_series = dic[pix]

                # print(time_series)
                time_series = np.array(time_series)
                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue

                try:
                    a,b,r,p=T.nan_line_fit(range(len(time_series)),time_series)
                    trend_dic[pix]=a
                    # print(a)
                except:
                    trend_dic[pix]=np.nan
            result_dic[fname]=trend_dic

        df = T.spatial_dics_to_df(result_dic)
        df = Dataframe_func(df).df
        T.print_head_n(df)
        for region in ['Dryland','Humid']:

            df_region=df[df['HI_class']==region]
            for col_name in product_list:
                if 'pix' in col_name:
                    continue
                df_region[col_name] = df_region[col_name].astype(float)
                average = np.nanmean(df_region[col_name])
                std = np.nanstd(df_region[col_name])*0.25
                print(f'{col_name} {region} average:{average:.2f} std:{std:.2f}')


            pass
    def window_extraction_trend(self):
        period= 'late'
        fdir_all = result_root + rf'extraction_original_val\\LAI\\'
        outdir = result_root + rf'\\window_analysis_trend\\{period}\\'
        T.mk_dir(outdir, force=True)
        for fdir in tqdm(os.listdir(fdir_all)):
            if not 'MCD' in fdir:
                continue

            for f in os.listdir(fdir_all + fdir):
                if not f'during_{period}_{fdir}' in f:
                    continue


                if not f.endswith('.npy'):
                    continue
                fpath = join(fdir_all, fdir, f)
                outf_i = join(outdir, fdir)
                if os.path.isfile(outf_i):
                    continue
                dic = T.load_npy(fpath)
                window = 10

                new_x_extraction_by_window_trend = {}
                for pix in tqdm(dic):

                    time_series = dic[pix]
                    time_series = np.array(time_series)

                    time_series[time_series < -999] = np.nan
                    if np.isnan(np.nanmean(time_series)):
                        print('error')
                        continue
                    print((len(time_series)))
                    ### if all values are identical, then continue
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue


                    new_x_extraction_by_window_trend[pix] = self.forward_window_extract_trend(time_series, window)


                np.save(outf_i, new_x_extraction_by_window_trend)

    def trend_window_trend(self):
        fdir=result_root+rf'\window_analysis_trend\early_peak\\'
        outdir=result_root+rf'\window_analysis_trend_trend\early_peak\\'
        T.mk_dir(outdir,force=1)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            fpath=join(fdir,f)
            outf=join(outdir,f)
            dic=T.load_npy(fpath)
            new_dic={}
            for pix in dic:
                vals=dic[pix]
                if len(vals)==0:
                    continue
                if np.nanmax(vals)==np.nanmin(vals):
                    continue

                a,b,r,p,q=stats.linregress(range(len(vals)),vals)
                new_dic[pix]=a
            np.save(outf,new_dic)
            array = DIC_and_TIF().pix_dic_to_spatial_arr(new_dic)
            DIC_and_TIF().arr_to_tif(array, outf.replace('.npy', '.tif'))

    pass

    def bivariate_trend(self):
        product_list= ['MCD','Trendy_ensemble','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                         'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',]
        import bivariate_package
        for product in product_list:
            tif1 = rf'D:\Greening\Result\trend_anaysis\original\\early_peak\{product}_trend.tif'
            tif2 = rf'D:\Greening\Result\trend_anaysis\original\\late/{product}_trend.tif'
            outdir=result_root+rf'\trend_anaysis\bivariate_result\\'
            T.mk_dir(outdir,force=1)

            outf = join(outdir, f'{product}.tif')
            x_label = 'early_peak_trend'
            y_label = 'late_trend'
            min1 = -0.03
            max1 = 0.03
            min2 = -0.03
            max2 = 0.03
            bivariate_package.Bivariate_plot().plot_bivariate_map(tif1, tif2, x_label, y_label, min1, max1, min2, max2, outf,n=(3,3))

    def forward_window_extract_anomaly(self, x, window):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window=[]
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                anomaly = []
                x_vals=[]
                for w in range(window):
                    x_val=(x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                # x_mean=np.nanmean(x_vals)

                # for i in range(len(x_vals)):
                #     if x_vals[0]==None:u
                #         continue
                #     x_anomaly=x_vals[i]-x_mean
                #
                #     anomaly.append(x_anomaly)
                new_x_extraction_by_window.append(x_vals)


        return new_x_extraction_by_window

    def forward_window_extract_trend(self, x, window): # extract 后在计算trend
        # 前窗滤波
        # window = window-1
        # 不改变数据长度
        # plt.plot(x)
        # plt.show()


        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_trend_by_window=[]
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:

                x_vals=[]
                for w in range(window):
                    x_val=(x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue
                ## 计算trned
                x_vals=np.array(x_vals)
                x_vals[x_vals < -999] = np.nan
                # r,p=stats.pearsonr(x_vals,range(len(x_vals)))
                # r1,p1=np.polyfit(x_vals,range(len(x_vals)),1)
                try:
                    a,b,r,p,q,=stats.linregress(range(len(x_vals)),x_vals)
                except:
                    a=np.nan
                # print(r1,a)
                # exit()
                new_x_trend_by_window.append(a)
        print(len(new_x_trend_by_window))


        return new_x_trend_by_window


class build_dataframe():




    def __init__(self):

        self.this_class_arr = result_root + 'Data_frame\distibution_across_pft\\'
        # self.this_class_arr = result_root + rf'Data_frame\Frequency\Trendy_ensemble\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'distibution.df'
        self.P_PET_fdir = data_root+rf'\Base_data\aridity_P_PET_dic\\'
        pass

    def run(self):

        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)
        df=self.add_trend_to_df(df)
        # df=self.add_detrend_zscore_to_df(df)
        #
        df = self.add_row(df)

        df = self.add_max_trend_to_df(df)

        df = self.add_NDVI_mask(df)

        df = self.add_GLC_landcover_data_to_df(df)
        #
        # P_PET_dic = self.P_PET_ratio(self.P_PET_fdir)
        # P_PET_reclass_dic = self.P_PET_reclass_2(P_PET_dic)
        # df = T.add_spatial_dic_to_df(df, P_PET_reclass_dic, 'HI_class')


        # df=self.show_field(df)
        # df = self.drop_field_df(df)
        # df=self.rename_dataframe_columns(df)

        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff)

    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass

    def foo1(self, df):

        f = result_root + rf'detrend_zscore\LAI\early_peak\MCD.npy'
        dic = {}
        outf = self.dff
        result_dic = {}
        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []
        f_name = f.split('\\')[-2]+'_'+f.split('\\')[-1].split('.')[0]
        print(f_name)

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 2002
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y + 1)
                y = y + 1
                if y+1>2021:
                    break
        df['pix'] = pix_list
        df[f_name] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def foo2(self, df):  # 新建trend

        f = rf'D:\Greening\Result\Data_frame\Frequency\MCD\tif\MCD\MCD\\strong_stabilization.tif'
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)

        dic=DIC_and_TIF().spatial_arr_to_dic(arr)


        # exit()

        pix_list = []
        for pix in tqdm(dic):
            pix_list.append(pix)
        df['pix'] = pix_list

        return df

    def add_detrend_zscore_to_df(self,df):
        period_list = ['peak', ]
        for period in period_list:


            fdir=result_root + rf'detrend_zscore\climate_monthly\{period}\\'

            for f in os.listdir(fdir):


                if not f.endswith('npy'):
                    continue

                NDVI_dic = T.load_npy(fdir+f)

                f_name = fdir.split('\\')[-3]+'_'+f.split('\\')[-1].split('.')[0]
                print(f_name)

                NDVI_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in NDVI_dic:
                        NDVI_list.append(np.nan)
                        continue

                    vals = NDVI_dic[pix]
                    # print(len(vals))
                    ## nanmean
                    vals = np.array(vals)
                    vals[vals < -999] = np.nan
                    if np.isnan(np.nanmean(vals)):
                        NDVI_list.append(np.nan)
                        continue
                    try:
                        v1 = vals[year - 2003]
                        NDVI_list.append(v1)
                    except:
                        NDVI_list.append(np.nan)

                df[f_name] = NDVI_list
        return df


    def add_row(self,df):
        r_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r,c=pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_max_trend_to_df(self, df):


        fdir = data_root+rf'/Base_data/lc_trend/'
        for f in (os.listdir(fdir)):
            # print()
            if not 'max_trend' in f:
                continue
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue

            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val * 20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list

        return df


    def add_trend_to_df(self, df):


        fdir = result_root+rf'\Data_frame\Frequency\MCD\tif\MCD\MCD\\'
        for f in (os.listdir(fdir)):
            # print()

            if not f.endswith('.tif'):
                continue
            if 'freq' in f:
                continue


            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            val_dic=DIC_and_TIF().spatial_arr_to_dic(arr)
            f_name = f.split('.')[0]+'_TRENDY_ensemble'
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]

                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)

            df[f_name] = val_list

        return df
    def add_NDVI_mask(self,df):
        f =data_root+rf'/Base_data/NDVI_mask.tif'


        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df
    def add_GLC_landcover_data_to_df(self, df):


        f = data_root+rf'\Base_data\LC_reclass2.npy'

        val_dic=T.load_npy(f)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df

    def P_PET_ratio(self, P_PET_fdir):

        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            vals[vals == 0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_reclass(self,dic):
        dic_reclass = {}
        for pix in dic:
            val = dic[pix]
            label = None
            # label = np.nan
            if val > 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Arid'
                # label = 0
            elif val > 0.2 and val < 0.5:
                label = 'Semi Arid'
                # label = 1
            elif val > 0.5 and val < 0.65:
                label = 'Semi Humid'
                # label = 2
            dic_reclass[pix] = label
        return dic_reclass


    def P_PET_reclass_2(self,dic):
        dic_reclass = {}
        for pix in dic:
            val = dic[pix]
            label = None
            # label = np.nan
            if val > 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Dryland'
                # label = 0
            elif val > 0.2 and val < 0.5:
                label = 'Dryland'
                # label = 1
            elif val > 0.5 and val < 0.65:
                label = 'Dryland'
                # label = 2
            dic_reclass[pix] = label
        return dic_reclass
    def show_field(self,df):
        for col in df.columns:
            print(col)
        return df

        pass

    def rename_dataframe_columns(self,df):
        df.rename(columns={'late_LPX-Bern_S2_lai': 'late_LPX_Bern_S2_lai',



                          }, inplace=True)
        return df
class long_term_trends():   ### early-and peak vs. late and early-peak and late season together

    def __init__(self):

        self.this_class_arr = result_root + 'Data_frame\zscore\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'zscore.df'

    def run(self):
        df = self.__gen_df_init(self.dff)
        # self.zscore_result_statistical_annual(df)
        # self.calculate_average(df)


        # self.plot_time_series_zscore()
        self.create_new_df_before_plot()
        # self.plot_feedback_vs_trend()
        # self.barplot()


        pass
    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass


    def zscore_result_statistical_annual(self,df): # 实现变量的三个季节画在一起

        df = df[df['row'] < 120]

        df = df[df['max_trend'] < 10]

        df = df[df['landcover_GLC'] != 'Crop']
        period_name=['early_peak','late','early_peak_late']
        product_list=['MCD','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai',  'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]

        result_dic={}
        for region in ['Humid','Dryland']:

            for period in period_name:

                df_pick=df[df['HI_class']==region]

                for variable in product_list:

                    column_name=f'{period}_{variable}'

                    print(column_name)

                    mean_value_yearly,up_list,bottom_list,fit_value_yearly,k_value,p_value=self.plot_calculation(df_pick,column_name)


                    key=f'{region}_{period}_{variable}'
                    result_dic[key]={
                        'mean_value_yearly':mean_value_yearly,
                        'up_list':up_list,
                        'bottom_list':bottom_list,
                        'fit_value_yearly':fit_value_yearly,
                        'k_value':k_value,
                        'p_value':p_value,

                    }
        outdir=result_root+rf'\\Data_frame\zscore_result_statistical_annual\\'
        T.mk_dir(outdir,force=1)
        outf=outdir+'zscore_result_statistical_annual2.npy'
        T.save_npy(result_dic,outf)
    def plot_calculation(self,df,column_name):
        dic = {}
        mean_val = {}
        confidence_value = {}
        std_val = {}
        # year_list = df['year'].to_list()
        # year_list = set(year_list)  # 取唯一
        # year_list = list(year_list)
        # year_list.sort()

        year_list = []
        for i in range(2003, 2022):
            year_list.append(i)
        print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan

            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            std_val_i = np.nanstd(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i
            std_val[year] = std_val_i

        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list = []  # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        xaxis = list(xaxis)
        print(len(mean_val_list))
        # r, p_value = stats.pearsonr(xaxis, mean_val_list)
        # k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)
        k_value, b_value, r, p_value = T.nan_line_fit(xaxis, mean_val_list)
        print(k_value)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly = []
        p_value_yearly = []

        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            # up_list.append(mean_val[year] + confidence_value[year])
            # bottom_list.append(mean_val[year] - confidence_value[year])
            up_list.append(mean_val[year] + 0.125 * std_val[year])
            bottom_list.append(mean_val[year] - 0.125 * std_val[year])

            fit_value_yearly.append(k_value * (year - year_list[0]) + b_value)



        return mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value
        # exit()

    pass
    def calculate_average(self,df):## 实现计算中位数 of individual product
        import pprint
        f_individual = result_root + rf'\\Data_frame\zscore_result_statistical_annual\\zscore_result_statistical_annual2.npy'
        dic_individual = T.load_npy(f_individual)
        # print(dic_individual)
        # pprint.pprint(dic_individual)
        # exit()


        product_list = [ 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                        'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                        'LPX-Bern_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]
        period_name = ['early_peak', 'late', 'early_peak_late']
        region_list = ['Humid', 'Dryland']
        # T.print_head_n(df)
        # exit()
        # result_dic = {}
        for region in region_list:
            for period in period_name:
                whole_list = []
                for product in product_list:
                    column_name = f'{period}_{product}'
                    df_pick = df[df['HI_class'] == region]

                    mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value = self.plot_calculation(
                        df_pick, column_name)
                    whole_list.append(mean_value_yearly)
                        ## average
                average_list = np.nanmean(whole_list, axis=0)
                ## calculate k,p
                xaxis = range(len(average_list))
                xaxis = list(xaxis)
                k_value, b_value, r, p_value = T.nan_line_fit(xaxis, average_list)
                fit_value_yearly_average = []
                for year in range(2003, 2022):

                    fit_value_yearly_average.append(k_value * (year - 2003) + b_value)

                dic_individual[f'{region}_{period}_average'] = {
                    'mean_value_yearly': average_list,
                    'up_list':[],
                    'bottom_list': [],
                    'fit_value_yearly': fit_value_yearly_average,
                    'k_value': k_value,
                    'p_value': p_value,}
                # pprint.pprint(result_dic)
                # dic_individual.update(result_dic)
                # exit()

        outdir = result_root + rf'\\Data_frame\zscore_result_statistical_annual\\'
        T.mk_dir(outdir, force=1)
        outf = outdir + 'zscore_result_statistical_annual_average_new.npy'
        T.save_npy(dic_individual, outf)
        pprint.pprint(dic_individual)



    def plot_time_series_zscore(self):  ###plot time series
        f = result_root + rf'\\Data_frame\zscore_result_statistical_annual\\zscore_result_statistical_annual_average_new.npy'
        dic = T.load_npy(f)
        period_name = ['early_peak', 'late', 'early_peak_late']


        product_list = ['MCD', 'average', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                        'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                        'LPX-Bern_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]
        color_list = ['green', 'black']
        color_list.extend('silver' for i in range(len(product_list) - 2))
        linewidth_list = [2, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                          0.5]

        centimeter = 1 / 2.54


        fig = plt.figure(figsize=(18 * centimeter, 12 * centimeter))
        i = 1

        for region in ['Humid', 'Dryland']:

            for period in period_name:
                ax = fig.add_subplot(2, 3, i)

                flag = 0
                for variable in product_list:

                    color = color_list[flag]
                    linewidth = linewidth_list[flag]
                    if flag >= 2:
                        zorder = 0
                    else:
                        zorder = 1

                    flag += 1
                    key = f'{region}_{period}_{variable}'
                    result_i = dic[key]
                    mean_value_yearly = result_i['mean_value_yearly']
                    # up_list = result_i['up_list']
                    # bottom_list = result_i['bottom_list']
                    fit_value_yearly = result_i['fit_value_yearly']
                    k_value = result_i['k_value']
                    p_value = result_i['p_value']
                    print(f'{region}_{variable}', 'k={:0.2f},p={:0.4f}'.format(k_value, p_value))
                    yearlist=range(2003,2022)

                    plt.plot(yearlist,mean_value_yearly, label=variable, c=color, zorder=zorder, linewidth=linewidth)
                    plt.plot(yearlist,fit_value_yearly, linestyle='--', label='k={:0.2f},p={:0.2f}'.format(k_value, p_value),
                             c=color, linewidth=linewidth)
                    # print(f'{region}_{variable}','k={:0.2f},p={:0.4f}'.format(k_value, p_value))
                    # plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.1, zorder=-1,
                    #                  color=color)
                    #### show text p value trend only show Trendy_ensemble and MCD
                    if variable == 'average' or variable == 'MCD':
                        ## position of text
                        x = 0
                        y = mean_value_yearly[0]+0.7
                        # plt.text(x, y, 'k={:0.2f},p={:0.2f}'.format(k_value, p_value), fontsize=10, color=color)


                plt.ylabel('zscore')
                plt.xlabel('year')
                major_xticks = np.arange(2003, 2022, 5)
                plt.xticks(major_xticks, rotation=45)
                plt.title(f'{period}_{region}')

                # major_yticks = np.arange(-10, 15, 5)
                major_yticks = np.arange(-0.8, 0.9, 0.4)
                plt.ylim(-0.8, 0.8)
                # major_ticks = np.arange(0, 40, 5)  ### 根据数据长度修改这里
                ax.set_xticks(major_xticks)
                ax.set_yticks(major_yticks)
                plt.grid(which='major', alpha=0.5)
                plt.tight_layout()
                i = i + 1
        outdir = result_root + rf'\\Data_frame\zscore_result_statistical_annual\\'
        T.mk_dir(outdir, force=1)
        outf = outdir + 'zscore_result_statistical_annual_average_new.pdf'
        plt.savefig(outf)
        plt.close()


        plt.show()

    def create_new_df_before_plot(self):  ### prepare for plot feedback_vs_trend
        f_trend = result_root + rf'\\Data_frame\zscore_result_statistical_annual\\zscore_result_statistical_annual2.npy'
        dic_trend = T.load_npy(f_trend)
        dff=result_root+rf'Data_frame\zscore_result_statistical_annual\frequency_temporal_change_TRENDY_individual_statistic\\result.df'
        df=T.load_df(dff)
        dff_new = result_root + rf'Data_frame\zscore_result_statistical_annual\frequency_temporal_change_TRENDY_individual_statistic\\result_new.df'
        period_name = ['early_peak', 'late','early_peak_late']
        product_list = ['MCD', 'Trendy_ensemble', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                        'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                        'LPX-Bern_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]
        # T.rename_dataframe_columns()
        model_name = df['model'].to_list()
        new_model_name_list = []
        for model in model_name:
            new_model_name = model.replace('Arid-', 'Dryland-')
            new_model_name_list.append(new_model_name)
        df = df.drop(columns='model')
        df['model'] = new_model_name_list
        # T.print_head_n(df)
        # exit()
        dict_result = {}


        for region in ['Humid', 'Dryland']:

            for variable in product_list:
                dict_j = {}
                new_key = f'{region}-{variable}'

                for period in period_name:

                    key = f'{region}_{period}_{variable}'

                    result_i = dic_trend[key]
                    k_value = result_i['k_value']
                    p_value = result_i['p_value']
                    dict_i = {
                        f'k_value_{period}': k_value,
                        f'p_value_{period}': p_value,
                    }
                    dict_j.update(dict_i)
                dict_result[new_key]=dict_j

        df_result = T.dic_to_df(dict_result,'model')
        # T.print_head_n(df_result)
        # exit()
        model_list = df_result['model'].to_list()
        print(model_list)
        T.print_head_n(df_result)
        ###save to df
        # df = pd.merge(df, df_result, on='model', how='left')
        # df_merge = pd.DataFrame()
        df_list = [df_result]
        print('-------------------')
        T.print_head_n(df)
        print('-------------------')
        T.print_head_n(df_result)
        df_merge = T.join_df_list(df,df_list,'model')
        df_merge = df_merge.dropna(how='any')
        T.print_head_n(df_merge)

        T.save_df(df_merge, dff_new)
        ##to excel
        T.df_to_excel(df_merge,dff_new, n=1000, random=False)




    def frequency_temporal_change_TRENDY_individual_statistic_boxplot(self):
        fdir = join(self.this_class_arr, 'frequency_temporal_change_TRENDY_individual_statistic')
        dff = join(fdir, 'result.df')
        df = T.load_df(dff)

        # T.print_head_n(df)
        model_list = df['model'].tolist()
        cols = ['a_amp', 'a_weak_stab', 'a_strong_stab']

        box_list = []
        label_list = []
        mcd_list = []
        ensemble_list = []
        individual_list = []
        for AI in ['Arid', 'Humid']:
            for col in cols:
                model_list_i = []
                for model in model_list:
                    AI_ = model.split('-')[0]
                    if AI_ == AI:
                        model_list_i.append(model)

                individual_models = []
                for model in model_list_i:
                    if 'MCD' in model:
                        continue
                    if 'ensemble' in model:
                        continue
                    individual_models.append(model)

                df_i = df[df['model'].isin(model_list_i)]
                vals = df_i[col].tolist()
                box_list.append(vals)
                label_list.append(f'{AI}-{col}')
                df_mcd = df_i[df_i['model'].str.contains('MCD')]
                df_ensemble = df_i[df_i['model'].str.contains('ensemble')]
                df_individual = df_i[df_i['model'].isin(individual_models)]
                val_mcd = df_mcd[col].tolist()[0]
                val_ensemble = df_ensemble[col].tolist()[0]
                vals_individual = df_individual[col].tolist()
                mcd_list.append(val_mcd)
                ensemble_list.append(val_ensemble)
                individual_list.append(vals_individual)

        plt.boxplot(box_list, labels=label_list, showfliers=False, zorder=-1)
        plt.scatter(np.arange(1, len(mcd_list) + 1), mcd_list, color='red', marker='*', label='MCD', zorder=2, s=70)
        plt.scatter(np.arange(1, len(ensemble_list) + 1), ensemble_list, color='blue', marker='X', label='ensemble',
                    zorder=2, s=70)
        for i in range(len(individual_list)):
            plt.scatter([i + 1] * len(individual_list[i]), individual_list[i], color='gray', marker='x', zorder=-1)
        plt.hlines(0, 0.5, len(label_list) + 0.5, color='black', zorder=-3, linestyles='--')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        pass

    def plot_feedback_vs_trend(self):
        dff=result_root+rf'Data_frame\zscore_result_statistical_annual\frequency_temporal_change_TRENDY_individual_statistic\\result_new.df'
        df=T.load_df(dff)
        marker_list = ['*','o', 's', 'D', 'v', 'p', 'P', '^', 'X', 'd', '<', '>']


        color_list = T.gen_colors(12)

        ##plot x =feedback and y = trend
        periods = ['early_peak_late','late']
        product_list = ['MCD', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
            'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
            'LPX-Bern_S2_lai', 'VISIT_S2_lai',  ]
        region_list=['Humid','Dryland']

        for region in region_list:

            for period in periods:
                x_list = []
                y_list = []
                label_list = []
                p_value_list_sta=[]
                p_value_list_trend=[]
                flag = 0
                for variable in product_list:

                    key=f'{region}-{variable}'
                    df_pick=df[df['model']==key]
                    y=df_pick[f'k_value_{period}'].to_list()
                    x=df_pick[f'a_strong_stab'].to_list()
                    x=x[0]*100
                    y=y[0]

                    p_value_sta=df_pick[f'p_strong_stab'].to_list()
                    p_value_trend=df_pick[f'p_value_{period}'].to_list()
                    x_list.append(x)

                    y_list.append(y)
                    p_value_list_sta.append(p_value_sta)
                    p_value_list_trend.append(p_value_trend)

                    label_list.append(f'{region}-{variable}-{period}')
                centimeter = 2.54
                plt.figure(figsize=(9/centimeter, 7/centimeter))

                for i in range(len(x_list)):
                    x=x_list[i]
                    y=y_list[i]
                    label=label_list[i]
                    p_value_trend=p_value_list_trend[i]

                    p_value_sta=p_value_list_sta[i]
                    plt.scatter(x,y,label=label,marker=marker_list[flag],color=color_list[flag])
                    plt.text(x, y, f'P={p_value_trend[0]:0.2f}\nP={p_value_sta[0]:0.2f}', fontsize=6, color='black')

                    flag+=1

                plt.title(f'{region}-{period}')
                plt.xlabel('strong_stabilization_feedback_trend (%/year)')
                plt.ylabel('interannual_growing_season_trend(/year)')
                plt.legend()
                plt.tight_layout()
                outdir=result_root+rf'\\Data_frame\zscore_result_statistical_annual\frequency_temporal_change_TRENDY_individual_statistic\\'
                outf=join(outdir,f'{region}_{period}.pdf')
                plt.savefig(outf)
                plt.close()


                # plt.show()

    def barplot(self):
        dff=result_root+rf'Data_frame\zscore_result_statistical_annual\frequency_temporal_change_TRENDY_individual_statistic\\result_new.df'
        df=T.load_df(dff)
        product_list = ['MCD','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
            'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
            'LPX-Bern_S2_lai', 'VISIT_S2_lai',  ]
        region_list=['Humid','Dryland']
        for region in region_list:
            trend_list=[]
            p_list=[]
            star_list=[]
            for variable in product_list:

                key=f'{region}-{variable}'
                df_pick=df[df['model']==key]

                trend=df_pick[f'a_amp'].to_list()
                p_value=df_pick[f'p_amp'].to_list()
                trend_list.append(trend)
                p_list.append(p_value)
                if p_value[0]<0.05:
                    star_list.append('**')
                else:
                    star_list.append('')
            plt.figure()

            for i in range(len(trend_list)):
                trend=trend_list[i]


                variable=product_list[i]

                plt.bar(variable,trend,label=variable)



            for idx, pval in enumerate(star_list):
                y_position = trend_list[idx][0]-0.001

                plt.text(x=idx, y=y_position, s=pval, ha='center', va='bottom', fontsize=20, color='black')

            plt.title(region)
            plt.ylabel('strong_stabilization_trend')
            plt.xticks(rotation=45)

            # plt.legend()
            plt.tight_layout()
            plt.show()





    pass
class build_dataframe_window_anaysis():
    def __init__(self):

        self.this_class_arr = result_root + 'Data_frame\dataframe_window_anaysis\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Dataframe_window_anaysis.df'
        self.P_PET_fdir = rf'C:\Users\pcadmin\Desktop\Data\Base_data\aridity_P_PET_dic\\'
        pass





    def run(self):

        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        df=self.add_detrend_zscore_to_df(df)

        df = self.add_row(df)
        #
        df = self.add_max_trend_to_df(df)

        df = self.add_NDVI_mask(df)

        df = self.add_GLC_landcover_data_to_df(df)

        P_PET_dic = self.P_PET_ratio(self.P_PET_fdir)
        P_PET_reclass_dic = self.P_PET_reclass_2(P_PET_dic)
        df = T.add_spatial_dic_to_df(df, P_PET_reclass_dic, 'HI_class')

        # df=self.__rename_dataframe_columns(df)
        # df=self.show_field(df)
        # df = self.drop_field_df(df)

        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff)

    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass

    def foo1(self, df):

        f = result_root + rf'detrend_zscore\climate_monthly\early_peak\MCD.npy'
        dic = {}
        outf = self.dff
        result_dic = {}
        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []
        f_name = f.split('\\')[-2]+'_'+f.split('\\')[-1].split('.')[0]
        print(f_name)

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 2002
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y + 1)
                y = y + 1
        df['pix'] = pix_list
        df[f_name] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def foo2(self, df):  # 新建trend

        f = 'zscore\daily_Y\peak/during_peak_LAI3g_zscore.npy'
        val_array = np.load(f)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            pix_list.append(pix)
        df['pix'] = pix_list

        return df

    def add_detrend_zscore_to_df(self,df):
        period_list = ['early', 'peak', 'late','early_peak']
        for period in period_list:

            fdir=result_root + rf'detrend_zscore\climate_monthly\{period}\\'

            for f in os.listdir(fdir):
                if not f.endswith('npy'):
                    continue

                NDVI_dic = T.load_npy(fdir+f)
                f_name = fdir.split('\\')[-3]+'_'+f.split('\\')[-1].split('.')[0]
                print(f_name)
                NDVI_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in NDVI_dic:
                        NDVI_list.append(np.nan)
                        continue

                    vals = NDVI_dic[pix]
                    if len(vals) != 20:
                        NDVI_list.append(np.nan)
                        continue
                    v1 = vals[year - 2003]
                    NDVI_list.append(v1)
                df[f_name] = NDVI_list
        return df

    def add_row(self,df):
        r_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r,c=pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_max_trend_to_df(self, df):

        # fdir = results_root + '/lc_trend/'
        fdir = r'C:/Users/pcadmin/Desktop/Data/Base_data/lc_trend/'
        for f in (os.listdir(fdir)):
            # print()
            if not 'max_trend' in f:
                continue
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue

            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val * 20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list

        return df
    def add_NDVI_mask(self,df):
        f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        # f=data_root+'NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=np.float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df
    def add_GLC_landcover_data_to_df(self, df):


        f = rf'C:\Users\pcadmin\Desktop\Data\Base_data\LC_reclass2.npy'

        val_dic=T.load_npy(f)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df

    def P_PET_ratio(self, P_PET_fdir):

        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            vals[vals == 0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_reclass(self,dic):
        dic_reclass = {}
        for pix in dic:
            val = dic[pix]
            label = None
            # label = np.nan
            if val > 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Arid'
                # label = 0
            elif val > 0.2 and val < 0.5:
                label = 'Semi Arid'
                # label = 1
            elif val > 0.5 and val < 0.65:
                label = 'Semi Humid'
                # label = 2
            dic_reclass[pix] = label
        return dic_reclass


    def P_PET_reclass_2(self,dic):
        dic_reclass = {}
        for pix in dic:
            val = dic[pix]
            label = None
            # label = np.nan
            if val > 0.65:
                label = 'Humid'
                # label = 3
            elif val < 0.2:
                label = 'Dryland'
                # label = 0
            elif val > 0.2 and val < 0.5:
                label = 'Dryland'
                # label = 1
            elif val > 0.5 and val < 0.65:
                label = 'Dryland'
                # label = 2
            dic_reclass[pix] = label
        return dic_reclass
class SEM_wen:
    def __init__(self):

        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + '\Data_frame\detrend_zscore_new\\'
        self.dff = self.this_class_arr + 'detrend_zscore_new.df'
        self.outdir= result_root+'SEM3//'
        T.mkdir(self.outdir,force=True)

        pass

    def run(self):
        df, dff = self.__load_df()
        des=self.model_description_not_detrend_new()

        df_clean=self.clean_df(df)
        self.SEM_model(df_clean,des)
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df, dff
        # return df_early,dff
    def clean_df(self,df):
        df = df[df['row'] < 120]
        # df = df[df['HI_class'] == 'Humid']
        df = df[df['HI_class'] == 'Dryland']
        df = df[df['max_trend'] < 10]
        df=df[df['early_peak_MCD']>0]
        ## late <early
        # df=df[df['late_MCD']>df['early_peak_MCD']] ###
        # df=df[df['late_MCD']<0]
        # df = df[df['detrend_late_MODIS_LAI_zscore'] > 0]

        df = df[df['landcover_GLC'] != 'Crop']

        return df

    def model_description_not_detrend(self):

        desc_water_limited_SMroot = '''
                            # regressions

                            early_peak_MCD~early_Temp+early_precip

                            peak_SMroot~  early_peak_MCD+peak_precip

                            late_MCD ~ peak_SMroot + late_Temp
                            late_MCD~peak_precip
                            

                            # residual correlations
                             early_peak_MCD~~early_peak_MCD
                                peak_SMroot~~peak_SMroot
                                late_MCD~~late_MCD

                                early_peak_MCD~~early_Temp
                                early_peak_MCD~~early_precip
                                peak_SMroot~~early_peak_MCD
                                peak_SMroot~~peak_precip
                                late_MCD~~peak_SMroot
                                late_MCD~~late_Temp
                                late_MCD~~peak_precip

                            '''

        desc_energy_limited_SMroot = '''
                                    # regressions

                                    early_peak_MCD~early_Temp+early_precip

                                    peak_SMroot~  early_peak_MCD+peak_precip

                                    late_MCD ~ peak_SMroot + late_Temp
                                    late_MCD~peak_precip


                                    # residual correlations
                                     early_peak_MCD~~early_peak_MCD
                                        peak_SMroot~~peak_SMroot
                                        late_MCD~~late_MCD

                                        early_peak_MCD~~early_Temp
                                        early_peak_MCD~~early_precip
                                        peak_SMroot~~early_peak_MCD
                                        peak_SMroot~~peak_precip
                                        late_MCD~~peak_SMroot
                                        late_MCD~~late_Temp
                                        late_MCD~~peak_precip

                                    '''

        return desc_water_limited_SMroot

    def model_description_not_detrend_new(self):
        desc_all_limited_SMroot = '''
                             # regressions

                             early_peak_MCD~early_Temp+early_precip

                             late_SMroot~  early_peak_MCD+late_Temp+peak_precip

                             late_MCD ~ late_SMroot + late_Temp+late_precip
                
                            
                             # residual correlations
                              early_peak_MCD~~early_peak_MCD
                                 late_SMroot~~late_SMroot
                                 late_MCD~~late_MCD

                                 early_peak_MCD~~early_Temp
                                 early_peak_MCD~~early_precip
                                 late_SMroot~~early_peak_MCD
                                 
                                 late_MCD~~late_SMroot
                                 late_SMroot~~late_Temp
                                 late_MCD~~late_Temp
                              
                                 late_SMroot~~peak_precip
                                 
                                    late_MCD~~late_precip
                                 

                             '''






        return desc_all_limited_SMroot

    def SEM_model(self,df,desc):
        mod = semopy.Model(desc)
        res = mod.fit(df)

        result = mod.inspect()
        T.print_head_n(result)
        outf = self.outdir + f'water_limited_MCD'
        T.save_df(result, outf + '.df')
        T.df_to_excel(result, outf + '.xlsx')

        outf = self.outdir + 'water_limited_MCD'
        semopy.report(mod, outf)


class SEM_wen_ET:
    def __init__(self):
        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + '\Data_frame\detrend_zscore_new\\'
        self.dff = self.this_class_arr + 'detrend_zscore_new.df'
        self.outdir = result_root + 'SEM_ET//'
        T.mkdir(self.outdir, force=True)

        pass

    def run(self):
        df, dff = self.__load_df()
        des = self.model_description_not_detrend()

        df_clean = self.clean_df(df)
        self.SEM_model(df_clean, des)
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df, dff
        # return df_early,dff

    def clean_df(self, df):
        df = df[df['row'] < 120]
        df = df[df['HI_class'] == 'Humid']
        # df = df[df['HI_class'] == 'Dryland']
        df = df[df['max_trend'] < 10]
        df = df[df['early_peak_MCD'] > 0]
        ## late <early
        # df=df[df['late_MCD']>df['early_peak_MCD']] ###
        # df=df[df['late_MCD']<0]
        # df = df[df['detrend_late_MODIS_LAI_zscore'] > 0]

        df = df[df['landcover_GLC'] != 'Crop']

        return df

    def model_description_not_detrend(self):
        desc_water_limited_SMroot = '''
                            # regressions
  # regressions
                                   early_peak_MCD~early_Temp+early_precip
                            early_peak_Et~ early_peak_MCD
                            late_SMroot~early_peak_Et+peak_precip
                           
                    
                            # residual correlations
                             early_peak_MCD~~early_peak_MCD
                                late_SMroot~~late_SMroot
                                early_peak_Et~~early_peak_Et
                                
                                early_peak_MCD~~early_Temp
                                early_peak_MCD~~early_precip
                       
                                late_SMroot~~early_peak_Et
                                late_SMroot~~peak_precip
                     

                            '''

        desc_energy_limited_SMroot = '''
                                    # regressions

                                   early_peak_MCD~early_Temp+early_precip
                            early_peak_Et~ early_peak_MCD
                            peak_SMroot~early_peak_Et+peak_precip
                           
                    
                            # residual correlations
                             early_peak_MCD~~early_peak_MCD
                                peak_SMroot~~peak_SMroot
                                early_peak_Et~~early_peak_Et
                                
                               

                                early_peak_MCD~~early_Temp
                                early_peak_MCD~~early_precip
                       
                                peak_SMroot~~early_peak_Et
                                peak_SMroot~~peak_precip
                                
                               
                              
                            
                                    '''

        return desc_energy_limited_SMroot


    def SEM_model(self, df, desc):
        mod = semopy.Model(desc)
        res = mod.fit(df)

        result = mod.inspect()
        T.print_head_n(result)
        outf = self.outdir + f'energy_limited_MCD'
        T.save_df(result, outf + '.df')
        T.df_to_excel(result, outf + '.xlsx')

        outf = self.outdir + 'energy_limited_MCD'
        semopy.report(mod, outf)


class SEM_wen_for_individual_model:
    def __init__(self):

        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + '\Data_frame\detrend_zscore_new\\'
        self.dff = self.this_class_arr + 'detrend_zscore_new.df'
        self.outdir= result_root+'SEM3//'
        T.mkdir(self.outdir,force=True)

        pass

    def run(self):
        df, dff = self.__load_df()

        T.print_head_n(df)
        # exit()
        z_val_name_list = ['CABLE_POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5_S2_lai', 'IBIS_S2_lai',
                           'ISAM_S2_lai', 'ISBA_CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'LPX_Bern_S2_lai',
                           'VISIT_S2_lai', ]


        for variable in z_val_name_list:
            df_clean=self.clean_df(df,variable)
            lai_variable=variable
            if 'lai' in lai_variable:
                sm_variable=lai_variable.replace('lai','mrso')
            else:
                sm_variable=lai_variable.replace('LAI','mrso')

            print(lai_variable,sm_variable)

            des = self.model_description_not_detrend_Trendy(lai_variable,sm_variable)
            self.SEM_model(df_clean,des,lai_variable)



        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df, dff
        # return df_early,dff
    def clean_df(self,df,variable):
        df = df[df['row'] < 120]
        df = df[df['HI_class'] == 'Humid']
        # df = df[df['HI_class'] == 'Dryland']
        df = df[df['max_trend'] < 10]

        df = df[df[f'early_peak_{variable}'] > 0]
        # df = df[df[f'late_{variable}']<0]
        ## late early
        # df=df[df[f'late_{variable}']>df[f'early_peak_{variable}']]


        df = df[df['landcover_GLC'] != 'Crop']

        return df


    def model_description_not_detrend_Trendy(self,lai_variable,sm_variable):
        desc_energy_limited = f'''
                            # regressions


                             early_peak_{lai_variable}~early_Temp+early_precip

                             late_{sm_variable}~ early_peak_{lai_variable}+late_Temp+peak_precip

                             late_{lai_variable} ~ late_{sm_variable} + late_Temp+late_precip
                
                
                            
                             # residual correlations
                              early_peak_{lai_variable}~~early_peak_{lai_variable}
                                 late_{sm_variable}~~late_{sm_variable}
                                 late_{lai_variable}~~late_{lai_variable}

                                 early_peak_{lai_variable}~~early_Temp
                                 early_peak_{lai_variable}~~early_precip
                                 late_{sm_variable}~~early_peak_{lai_variable}
                                 
                                 late_{lai_variable}~~late_{sm_variable}
                                 late_{sm_variable}~~late_Temp
                                 late_{lai_variable}~~late_Temp
                              
                                 late_{sm_variable}~~peak_precip
                                
                                late_{lai_variable}~~late_precip
                                 

                             '''

        desc_water_limited = f'''
                                    # regressions


                                     early_peak_{lai_variable}~early_Temp+early_precip

                                     late_{sm_variable}~ early_peak_{lai_variable}+late_Temp+peak_precip

                                     late_{lai_variable} ~ late_{sm_variable} + late_Temp+late_precip




                                     # residual correlations
                                      early_peak_{lai_variable}~~early_peak_{lai_variable}
                                         late_{sm_variable}~~late_{sm_variable}
                                         late_{lai_variable}~~late_{lai_variable}

                                         early_peak_{lai_variable}~~early_Temp
                                         early_peak_{lai_variable}~~early_precip
                                         late_{sm_variable}~~early_peak_{lai_variable}

                                         late_{lai_variable}~~late_{sm_variable}
                                         late_{sm_variable}~~late_Temp
                                         late_{lai_variable}~~late_Temp

                                         late_{sm_variable}~~peak_precip
                                        
                                        late_{lai_variable}~~late_precip

                                     '''

        return desc_water_limited


    def SEM_model(self,df,desc,var_name):
        mod = semopy.Model(desc)
        res = mod.fit(df)
        result=mod.inspect()
        T.print_head_n(result)
        outf=self.outdir+f'energy_limited_{var_name}'
        T.save_df(result, outf+'.df')
        T.df_to_excel(result, outf+'.xlsx')
        # exit()
        # semopy.report(mod, f'SEM_result/{ltd}-{lc}')
        # semopy.report(mod, f'SEM_result/{ltd}')
        # outf=self.outdir+'energy_limited'
        outf = self.outdir + f'energy_limited_{var_name}'

        semopy.report(mod, outf)
class SEM_anaysis():  #### statistical SEM results for different pathways
    def __init__(self):
        self.this_class_arr = result_root + 'SEM3\\'
        T.mk_dir(self.this_class_arr, force=1)



        pass
    def run(self):
        # self.SEM_process_comparision()
        self.model_performance()

        pass


    def SEM_process_comparision(self):
        marker_list = ['o', 's', 'D', 'v', 'p', 'P', '^', 'X', 'd']
        # color_list=['g','b','y','c','m','k','orange','purple','brown']
        color_list = T.gen_colors(9)

        fdir = join(self.this_class_arr, 'df_model')
        # region = 'water_limited'
        region = 'water_limited'
        val_list = []
        result_dict = {}

        ax=plt.subplot(111)

        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            if not f.startswith(region):
                continue
            # print(f)
            model = f.split('.')[0].replace(region + '_', '').replace('_lai', '')
            path_list_l = self.path_list_left(model)
            path_list_r = self.path_list_right(model)
            dff = join(fdir, f)
            df = T.load_df(dff)
            df = df[df['op'] != '~~']

            for i in range(len(path_list_l)):
                lv = path_list_l[i]
                rv = path_list_r[i]

                df_l = df[df['lval'] == lv]
                df_r = df_l[df_l['rval'] == rv]
                print (df_r)

                if len(df_r) != 1:
                    print(len(df_r))
                    raise ValueError
                Estimate = df_r['Estimate'].values[0]
                # print(Estimate)
                lv = lv.replace(model + '_', '')
                rv = rv.replace(model + '_', '')
                key = f'{rv}--->{lv}'
                if not key in result_dict:
                    result_dict[key] = {}
                result_dict[key][model] = Estimate
                # result_dict[key] = Estimate
        df_result = T.dic_to_df(result_dict, 'path')
        print(df_result)
        T.print_head_n(df_result)
        obs_SEM_result = self.obs_SEM(region)
        path = df_result['path']
        boxes = []
        x_lable = []

        for p in path:
            flag = 0
            df_p = df_result[df_result['path'] == p]

            vals = df_p.values[0]
            vals_i_list = []
            for v in vals:
                if type(v) == float:
                    vals_i_list.append(v)

                    print(flag)

                    print(df_p.columns)
                    model_name = df_p.columns[flag+1]
                    plt.scatter(v,p, marker=marker_list[flag], color=color_list[flag], s=30,zorder=5,linewidths=0.5,edgecolors='k',label=model_name)
                    flag+=1
            boxes.append(vals_i_list)
            x_lable.append(p)
            # plt.scatter(vals_i_list, [p for i in range(len(vals_i_list))], marker=marker_list[flag], color=color_list[flag], s=50)
        # x_lable_obs = obs_SEM_result.keys()

        # x_lable_obs = list(x_lable_obs)

        vals_obs = [obs_SEM_result[x] for x in x_lable]
        plt.boxplot(boxes, labels=x_lable, showfliers=False, vert=False, positions=range(len(x_lable)),zorder=100,patch_artist=True,boxprops=dict(facecolor='none', color='k'),whiskerprops=dict(color='black'),medianprops=dict(color='black'))

        plt.scatter(vals_obs, x_lable, marker='*', color='r', s=100,zorder=10)
        ### reverse the y axis
        plt.gca().invert_yaxis()
        plt.title(region)
        # plt.legend()

        plt.tight_layout()
        plt.show()

    def obs_SEM(self, region):
        fdir = join(self.this_class_arr, 'df_obs')
        # region = 'water_limited'
        fpath = join(fdir, f'{region}_MCD.df')
        df = T.load_df(fpath)
        df = df[df['op'] == '~']
        path_list = []
        T.print_head_n(df)
        results_dict = {}
        for i, row in df.iterrows():
            lv = row['lval']
            rv = row['rval']
            # print(lv, rv)
            lv = lv.replace('MCD', 'lai')
            rv = rv.replace('MCD', 'lai')

            lv = lv.replace('SMroot', 'mrso')
            rv = rv.replace('SMroot', 'mrso')
            label = f'{rv}--->{lv}'
            # path_list.append(f'{rv}--->{lv}')
            val = row['Estimate']
            results_dict[label] = val
        return results_dict

    def path_list_left(self, model):
        path_list = [
            f'early_peak_{model}_lai',
            f'early_peak_{model}_lai',

            f'late_{model}_mrso',
            f'late_{model}_mrso',
            f'late_{model}_mrso',


            f'late_{model}_lai',
            f'late_{model}_lai',
            f'late_{model}_lai',
        ]
        return path_list

    def path_list_right(self,model):
        path_list = [
            'early_Temp',
            'early_precip',

            f'early_peak_{model}_lai',
            'peak_precip',
            'late_Temp',

            f'late_{model}_mrso',
            'late_Temp',
            'late_precip',
        ]
        return path_list

    def model_performance(self):
        fig = plt.figure(figsize=(10, 5))
        xlsx = result_root + 'SEM3\model_performance_energy_limited.xlsx'
        df = pd.read_excel(xlsx)
        T.print_head_n(df)
        attriute_list = ['RMSE', 'GFI', 'AGFI', ]
        ### read columns model name and RMSEA
        flag=1
        for attribute in attriute_list:
            ax = fig.add_subplot(1, 3, flag)

            model_list = df['Model_name']
            RMSEA_list = df[attribute]
            ## reverse the RMSEA and model name
            RMSEA_list = RMSEA_list[::-1]
            model_list = model_list[::-1]
            RMSEA_list = RMSEA_list.to_list()
            model_list = model_list.to_list()
            print(model_list)
            print(RMSEA_list)

            ## add a line to show the threshold of MODIS
            ## get MODIS RMSEA
            modis_RMSEA = df[df['Model_name'] == 'MODIS'][attribute].values[0]
            print(modis_RMSEA)
            ## plot all attribute

            ax.bar(range(len(model_list)), RMSEA_list, color='b', label=attribute)
            ax.axhline(y=modis_RMSEA, color='r', linestyle='--', label='MODIS')
            ax.set_xticks(range(len(model_list)))
            ax.set_xticklabels(model_list, rotation=45)
            ax.set_ylabel(attribute)
            ax.set_xlabel('Model')
            ax.legend()

            flag=flag+1
        plt.tight_layout()
        plt.show()






        pass





class anaysize_fluxnet():
    def __init__(self):


        pass

    def run(self):
        # self.unzip_fluxnet_data()
        # self.parse_html()
        # self.screening_fluxnet_data()
        # self.read_fluxnet_data_multiyear()
        # self.read_fluxnet_data_single_year()
        # self.hants_fluxnet_data()
        # self.annual_phenology()
        # self.average_phenology()
        # self.pick_daily_phenology()
        # self.extraction_variables_static_during_daily()
        # self.plot_check()
        # self.detrend_zscore()
        # self.composit_sites_df()
        # self.matching_aridity_long_lat()
        # self.plot_early_peak_vs_late()
        self.plot_early_peak_vs_late_2()
        pass

    def unzip_fluxnet_data(self):
        fdir = data_root + '/FLUXNET_2015\zips\\'
        outdir = data_root + rf'/FLUXNET_2015\unzips\\'
        T.mk_dir(outdir, force=True)
        T.unzip(fdir, outdir)

    def parse_html(self):
        f=data_root + '/FLUXNET_2015\List of FLUXNET 2015 Sites.html'
        df=pd.read_html(f)
        T.print_head_n(df[0],10)
        T.save_df(df[0],data_root + '/FLUXNET_2015\Metainfo.df')
        T.df_to_excel(df[0],data_root + '/FLUXNET_2015\Metainfo.xlsx')


    def screening_fluxnet_data(self):
        f_metainfo=data_root + '/FLUXNET_2015\Metainfo.df'

        ##################1. screening by lat lon

        df_metainfo=pd.read_pickle(f_metainfo)
        # T.print_head_n(df_metainfo,10)

        point_list=[]

        site_id_list=[]
        lat_loc=df_metainfo['LOCATION_LAT'].to_list()
        lon_loc = df_metainfo['LOCATION_LONG'].to_list()
        for i in range(len(lat_loc)):
            lat_loc[i]=str(lat_loc[i])
            lon_loc[i] = str(lon_loc[i])

            lat_loc[i]=float(lat_loc[i].split('.')[0])
            lon_loc[i] = float(lon_loc[i].split('.')[0])
            if lat_loc[i]<30:
                continue

            point_list.append([lon_loc[i],lat_loc[i],{'site_id':df_metainfo['SITE_ID'].to_list()[i]}])

            site_id=df_metainfo['SITE_ID'].to_list()[i]
            site_id_list.append(site_id)
        print(len(site_id_list))

        # T.point_to_shp(point_list,data_root + '/FLUXNET_2015\screening_sites.shp')

        ################ 2. screening data length
        f_availability=data_root + 'FLUXNET_2015\document\data-availability-20220214011650.csv'
        df_availability=pd.read_csv(f_availability)
        df_availability['star_num'] = np.nan

        for i, row in tqdm(df_availability.iterrows(), total=len(df_availability)):

            #calculate '*' number for each line
            star_num=0
            for col in df_availability.columns:
                if '+' in str(row[col]):
                    star_num+=1
            df_availability.loc[i,'star_num']=star_num
        # T.print_head_n(df_availability,10)
        df_availability_above5years=df_availability[df_availability['star_num']>=5]
        site_id_list_above5years=df_availability_above5years['Year/Site ID'].to_list()
        # T.print_head_n(df_availability_above5years,10)
        # exit()
        print(len(site_id_list_above5years))
        site_overlap_list=list(set(site_id_list_above5years).intersection(set(site_id_list)))
        print(len(site_overlap_list))
        # exit()

        ################ 3. get data after screening
        screening_outdir = data_root + '/FLUXNET_2015\screening_sites\\'
        T.mk_dir(screening_outdir, force=True)
        fdir_all=data_root + rf'/FLUXNET_2015\unzips\\'
        for fdir in os.listdir(fdir_all):
            fdir_site_name=fdir.split('_')[1]

            if fdir_site_name not in site_overlap_list:
                continue
            for f in os.listdir(fdir_all+fdir):
                if 'DD' not in f:
                    continue
                df=pd.read_csv(fdir_all+fdir+'/'+f)
                T.save_df(df,screening_outdir+fdir_site_name+'.df')
                T.df_to_excel(df,screening_outdir+fdir_site_name+'.xlsx')


            pass




    def read_fluxnet_data_multiyear(self):
        fdir = data_root + 'FLUXNET_2015\screening_sites\\'
        for f in os.listdir(fdir):
            print(f)
            if not f.endswith('.df'):
                continue
            df=pd.DataFrame()
            df=T.load_df(fdir+f)
            GPP_day= df['GPP_DT_VUT_REF'].to_list()
            GPP_night = df['GPP_NT_VUT_REF'].to_list()
            time=df['TIMESTAMP']
            time_list=[]
            for i in range(len(time)):

                t=str(time[i])
                year=t[0:4]
                month=t[4:6]
                day=t[6:8]
                year=int(year)
                month=int(month)
                day=int(day)
                date_obj = datetime.datetime(year, month, day)
                time_list.append(date_obj)

            plt.scatter(time_list,GPP_day)
            plt.scatter(time_list,GPP_day)
            plt.title(f)
            plt.legend(['GPP_day','GPP_night'])
            plt.show()

    def read_fluxnet_data_single_year(self):

        fdir = data_root + 'FLUXNET_2015\screening_sites\\'
        for f in os.listdir(fdir):
            print(f)
            if not f.endswith('.df'):
                continue
            df = pd.DataFrame()
            df = T.load_df(fdir + f)
            GPP_day = df['GPP_DT_VUT_REF'].to_list()
            GPP_night = df['GPP_NT_VUT_REF'].to_list()
            time = df['TIMESTAMP']
            time_list = []

            GPP_day_list = []
            GPP_night_list = []

            for i in range(len(time)):
                t = str(time[i])
                year = t[0:4]
                year = int(year)
                if year == 2007:
                    t = str(time[i])
                    year = t[0:4]
                    month = t[4:6]
                    day = t[6:8]
                    year = int(year)
                    month = int(month)
                    day = int(day)
                    date_obj = datetime.datetime(year, month, day)
                    time_list.append(date_obj)
                    GPP_day_i = GPP_day[i]
                    GPP_day_list.append(GPP_day_i)
                    GPP_night_i = GPP_night[i]
                    GPP_night_list.append(GPP_night_i)

            plt.plot(time_list, GPP_day_list)
            # plt.scatter(time_list, GPP_night_list)
            plt.title(f)
            plt.legend(['GPP_day', 'GPP_night'])
            plt.show()

    def hants_fluxnet_data(self):
        fdir=data_root + 'FLUXNET_2015\screening_sites\\'
        outdir=data_root + 'FLUXNET_2015\screening_sites_hants\\'
        T.mk_dir(outdir,force=True)
        for f in os.listdir(fdir):


            if not f.endswith('.df'):
                continue
            print(f)
            df=pd.DataFrame()

            df=T.load_df(fdir+f)
            GPP_day= df['GPP_DT_VUT_REF'].to_list()
            GPP_day=np.array(GPP_day)
            GPP_day[GPP_day< 0.5]=np.nan
            # print(GPP_day)
            # plt.scatter(range(len(GPP_day)),GPP_day)
            # plt.show()

            GPP_night = df['GPP_NT_VUT_REF'].to_list()
            time=df['TIMESTAMP']
            time_list=[]
            for i in range(len(time)):
                t=str(time[i])
                year=t[0:4]
                month=t[4:6]
                day=t[6:8]
                year=int(year)
                month=int(month)
                day=int(day)
                date_obj = datetime.datetime(year, month, day)
                time_list.append(date_obj)

            time_list=np.array(time_list)
            selected_GPP_day=[]
            selected_time_list=[]
            for i in range(len(GPP_day)):
                GPP=GPP_day[i]
                date=time_list[i]
                if np.isnan(GPP):
                    continue
                selected_GPP_day.append(GPP)
                selected_time_list.append(date)
            selected_GPP_day=np.array(selected_GPP_day)
            selected_time_list=np.array(selected_time_list)


            results = HANTS().hants_interpolate(values_list=selected_GPP_day, dates_list=selected_time_list, valid_range=[0.001, 10],
                                                    nan_value=0)
            fname=f.split('.')[0]

            np.save(outdir + f'hants365_{fname}.npy', results)

        pass

    def annual_phenology(self, threshold_i=0.2, ):
        fdir = data_root + rf'FLUXNET_2015\screening_sites_hants\\'


        out_dir = data_root + rf'\FLUXNET_2015\phenology\\annual_phenology\\'
        T.mkdir(out_dir, force=True)

        for f in T.listdir(fdir):
            fname=f.split('.')[0]

            outf_i = join(out_dir, fname)
            hants_smooth_f = join(fdir, f)
            hants_dic = T.load_npy(hants_smooth_f)
            result_dic = {}
            for year in tqdm(hants_dic,):

                vals = hants_dic[year]
                # plt.plot(vals)
                # plt.show()
                result = self.pick_phenology(vals, threshold_i)
                result_dic[year] = result
            df = T.dic_to_df(result_dic, 'year')
            T.save_df(df, outf_i+'.df')
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


    def data_clean(self, ):  # 盖帽法

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
            # save
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

    def average_phenology(self, ):  # 将多年物候期平均
        fdir = data_root + f'FLUXNET_2015\phenology/annual_phenology/'

        outdir = data_root + f'FLUXNET_2015/phenology/average_phenology/'
        T.mkdir(outdir, force=True)


        for f in T.listdir( fdir):
            fname = f.split('.')[0]
            outf = join(outdir, f'{fname}_phenology_dataframe.df')
            if not f.endswith('.df'):
                continue
            df = T.load_df(join( fdir, f))
            columns = df.columns
            column_list = []
            for col in columns:
                if col == 'year':
                    continue
                column_list.append(col)


            result={}
            for col in column_list:
                value_list = []
                for i,row in tqdm(df.iterrows(),total=len(df)):
                    value=row[col]
                    value_list.append(value)

                value_mean = np.mean(value_list)
                result[col]=value_mean

            ##dic to npy

            T.save_npy(result, outf)


    def pick_daily_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]
        fdir = data_root + f'FLUXNET_2015/phenology/average_phenology/'

        outdir = data_root + f'FLUXNET_2015/phenology/pick_daily_phenology//'
        T.mkdir(outdir, force=True)
        for f in T.listdir( fdir):
            if not f.endswith('.npy'):
                continue
            fname = f.split('.')[0].split('_')[1]
            outf = join(outdir, f'{fname}.npy')
            print(outf)

            result = T.load_npy(join( fdir, f))


            all_result_dic = {}


            early_start=result['early_start']
            early_end=result['early_end']
            late_start=result['late_start']
            late_end=result['late_end']


            early_period = np.arange(int(early_start), int(early_end), 1)
            # print(early_period)
            peak_period = np.arange(int(early_end), int(late_start), 1)
            # print(peak_period)
            late_period = np.arange(int(late_start), int(late_end), 1)
            # print(late_period)
            all_result_dic['early'] = early_period
            all_result_dic['peak'] = peak_period
            all_result_dic['late'] = late_period
            all_result_dic['early_peak'] = np.concatenate((early_period, peak_period))
            all_result_dic['early_peak_late'] = np.concatenate((early_period, peak_period, late_period))
                # print(all_result_dic[pix])

            T.save_npy(all_result_dic, outf)


    def extraction_variables_static_during_daily(self):
        fdir_hants = data_root + f'/FLUXNET_2015\screening_sites_hants/'

        fdir_phenology = data_root + f'FLUXNET_2015/phenology/pick_daily_phenology/'

            # 加载变量数据
        for f_hants in os.listdir(fdir_hants):
            dic_variables = {}
            if not f_hants.endswith('.npy'):
                continue
            dic_variables = dict(np.load(fdir_hants + f_hants, allow_pickle=True, encoding='latin1').item())
            f_temp=f_hants.split('_')[1]
            print(f_temp)
            dic_phenology= dict(np.load(fdir_phenology + f_temp, allow_pickle=True, encoding='latin1').item())


            period_list = ['early', 'peak', 'late', 'early_peak', 'early_peak_late']

            for period in period_list:
                dic_during_variables = {}


                outdir = result_root + rf'extraction_original_val/FLUXNET_2015/'

                Tools().mk_dir(outdir, True)
                picked_daily = dic_phenology[period]
                picked_daily = np.array(picked_daily, dtype=int)


                for year in tqdm(dic_variables):

                    dic_during_variables[year]=[]

                    if picked_daily[0] <= 0:
                        continue

                    time_series = dic_variables[year]
                    time_series = np.array(time_series, dtype=float)


                    picked_daily = np.array(picked_daily, dtype=int)
                    print(len(time_series))


                    during_time_series = time_series[picked_daily]

                    # print(picked_month)#!!!!!

                    during_time_series = np.array(during_time_series, dtype=float)

                    during_time_series[during_time_series < -99.] = np.nan


                    variable_mean = np.nanmean(during_time_series)  # !!! 降雨需要是sum  # 其他变量是平均值 nanmean
                    dic_during_variables[year].append(variable_mean)


                np.save(outdir + 'during_{}_{}'.format(period, f_temp), dic_during_variables)  # 修改

    def plot_check(self):

        # f='/Volumes/SSD_sumsang/project_greening/Result/detrend/extraction_during_late_growing_season_static/during_late_CSIF_par/per_pix_dic_008.npy'
        f = rf'D:\Greening\Result\extraction_original_val\FLUXNET_2015\during_early_FI-Sod.npy'
        # f = rf'D:\Greening\Data\Trendy\DIC\\CABLE-POP_S2_lai\per_pix_dic_014.npy'
        # f='/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_window/1982-2015_during_early/during_early_CO2.npy'
        result_dic = {}
        spatial_dic = {}
        # array = np.load(f)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)
        dic = dict(np.load(f, allow_pickle=True, encoding='latin1').item())
        # ///////check 字典是否不缺值////
        val_list = []
        for year in tqdm(dic, desc='interpolate'):

            val = dic[year]
            val_list.append(val)
        print(len(val_list))

        plt.plot(val_list)

        plt.show()

    def detrend_zscore(self): #
        fdir= result_root + rf'extraction_original_val\FLUXNET_2015\\'
        outdir = result_root + rf'detrend_zscore\\FLUXNET_2015\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            outf=outdir+f
            print(outf)
            # exit()

            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

            detrend_zscore_dic={}
            time_series=[]

            for year in tqdm(dic):
                detrend_zscore_dic[year]=[]
                val=dic[year]
                time_series.append(val)
            time_series=np.array(time_series)
            time_series_flatten=time_series.flatten()

            delta_time_series = []
            mean = np.nanmean(time_series_flatten)
            std=np.nanstd(time_series_flatten)

            delta_time_series = (time_series_flatten - mean) / std

            detrend_delta_time_series = signal.detrend(delta_time_series)
            print(detrend_delta_time_series)


            for i,year in enumerate(dic):
                detrend_zscore_dic[year]=detrend_delta_time_series[i]


            # plt.plot(detrend_delta_time_series)
            # plt.show()


            np.save(outf, detrend_zscore_dic)
    def composit_sites_df(self):
        fdir_early_peak = result_root + 'detrend_zscore\FLUXNET_2015\early_peak\\'
        fdir_late = result_root + 'detrend_zscore\FLUXNET_2015\late\\'
        f_metainfo = r'D:\Greening\Data\FLUXNET_2015\Metainfo.df'
        metainfo = pd.read_pickle(f_metainfo)
        T.print_head_n(metainfo)
        # exit()
        lat_list=[]
        lon_list=[]
        year_list=[]
        early_peak_list=[]
        late_list=[]
        site_name_list=[]

        for f in os.listdir(fdir_early_peak):
            site=f.split('_')[3].split('.')[0]
            print(site)


            lat=metainfo[metainfo['SITE_ID']==site]['LOCATION_LAT'].values[0]
            lon=metainfo[metainfo['SITE_ID']==site]['LOCATION_LONG'].values[0]
            print(lat,lon)


            f_early_peak_path=fdir_early_peak+f
            f_late_path=result_root+'detrend_zscore\FLUXNET_2015\late\\'+'during_late_'+site+'.npy'
            print(f_early_peak_path)
            print(f_late_path)

            dic_early_peak = dict(np.load(f_early_peak_path, allow_pickle=True, encoding='latin1').item())
            dic_late = dict(np.load(f_late_path, allow_pickle=True, encoding='latin1').item())
            for year in dic_early_peak:
                val_early_peak=dic_early_peak[year]
                val_late=dic_late[year]
                early_peak_list.append(val_early_peak)
                late_list.append(val_late)
                year_list.append(year)
                site_name_list.append(site)
                lat_list.append(lat)
                lon_list.append(lon)


        df=pd.DataFrame()
        df['Site_name']=site_name_list
        df['year'] = year_list
        df['Lat']=lat_list
        df['Lon']=lon_list
        df['early_peak']=early_peak_list
        df['late']=late_list

        # T.print_head_n(df)

        T.save_df(df, result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.df')
        T.df_to_excel(df, result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.xlsx')


        pass

    def matching_aridity_long_lat(self):  ##matching aridity index and matching long-lat to AI to composit_sites_df
        dff=result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.df'
        df=pd.read_pickle(dff)
        early_peak=df['early_peak'].values.tolist()
        late=df['late'].values.tolist()

        Aridity_tiff=rf'C:\Users\pcadmin\Desktop\Data\Base_data\\aridity_index.tif'
        Aridity_dic=DIC_and_TIF().spatial_tif_to_dic(Aridity_tiff)
        long_list=df['Lon'].values.tolist()
        lat_list=df['Lat'].values.tolist()
        pix_list=DIC_and_TIF().lon_lat_to_pix(long_list,lat_list)
        #####mathing aridity index to pix_list
        df['pix']=pix_list
        class_aridity=[]
        for i,row in df.iterrows():
            pix=row['pix']
            aridity=Aridity_dic[pix]
            df.loc[i,'aridity']=aridity
            if aridity<=0.65:
                class_aridity.append('water_limited')
            elif aridity>0.65:
                class_aridity.append('energy_limited')
            else:
                raise IOError('aridity index error')
        df['class_aridity']=class_aridity

        T.print_head_n(df)
        T.save_df(df, result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.df')
        T.df_to_excel(df, result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.xlsx')


        # np.save(outf, early_peak_vs_late_dic)

        pass
    def plot_early_peak_vs_late(self):
        df=result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.df'
        df=pd.read_pickle(df)
        # df=df[df['class_aridity']=='water_limited']
        df = df[df['class_aridity'] == 'energy_limited']
        df=df[df['early_peak']>=0]
        df=df[df['year']>=2000]

        site_list=T.get_df_unique_val_list(df,'Site_name')
        selected_site=[]
        for site in site_list:
            df_site=df[df['Site_name']==site]
            year_list=df_site['year'].values.tolist()

            if len(year_list)>6:
                selected_site.append(site)
        selected_df=df[df['Site_name'].isin(selected_site)]
        # T.print_head_n(selected_df)
        # exit()

        #####  build frequncy distribution

        threshold_early_list = [0, 0.5, 1, 1.5, 2 ]

        threshold_late_list = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        x_name_list = []
        y_name_list = []
        frequence_list = []

        for i in tqdm(range(len(threshold_early_list))):
            if i >= len(threshold_early_list) - 1:
                break

            for j in range(len(threshold_late_list)):
                if j >= len(threshold_late_list) - 1:
                    break

                early_threshold = threshold_early_list[i]
                late_threshold = threshold_late_list[j]

                df_early_peak = selected_df[selected_df['early_peak'] >= early_threshold]
                df_early_peak = df_early_peak[df_early_peak['early_peak'] < threshold_early_list[i + 1]]

                df_late = df_early_peak[df_early_peak['late'] >= late_threshold]
                df_late = df_late[df_late['late'] < threshold_late_list[j + 1]]
                y_name = str(early_threshold) + '-' + str(threshold_early_list[i + 1])
                x_name = str(late_threshold) + '-' + str(threshold_late_list[j + 1])
                x_name_list.append(x_name)
                y_name_list.append(y_name)
                freq = len(df_late) / len(df_early_peak) * 100
                frequence_list.append(freq)


        df_frequencey = pd.DataFrame()
        df_frequencey['x_name'] = x_name_list
        df_frequencey['y_name'] = y_name_list
        df_frequencey['frequence'] = frequence_list
        T.print_head_n(df_frequencey)


        #### plot heatmap

        x_name_list = df_frequencey['x_name'].values.tolist()
        y_name_list = df_frequencey['y_name'].values.tolist()
        frequence_list = df_frequencey['frequence'].values.tolist()

        x_name_list = list(set(x_name_list))
        y_name_list = list(set(y_name_list))
        x_name_list.sort()
        y_name_list.sort()


        frequence_list = np.array(frequence_list)

        frequence_list = frequence_list.reshape(len(y_name_list) , len(x_name_list) )
        sns.heatmap(frequence_list, cmap='RdBu', vmin=0, vmax=100, annot=True, fmt='.1f', xticklabels=x_name_list,)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        plt.xlabel(xlabel='late')
        plt.ylabel(ylabel='early_peak')
        plt.tight_layout()
        plt.show()

###########################################################################3
        plt.imshow(frequence_list, cmap='RdBu', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(x_name_list)), x_name_list, rotation=45, ha='right')
        threshold_early_list_label = [str(i) for i in threshold_early_list]
        threshold_early_list_label.reverse()

        plt.yticks(range(len(threshold_early_list_label)), threshold_early_list_label)
        plt.xlabel(xlabel='late')
        plt.ylabel(ylabel='early_peak')
        plt.tight_layout()
        plt.show()


        # for site in selected_site:
        #     df_site=selected_df[selected_df['Site_name']==site]
        #     year_list=df_site['year'].values.tolist()
        #     early_peak_list=df_site['early_peak'].values.tolist()
        #     late_list=df_site['late'].values.tolist()
        #     plt.scatter(early_peak_list,late_list)
        #     plt.xlim(0,1.5)
        #     plt.ylim(-1.5,1.5)


        pass

    def plot_early_peak_vs_late_2(self):  ### using the same way to plot heatmap as plot remote sensing data. all statistical anaysis is the same
        df=result_root+'detrend_zscore\FLUXNET_2015\composit_sites_df.df'

        df=pd.read_pickle(df)


        df=df[df['class_aridity']=='water_limited']
        # df = df[df['class_aridity'] == 'energy_limited']
        df=df[df['early_peak']>=0]
        df=df[df['year']>=2000]

        site_list=T.get_df_unique_val_list(df,'Site_name')
        selected_site=[]
        for site in site_list:
            df_site=df[df['Site_name']==site]
            year_list=df_site['year'].values.tolist()
            selected_site.append(site)

            # if len(year_list)>4:   ######### 由于数据量太少，所以不做限制
            #     selected_site.append(site)

        selected_df=df[df['Site_name'].isin(selected_site)]

        T.print_head_n(selected_df)
        # exit()

        #####  build frequncy distribution

        threshold_early_list = [0, 0.5, 1, 1.5, 2 ]

        threshold_late_list = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        x_name_list = []
        y_name_list = []
        frequence_list = []

        for i in tqdm(range(len(threshold_early_list))):
            if i >= len(threshold_early_list) - 1:
                break


            for j in range(len(threshold_late_list)):
                if j >= len(threshold_late_list) - 1:
                    break

                if threshold_late_list[j] >= 0:
                    factor = 1
                else:
                    factor = -1

                early_threshold = threshold_early_list[i]
                late_threshold = threshold_late_list[j]

                df_early_peak = selected_df[selected_df['early_peak'] >= early_threshold]
                df_early_peak = df_early_peak[df_early_peak['early_peak'] < threshold_early_list[i + 1]]

                df_late = df_early_peak[df_early_peak['late'] >= late_threshold]
                df_late = df_late[df_late['late'] < threshold_late_list[j + 1]]
                y_name = str(early_threshold) + '-' + str(threshold_early_list[i + 1])
                x_name = str(late_threshold) + '-' + str(threshold_late_list[j + 1])
                x_name_list.append(x_name)
                y_name_list.append(y_name)

                if len(df_early_peak)==0:
                    freq=np.nan
                    frequence_list.append(freq)
                    continue
                freq = len(df_late) / len(df_early_peak) * 100*factor
                frequence_list.append(freq)
                # print(len(df_early_peak))
                # exit()


        df_frequencey = pd.DataFrame()
        df_frequencey['x_name'] = x_name_list
        df_frequencey['y_name'] = y_name_list
        df_frequencey['frequence'] = frequence_list
        T.print_head_n(df_frequencey)


        #### plot heatmap

        x_name_list = df_frequencey['x_name'].values.tolist()
        y_name_list = df_frequencey['y_name'].values.tolist()
        frequence_list = df_frequencey['frequence'].values.tolist()

        x_name_list = list(set(x_name_list))
        y_name_list = list(set(y_name_list))
        x_name_list.sort()
        y_name_list.sort()


        frequence_list = np.array(frequence_list)

        ##################3 start to plot heatmap

        cm = 1 / 2.54

        plt.figure(figsize=(15 * cm, 7 * cm))

        threshold_late_list_reverse = threshold_late_list
        threshold_early_list_reverse = threshold_early_list[::-1]

        threshold_early_list_str = [f'{i:.5f}' for i in threshold_early_list]
        threshold_late_list_str = [f'{i:.5f}' for i in threshold_late_list]


        z_list = np.array(frequence_list)
        z_list = z_list.reshape(len(threshold_early_list_str) - 1, len(threshold_late_list_str) - 1)
        # z_list_T = z_list.T
        z_list_T = z_list
        z_list_T = z_list_T[::-1]
        label_matrix = abs(z_list_T)
        label_matrix = np.round(label_matrix, 2)
        plt.imshow(label_matrix)
        plt.colorbar()
        plt.show()
        np.save(result_root + rf'Data_frame\\Frequency\\Flux_all.npy', label_matrix)
        exit()

        ax = sns.heatmap(z_list_T, annot=label_matrix, linewidths=0.75, yticklabels=threshold_early_list_str,
                         xticklabels=threshold_late_list_str, cmap='RdBu', vmin=-30, vmax=30,
                         annot_kws={'fontsize': 8},
                         cbar_kws={'label': 'Frenquency (%)', 'ticks': [-30, -20, -10, 0,10, 20, 30]}, fmt='.1f')
        threshold_early_list_str_format = [f'{i:.2f}' for i in threshold_early_list_reverse]
        threshold_late_list_str_format = [f'{i:.2f}' for i in threshold_late_list_reverse]

        ax.set_xticklabels(threshold_late_list_str_format, rotation=45, horizontalalignment='right')

        ax.set_yticklabels(threshold_early_list_str_format, rotation=0, horizontalalignment='right')
        cbar = ax.collections[0].colorbar
        cbar.ax.set_yticklabels([30, 20, 10, 0,10, 20, 30])
        # plt.tight_layout()

        plt.title('water_limited sites')

        # plt.show()
        # plt.close()
        plt.savefig(result_root + rf'Data_frame\\Frequency\\Flux_water_limited.pdf', dpi=300, )
        plt.close()










class ResponseFunction:  # figure 5 in paper
    def __init__(self):

        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + '\Data_frame\detrend_zscore_monthly\\'
        self.dff = self.this_class_arr + 'detrend_zscore_monthly.df'
        self.outdir = result_root + 'response_function/'
        T.mkdir(self.outdir, force=True)
        pass

    def run(self):
        # self.build_df()
        df, dff = self.__load_df()

        df_clean = self.clean_df(df)

        self.plot_response_func(df_clean)
        pass
    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df, dff
        # return df_early,dff
    def clean_df(self,df):
        df = df[df['row'] < 120]
        # df = df[df['HI_class'] == 'Humid']
        # df = df[df['HI_class'] == 'Dryland']
        df = df[df['max_trend'] < 10]


        # df = df[df['late_MCD'] < 0]

        df = df[df['landcover_GLC'] != 'Crop']

        return df

    def plot_response_func(self,df):
        T.print_head_n(df, 10)
        z_val_name_list=['MCD','Trendy_ensemble','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                         'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',]

        z_val_name_list = ['MCD','Trendy_ensemble', 'CABLE_POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                           'ISAM_S2_LAI', 'ISBA_CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                           'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]


        regions = ['Humid', 'Dryland']
        cm = 1 / 2.54

        for z_val_name in z_val_name_list:

            for region in regions:
                plt.figure(figsize=(15 * cm, 7 * cm))

                df_temp = df[df['HI_class'] == region]
                # df = df[df[f'early_peak_{z_val_name}'] > 0]  ### clever!
                x_var='peak_precip'
                y_var='late_Temp'
                z_var=f'late_{z_val_name}'

                x_bins = np.arange(-1.5, 1.6, 0.5)
                y_bins = np.arange(-1.5, 1.6, 0.5)

                matrix=[]
                for i in range(len(y_bins)):
                    if i==len(y_bins)-1:
                        continue

                    y_left=y_bins[i]
                    y_right=y_bins[i+1]

                    matrix_i=[]
                    for j in range(len(x_bins)):
                        if j==len(x_bins)-1:
                            continue
                        x_left=x_bins[j]
                        x_right=x_bins[j+1]

                        df_temp_i=df_temp[df_temp[x_var]>=x_left]
                        df_temp_i=df_temp_i[df_temp_i[x_var]<x_right]
                        df_temp_i=df_temp_i[df_temp_i[y_var]>=y_left]
                        df_temp_i=df_temp_i[df_temp_i[y_var]<y_right]
                        mean=np.nanmean(df_temp_i[z_var].tolist())
                        matrix_i.append(mean)
                    matrix.append(matrix_i)
                matrix=np.array(matrix)
                matrix=matrix[::-1,:]  # reverse
                plt.imshow(matrix, cmap='RdBu', interpolation='nearest')

                plt.title(f'{region} {z_val_name}')
                plt.show()
                # plt.savefig(self.outdir + f'{region}_{z_val_name}.pdf', dpi=300)
                # plt.close()



def main():
    # nctotif().run()
    # Resample().run()
    # TIFtoDIC().run()
    # Check_plot().run()
    # Phenology().run()
    # process_LAI().run()
    # statistic_analysis().run()
    frequency_analysis().run()
    # frequency_analysis_for_two_period().run()

    # trends_seasonal_feedback().run()
    # long_term_seasonal_feedbacks_window_anaysis().run()
    # build_dataframe().run()
    # long_term_trends().run()
    # SEM_wen().run()
    # SEM_wen_ET().run()
    # SEM_wen_for_individual_model().run()
    # SEM_anaysis().run()

    # anaysize_fluxnet().run()
    # ResponseFunction().run()



    pass

if __name__ == '__main__':
    main()