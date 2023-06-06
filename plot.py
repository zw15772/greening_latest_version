# coding=utf-8
import re

import matplotlib.pyplot as plt
# import xymap
# import xycmap
# import pymannkendall as mk
# import Main_flow_2
import semopy
import lytools
from __init__ import *
land_tif = '/Volumes/NVME2T/greening_project_redo/conf/land.tif'
result_root_this_script = '/Users/liyang/Desktop/detrend_zscore_test_factors/results'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8



class frequency_heatmap:

    def __init__(self):

        # This class is used to calculate the structural equation model
        self.this_class_arr = results_root + 'Data_frame\\Frequency\LAI3g\\'
        self.dff = self.this_class_arr + 'frequency_dataframe.df'


        pass

    def run(self):


        # self.pick_greening_year_frequency_three_classfication()
        # self.pick_greening_year_frequency_TIFF()
        # self.pick_greening_year_frequency_composite_df()
        # self.frenquency_bar()

        df, dff = self.__load_df()
        df_clean = self.clean_df(df)
        # self.pick_greening_year_frequency_heatmap()
        self.frenquency_heatmap(df_clean)

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

        df = df[df['landcover_GLC'] != 'Crop']

        return df

    def pick_greening_year_frequency_three_classfication(self):  # 通过pick years and calculate frequency only three classification

        product_list = ['MODIS_LAI','LAI3g','Trendy_ensemble','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5', 'IBIS_S2_lai',
                               'ISAM_S2_LAI',
                               'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                               'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
                               'ISBA-CTRIP_S2_lai', ]
        for product in product_list:

            f_early_peak_LAI = results_root + rf'\detrend\detrend_zscore\1982_2018\Y\detrend_early_peak_{product}_zscore.npy'
            f_late_LAI = results_root + rf'\detrend\detrend_zscore\1982_2018\Y\detrend_late_{product}_zscore.npy'
            outdir = results_root + rf'Data_frame/Frequency//ALL//'
            outf = outdir + f'{product}.df'
            T.mk_dir(outdir, force=1)
            dic_early_peak_LAI = dict(np.load(f_early_peak_LAI, allow_pickle=True, ).item())
            dic_late_LAI = dict(np.load(f_late_LAI, allow_pickle=True, ).item())



            all_result_dic = {}



            for pix in tqdm(dic_early_peak_LAI, desc=product):


                early_peak_LAI = dic_early_peak_LAI[pix]

                # print(len(early_peak_LAI))
                if not pix in dic_late_LAI:
                    continue
                late_LAI = dic_late_LAI[pix]
                classification_list = []
                father = 0
                for i in range(len(early_peak_LAI)):
                    early_peak_LAI_i = early_peak_LAI[i]
                    late_LAI_i = late_LAI[i]
                    if early_peak_LAI_i <0:
                        continue
                    if late_LAI_i > early_peak_LAI_i:
                        classification = 'amplifying'
                    elif late_LAI_i > 0:
                        classification = 'weak stabilizing'
                    else:
                        classification = 'strong stabilizing'

                    classification_list.append(classification)

                    father+=1

                if father ==0:
                    continue
                amplifying = classification_list.count('amplifying')/father*100
                weak_stabilizing = classification_list.count('weak stabilizing')/father*100
                strong_stabilizing = classification_list.count('strong stabilizing')/father*100
                dict_i={f'{product}_amplifying':amplifying,f'{product}_weak_stabilizing':weak_stabilizing,f'{product}_strong_stabilizing':strong_stabilizing}
                all_result_dic[pix] = dict_i

            df= T.dic_to_df(all_result_dic,key_col_str='pix')

            T.save_df(df, outf)
            T.df_to_excel(df, outf)

    def pick_greening_year_frequency_TIFF(self): #
        product_list = ['MODIS_LAI', 'LAI3g', 'Trendy_ensemble', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5',
                        'IBIS_S2_lai',
                        'ISAM_S2_LAI',
                        'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                        'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
                        'ISBA-CTRIP_S2_lai', ]
        fdir = results_root + rf'Data_frame/Frequency//ALL//'
        outdir = results_root + rf'Data_frame/Frequency/TIFF//'
        T.mk_dir(outdir, force=1)

        for product in product_list:
            dff=fdir+f'{product}.df'
            df = T.load_df(dff)
            column_name = {f'{product}_amplifying', f'{product}_weak_stabilizing',
                      f'{product}_strong_stabilizing'}
            for column in column_name:
                spatial_dic = T.df_to_spatial_dic(df, column)
                outtif = outdir + f'{column}.tif'
                DIC_and_TIF().pix_dic_to_tif(spatial_dic, outtif)




    def pick_greening_year_frequency_composite_df(self): #
        fdir= results_root + rf'Data_frame/Frequency//TIFF//'
        outdir = results_root + rf'Data_frame/Frequency//composite_df//'
        T.mk_dir(outdir, force=1)
        outf= outdir + f'composite_df.df'
        all_result_dic = {}
        for f in os.listdir(fdir):
            if f.endswith('.tif'):
                fpath = fdir + f
                spatial_dic = DIC_and_TIF().spatial_tif_to_dic(fpath)
                col_name= f.split('.')[0]
                all_result_dic[col_name] = spatial_dic
        df = T.spatial_dics_to_df(all_result_dic)
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def frenquency_bar(self, ):


        product_list = ['MODIS_LAI', 'LAI3g', 'Trendy_ensemble', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5',
                        'IBIS_S2_lai',
                        'ISAM_S2_LAI',
                        'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                        'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
                        'ISBA-CTRIP_S2_lai', ]
        color_list = ['#F7B36B', '#F9E29F', '#E1ECB2', ]

        color_list.extend(['lavender'] * 1000)
        color_list.append('k')


        df = T.load_df(rf'D:\Greening\Result\Data_frame\Frequency\composite_df\composite_df.df')
        T.print_head_n(df, 5)
        regions = ['Dryland', 'Humid']
        classfication_list= ['amplifying', 'weak_stabilizing', 'strong_stabilizing']

        for region in regions:


            matrix_mean_list = []
            matrix_mean_name = []

            df_temp = df[df['HI_class'] == region]

            result_dic = {}

            for product in product_list:
                result_dict_i = {}
                for classfication in classfication_list:

                    column = f'{product}_{classfication}'
                    matrix = df_temp[column].to_list()
                    matrix = np.array(matrix)

                    matrix_mean = np.nanmean(matrix, axis=0)
                    matrix_mean_list.append(matrix_mean)
                    result_dict_i[classfication] = matrix_mean
                    result_dic[product]= result_dict_i

            df = pd.DataFrame(result_dic)
            df.plot(kind='bar', stacked=False, color=color_list, edgecolor='black', linewidth=1, legend=True, )
            plt.xticks(rotation=0)
            plt.ylabel('Percentage (%)')
            plt.tight_layout()
            plt.show()


    pass


    def pick_greening_year_frequency_heatmap(self): # 通过pick years and calculate frequency

        f_early_peak_LAI = results_root + rf'\detrend\detrend_zscore\2000_2018\\detrend_early_peak_LAI3g_zscore.npy'
        f_late_LAI = results_root + rf'\detrend\detrend_zscore\2000_2018\\detrend_late_LAI3g_zscore.npy'
        outdir = results_root + rf'Data_frame/Frequency/LAI3g_test///'
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

            ax=sns.heatmap(z_list_T, annot=label_matrix, linewidths=0.75,yticklabels=threshold_early_list_str,
                           xticklabels=threshold_late_list_str,cmap='RdBu',vmin=-15,vmax=15,
                           cbar_kws={'label': 'Frenquency (%)','ticks':[-15, -10,-5, 0, 5, 10,15]},fmt='.1f')
            threshold_early_list_str_format = [f'{i:.2f}' for i in threshold_early_list_reverse]
            threshold_late_list_str_format = [f'{i:.2f}' for i in threshold_late_list_reverse]

            ax.set_xticklabels(threshold_late_list_str_format, rotation=45, horizontalalignment='right')

            ax.set_yticklabels(threshold_early_list_str_format, rotation=0, horizontalalignment='right')
            cbar=ax.collections[0].colorbar
            cbar.ax.set_yticklabels([15, 10, 5, 0, 5, 10,15])
            plt.tight_layout()

            plt.title(f'{region}')



        plt.show()

def main():
    frequency_heatmap().run()

    pass

if __name__ == '__main__':
    main()
