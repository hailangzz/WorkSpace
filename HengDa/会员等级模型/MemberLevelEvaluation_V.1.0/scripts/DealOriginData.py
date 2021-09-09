#!/usr/bin/env bash
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import logs

class dealorigindata:
    update_time = ''

    total_broker_origin_df = pd.DataFrame()
    total_broker_origin_df_buff = pd.DataFrame()
    city_group_df = pd.DataFrame()  
    result_broker_df = pd.DataFrame()

    broker_df_list = []  
    useful_bool = pd.Series()

    def all_path(self, dirname):
        result = []  
        for maindir, subdir, file_name_list in os.walk(dirname):
            for filename in file_name_list:
                if '.csv' in filename:
                    apath = os.path.join(maindir, filename)  
                    result.append(apath)
        return result

    def deal_city(self, rowdata):

        if rowdata.res_city =='NULL':  
            if rowdata.working_city !='NULL':
                return rowdata.working_city
            else:
                if rowdata.register_city !='NULL':
                    return rowdata.register_city
                else:
                    return '城市为空'
        else:
            return rowdata.res_city

    def read_borker_data(self, all_origin_data_df):

        self.total_broker_origin_df = all_origin_data_df
        self.update_time = self.total_broker_origin_df.loc[0, 'update_time'][:10]

        self.total_broker_origin_df['city'] = self.total_broker_origin_df.apply(self.deal_city, axis=1)
        self.total_broker_origin_df.drop(
            columns=['res_city', 'register_city', 'working_city', 'is_manager', 'update_time'], inplace=True)

    def clean_broker_df(self, ):

        self.total_broker_origin_df.city.fillna('城市为空', inplace=True)
        self.total_broker_origin_df_buff = self.total_broker_origin_df.fillna(0)
        deal_partdf = self.total_broker_origin_df_buff[
            [ 'order_deal_amt_d7','order_deal_amt_d15','order_deal_amt_d30', 'order_deal_amt_d60','order_deal_amt_d90','order_deal_amt_d180','order_deal_amt_d360']]
        # deal_partdf = np.log(1+deal_partdf) # 进行对数转化，减弱极大值影响···

        data_std = deal_partdf[deal_partdf > 0].std()
        data_mean = deal_partdf[deal_partdf > 0].mean()
        lower_limit = data_mean - 3 * data_std
        upper_limit = data_mean + 3 * data_std

        # print(lower_limit,upper_limit)
        uplimit_bool_flag = deal_partdf < upper_limit
        uplimit_bool_flag_sum = uplimit_bool_flag.sum(axis=1)

        self.useful_bool = uplimit_bool_flag_sum == deal_partdf.shape[1]
        # print(self.useful_bool.sum())

    def deal_amount_std_data(self, rowdf, city_hours_price_mean, column):
        return city_hours_price_mean.loc[rowdf.city][column]

    def get_city_amount_avg(self, deal_columns=['order_deal_amt_d7','order_deal_amt_d15','order_deal_amt_d30', 'order_deal_amt_d60','order_deal_amt_d90','order_deal_amt_d180','order_deal_amt_d360']):

        a = self.total_broker_origin_df.replace(0, np.NaN)[self.useful_bool].groupby('city').mean()
        print(a.columns,len(a.columns))

        city_hours_price_mean = self.total_broker_origin_df.replace(0, np.NaN)[self.useful_bool].groupby('city')[
            deal_columns].mean().sort_values(by='order_deal_amt_d360', ascending=False)

        for column in deal_columns:
            self.total_broker_origin_df[column + '_citymean'] = self.total_broker_origin_df.apply(
                self.deal_amount_std_data, axis=1, args=(city_hours_price_mean, column))
            self.total_broker_origin_df[column] = self.total_broker_origin_df[column] / self.total_broker_origin_df[
                column + '_citymean']

    def deal_lday_feature(self,): # 对连续未使用时长进行变换处理···
        all_lday_feature = ['recommend_lday_cnt','visit_lday_cnt','order_deal_lday_cnt','app_user_lday_cnt_d30']
        for singel_lday_feature in all_lday_feature:
            self.total_broker_origin_df[singel_lday_feature]=-np.log(self.total_broker_origin_df[singel_lday_feature]+0.1)
            feature_min = self.total_broker_origin_df[singel_lday_feature].min()
            feature_max = self.total_broker_origin_df[singel_lday_feature].max()
            self.total_broker_origin_df[singel_lday_feature] = (self.total_broker_origin_df[singel_lday_feature]-feature_min)/(feature_max-feature_min)
            # print(self.total_broker_origin_df[singel_lday_feature].describe())

    def get_result_broker_data(self, ):
        # self.result_broker_df = self.total_broker_origin_df.iloc[:, :-7]
        self.result_broker_df = self.total_broker_origin_df.iloc[:, :]
        print('get_result_broker_data',self.result_broker_df.head())
        # for featrue in ['recommend_lday_cnt','visit_lday_cnt','order_deal_lday_cnt','app_user_lday_cnt_d30']:
        #     print(self.result_broker_df[featrue].describe())

    def __init__(self, all_origin_data_df):
        log_object = logs.logging()

        try:
            self.read_borker_data(all_origin_data_df)
            self.clean_broker_df()
            # self.get_city_amount_avg()
            self.deal_lday_feature()
            self.get_result_broker_data()
            log_object.write_logs('数据预处理成功！')
        except Exception as e:
            log_object.write_logs(e)

