#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logs
import pandas as pd
import numpy as np
import copy
from scipy import stats
import get_broker_data
from DealOriginData import dealorigindata
import ControlHive
import pickle
import os
import joblib

class evaluate_broker_data:
    broker_null_origin_df = pd.DataFrame()
    # 空值会员数据···
    broker_origin_df = pd.DataFrame()
    # 原始会员数据···
    broker_clean_df = pd.DataFrame()
    # 清洗后会员数据···
    # broker_std_df = pd.DataFrame()
    # //标准化会员数据···

    broker_data_df = {}
    # 会员能力特征数据···
    broker_data_std_df = {}
    # 标准化会员能力特征数据···
    broker_data_p_df = {}
    broker_entropy_value_df = {}
    update_time = ''
    broker_feature_dict = {

        '客户扩展能力': {'feature_score': 35,
                   'feature_columns': [
                       'recommend_cnt_d7', 'recommend_cnt_d15', 'recommend_cnt_d30', 'recommend_cnt_d60',
                       'recommend_cnt_d90', 'recommend_cnt_d180', 'recommend_cnt_d360',
                       'recommend_lday_cnt',
                       'visit_cnt_d7', 'visit_cnt_d15', 'visit_cnt_d30', 'visit_cnt_d60', 'visit_cnt_d90',
                       'visit_cnt_d180', 'visit_cnt_d360',
                       'visit_lday_cnt'
                   ]
                   },
        '业绩成交能力': {'feature_score': 50,
                   'feature_columns': [
                       'order_deal_cnt_d7', 'order_deal_cnt_d15', 'order_deal_cnt_d30', 'order_deal_cnt_d60',
                       'order_deal_cnt_d90', 'order_deal_cnt_d180', 'order_deal_cnt_d360',
                       'order_deal_lday_cnt',
                       'order_deal_amt_d7', 'order_deal_amt_d15', 'order_deal_amt_d30', 'order_deal_amt_d60',
                       'order_deal_amt_d90', 'order_deal_amt_d180', 'order_deal_amt_d360',
                   ]
                   },
        '平台活跃度': {'feature_score': 15,
                  'feature_columns': [
                      'app_user_cnt_d7', 'app_user_cnt_d15', 'app_user_cnt_d30', 'app_user_cnt_d60', 'app_user_cnt_d90',
                      'app_user_cnt_d180', 'app_user_cnt_d360',
                      'app_user_day_cnt_d7', 'app_user_day_cnt_d15', 'app_user_day_cnt_d30', 'app_user_day_cnt_d60',
                      'app_user_day_cnt_d90', 'app_user_day_cnt_d180', 'app_user_day_cnt_d360',
                      'app_user_lday_cnt_d30'
                  ]
                  }
    }

    target_coeff_dict = {'target_list': [], 'target_coeff': {}}

    broker_data_father_path = 'F:\工作文档\会员能力评价项目\\'
    single_parameter_dict = {
        'standard_par': {'mean_x': pd.Series(), 'std_x': pd.Series(), 'max_x': pd.Series(), 'min_x': pd.Series()},
        'shape_n': 10000, 'entropy': pd.Series(), 'entropy_weight': pd.Series(),
        'entropy_value_norm_test': {'statistic': 0, 'pvalue': 0}
        }
    entropy_parameter_dict = {}

    broker_cluster_info_dict = {}  # 会员分群信息字典
    broker_level_quantile_dict = {'S': {'tantile': 0.043, 'value': 0},
                                  'A': {'tantile': 0.045, 'value': 0},
                                  'B': {'tantile': 0.036, 'value': 0},
                                  'C': {'tantile': 0.266, 'value': 0},
                                  'D': {'tantile': 0.620, 'value': 0}}

    def __init__(self, origin_broker_data, update_time):
        self.get_broker_origin_df(origin_broker_data)
        self.update_time = update_time

    def get_broker_origin_df(self, origin_broker_data):
        self.broker_origin_df = origin_broker_data
        self.broker_origin_df = self.broker_origin_df.iloc[:, :-1]
        self.broker_origin_df.set_index('guid', inplace=True)

    def deal_null_value(self):
        self.broker_null_origin_df = pd.DataFrame(
            self.broker_origin_df[self.broker_origin_df.iloc[:, 1:].isnull().all(axis=1)].index)
        self.broker_origin_df = self.broker_origin_df.dropna(axis=0, how='all')
        self.broker_origin_df.fillna(0, inplace=True)

    def deal_abnormal_value(self, std_num=3):
        data_std = self.broker_origin_df[self.broker_origin_df > 0].std()
        data_mean = self.broker_origin_df[self.broker_origin_df > 0].mean()
        lower_limit = data_mean - std_num * data_std
        upper_limit = data_mean + std_num * data_std
        upper_limit.fillna(0, inplace=True)

        print(self.broker_origin_df.max())
        print('upper_limit',upper_limit)
        uplimit_bool_flag = self.broker_origin_df <= upper_limit
        uplimit_bool_flag_sum = uplimit_bool_flag.sum(axis=1)
        useful_bool_flag = uplimit_bool_flag_sum == self.broker_origin_df.shape[1]
        self.broker_clean_df = self.broker_origin_df[useful_bool_flag]

        print(self.broker_clean_df.shape)

    def standard_data(self):
        for feature in self.broker_feature_dict:
            if feature not in self.broker_data_df:

                self.broker_data_df[feature] = self.broker_origin_df[
                    self.broker_feature_dict[feature]['feature_columns']]
                self.entropy_parameter_dict[feature] = copy.deepcopy(self.single_parameter_dict)

                self.entropy_parameter_dict[feature]['shape_n'] = \
                    self.broker_origin_df[self.broker_feature_dict[feature]['feature_columns']].shape[0]
                self.entropy_parameter_dict[feature]['standard_par']['mean_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].mean()
                self.entropy_parameter_dict[feature]['standard_par']['std_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].std()
                self.entropy_parameter_dict[feature]['standard_par']['max_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].max()
                self.entropy_parameter_dict[feature]['standard_par']['min_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].min()


            else:
                self.broker_data_df[feature] = self.broker_origin_df[
                    self.broker_feature_dict[feature]['feature_columns']]
                self.entropy_parameter_dict[feature] = copy.deepcopy(self.single_parameter_dict)

                self.entropy_parameter_dict[feature]['shape_n'] = \
                    self.broker_origin_df[self.broker_feature_dict[feature]['feature_columns']].shape[0]
                self.entropy_parameter_dict[feature]['standard_par']['mean_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].mean()
                self.entropy_parameter_dict[feature]['standard_par']['std_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].std()
                self.entropy_parameter_dict[feature]['standard_par']['max_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].max()
                self.entropy_parameter_dict[feature]['standard_par']['min_x'] = self.broker_clean_df[
                    self.broker_feature_dict[feature]['feature_columns']].min()

    def get_entropy_parameter(self):
        log_object = logs.logging()
        try:
            for feature in self.broker_data_df:
                self.broker_data_std_df[feature] = (self.broker_data_df[feature] -
                                                    self.entropy_parameter_dict[feature]['standard_par']['min_x']) / \
                                                   (self.entropy_parameter_dict[feature]['standard_par']['max_x'] -
                                                    self.entropy_parameter_dict[feature]['standard_par']['min_x'])
                print('max:',self.broker_data_std_df[feature].max())
                self.broker_data_p_df[feature] = self.broker_data_std_df[feature] / (
                    np.sum(self.broker_data_std_df[feature], axis=0))
                self.broker_data_p_df[feature] = self.broker_data_p_df[feature] * 1.0
                try:
                    self.broker_data_p_df[feature][np.where(self.broker_data_p_df[feature] == 0)] = 0.00000000000001
                except:
                    pass
                self.entropy_parameter_dict[feature]['entropy'] = (-1.0 / np.log(
                    self.entropy_parameter_dict[feature]['shape_n'])) * np.sum(
                    self.broker_data_p_df[feature] * np.log(self.broker_data_p_df[feature]), axis=0)

                self.entropy_parameter_dict[feature]['entropy_weight'] = (1 - self.entropy_parameter_dict[feature][
                    'entropy']) / np.sum(1 - self.entropy_parameter_dict[feature]['entropy'])

                print(feature,self.entropy_parameter_dict[feature]['entropy_weight'])
                self.broker_entropy_value_df[feature] = np.sum(
                    100*self.broker_data_std_df[feature] * self.entropy_parameter_dict[feature]['entropy_weight'], axis=1)
                print(self.broker_entropy_value_df[feature].describe())
            log_object.write_logs('熵权计算成功！')
        except Exception as e:
            log_object.write_logs(e)

    def get_target_coeff(self):
        get_coeff_df = pd.DataFrame()
        for feature in self.broker_entropy_value_df:
            self.broker_entropy_value_df[feature] = pd.DataFrame(self.broker_entropy_value_df[feature],
                                                                 columns=['entropy_value'])

        for target_index in range(len(self.broker_entropy_value_df)):

            self.target_coeff_dict['target_list'].append(list(self.broker_entropy_value_df.keys())[target_index])
            self.target_coeff_dict['target_coeff'][list(self.broker_entropy_value_df.keys())[target_index]] = 0
            if target_index == 0:
                get_coeff_df = copy.deepcopy(pd.DataFrame(
                    self.broker_entropy_value_df[list(self.broker_entropy_value_df.keys())[target_index]][
                        'entropy_value']))

                get_coeff_df.columns = self.target_coeff_dict['target_list']
            elif target_index > 0:

                get_coeff_df = pd.merge(get_coeff_df, pd.DataFrame(
                    self.broker_entropy_value_df[list(self.broker_entropy_value_df.keys())[target_index]][
                        'entropy_value']), left_index=True, right_index=True)
                get_coeff_df.columns = self.target_coeff_dict['target_list']

        get_coeff_df = pd.merge(get_coeff_df, pd.DataFrame(self.broker_origin_df.order_deal_cnt_d360), left_index=True,
                                right_index=True).corr()
        print(get_coeff_df.keys())
        for index in get_coeff_df['order_deal_cnt_d360'].iloc[:-1].index:
            self.target_coeff_dict['target_coeff'][index] = get_coeff_df['order_deal_cnt_d360'].iloc[:-1][index] / \
                                                            get_coeff_df[
                                                                'order_deal_cnt_d360'].iloc[
                                                            :-1].sum() * 100
        print(self.target_coeff_dict['target_coeff'])

    def get_broker_score(self, method='dense'):
        for feature in self.broker_entropy_value_df:

            self.entropy_parameter_dict[feature]['entropy_value_norm_test']['statistic'], \
            self.entropy_parameter_dict[feature]['entropy_value_norm_test']['pvalue'] = stats.kstest(
                self.broker_entropy_value_df[feature]['entropy_value'], 'norm', (
                    self.broker_entropy_value_df[feature]['entropy_value'].mean(),
                    self.broker_entropy_value_df[feature]['entropy_value'].std()))

            if self.entropy_parameter_dict[feature]['entropy_value_norm_test']['pvalue'] < 0.05:

                self.broker_entropy_value_df[feature]['rank'] = self.broker_entropy_value_df[feature][
                    'entropy_value'].rank(ascending=True, method=method)

                if method == 'first':
                    self.broker_entropy_value_df[feature]['score'] = self.target_coeff_dict['target_coeff'][feature] * (
                            self.broker_entropy_value_df[feature]['rank'] / self.entropy_parameter_dict[feature][
                        'shape_n']).round(10)
                    self.broker_entropy_value_df[feature]['single_ability_score'] = 100 * (
                            (self.broker_entropy_value_df[feature]['rank'] - self.broker_entropy_value_df[feature][
                                'rank'].min()) / (self.broker_entropy_value_df[feature]['rank'].max() -
                                                  self.broker_entropy_value_df[feature]['rank'].min())).round(10)
                elif method == 'dense':

                    self.broker_entropy_value_df[feature]['score'] = self.target_coeff_dict['target_coeff'][feature] * (
                            (self.broker_entropy_value_df[feature]['rank'] - self.broker_entropy_value_df[feature][
                                'rank'].min()) / (self.broker_entropy_value_df[feature]['rank'].max() -
                                                  self.broker_entropy_value_df[feature]['rank'].min())).round(10)

                    self.broker_entropy_value_df[feature]['single_ability_score'] = 100 * (
                            (self.broker_entropy_value_df[feature]['rank'] - self.broker_entropy_value_df[feature][
                                'rank'].min()) / (self.broker_entropy_value_df[feature]['rank'].max() -
                                                  self.broker_entropy_value_df[feature]['rank'].min())).round(10)

                self.entropy_parameter_dict[feature]['standard_par']['rank_min']=self.broker_entropy_value_df[feature]['rank'].min()
                self.entropy_parameter_dict[feature]['standard_par']['rank_max'] = self.broker_entropy_value_df[feature]['rank'].max()

            else:

                # self.broker_entropy_value_df[feature]['rank'] = self.standard(self.broker_entropy_value_df[feature])
                #
                # self.broker_entropy_value_df[feature]['score'] = self.target_coeff_dict['target_coeff'][feature] * \
                #                                                  self.broker_entropy_value_df[feature]['rank'].round(10)
                self.broker_entropy_value_df[feature]['rank'] = stats.norm.cdf(
                    self.broker_entropy_value_df[feature]['entropy_value'])
                self.broker_entropy_value_df[feature]['score'] = self.target_coeff_dict['target_coeff'][feature] * \
                                                                 self.broker_entropy_value_df[feature]['rank'].round(10)

                self.broker_entropy_value_df[feature]['single_ability_score'] = 100 * \
                                                                                self.broker_entropy_value_df[feature][
                                                                                    'rank'].round(10)

    def flex_broker_score(self):
        for feature in self.broker_entropy_value_df:
            self.broker_entropy_value_df[feature]['score_flex'] = (
                    (self.target_coeff_dict['target_coeff'][feature] * 0.1) + self.broker_entropy_value_df[feature][
                'score'] * (self.target_coeff_dict['target_coeff'][feature] * 0.95 -
                            self.target_coeff_dict['target_coeff'][feature] * 0.1) /
                    self.target_coeff_dict['target_coeff'][feature]).round(10)

            self.broker_entropy_value_df[feature]['single_ability_score_flex'] = (
                    (100 * 0.1) + self.broker_entropy_value_df[feature]['single_ability_score'] * (
                    100 * 0.95 - 100 * 0.1) / 100).round(10)
            print(self.broker_entropy_value_df[feature]['score_flex'].describe())

            print(self.broker_entropy_value_df[feature]['single_ability_score_flex'].describe())

    def total_broker_score(self):
        feature_list = []
        for broker_feature_index in range(len(self.broker_entropy_value_df.keys())):
            if broker_feature_index == 0:
                feature_list.append(list(self.broker_entropy_value_df.keys())[broker_feature_index])
                self.broker_entropy_value_df['总分'] = pd.DataFrame(
                    self.broker_entropy_value_df[list(self.broker_entropy_value_df.keys())[broker_feature_index]][
                        'score_flex'])

            else:
                feature_list.append(list(self.broker_entropy_value_df.keys())[broker_feature_index])
                self.broker_entropy_value_df['总分'] = pd.DataFrame(self.broker_entropy_value_df['总分'].merge(pd.DataFrame(
                    self.broker_entropy_value_df[list(self.broker_entropy_value_df.keys())[broker_feature_index]][
                        'score_flex']), left_index=True, right_index=True))

        self.broker_entropy_value_df['总分'].columns = feature_list
        self.broker_entropy_value_df['总分']['total_score'] = self.broker_entropy_value_df['总分'].sum(axis=1)

        self.broker_entropy_value_df['总分']['客户扩展能力_雷达分值'] = self.broker_entropy_value_df['客户扩展能力'][
            'single_ability_score_flex']
        self.broker_entropy_value_df['总分']['业绩成交能力_雷达分值'] = self.broker_entropy_value_df['业绩成交能力'][
            'single_ability_score_flex']
        self.broker_entropy_value_df['总分']['平台活跃度_雷达分值'] = self.broker_entropy_value_df['平台活跃度'][
            'single_ability_score_flex']
        self.broker_entropy_value_df['总分']['total_score_radar'] = self.broker_entropy_value_df['客户扩展能力'][
                                                                      'single_ability_score_flex'] * 0.25 + \
                                                                  self.broker_entropy_value_df['业绩成交能力'][
                                                                      'single_ability_score_flex'] * 0.70 + \
                                                                  self.broker_entropy_value_df['平台活跃度'][
                                                                      'single_ability_score_flex'] * 0.05
        # print('total_broker_score',self.broker_entropy_value_df['总分'])

        # 获取分位点数据，数据信息···
        accumulate_tantile = 0
        self.broker_level_quantile_dict['S']['value']=self.broker_entropy_value_df['总分']['total_score_radar'].quantile(1 - self.broker_level_quantile_dict['S']['tantile'])
        self.broker_level_quantile_dict['A']['value'] = self.broker_entropy_value_df['总分'][
            'total_score_radar'].quantile(1 - (self.broker_level_quantile_dict['S']['tantile']+self.broker_level_quantile_dict['A']['tantile']))
        self.broker_level_quantile_dict['B']['value'] = self.broker_entropy_value_df['总分'][
            'total_score_radar'].quantile(
            1 - (self.broker_level_quantile_dict['S']['tantile'] + self.broker_level_quantile_dict['A']['tantile']+ self.broker_level_quantile_dict['B']['tantile']))
        self.broker_level_quantile_dict['C']['value'] = self.broker_entropy_value_df['总分'][
            'total_score_radar'].quantile(
            1 - (self.broker_level_quantile_dict['S']['tantile'] + self.broker_level_quantile_dict['A']['tantile']+ self.broker_level_quantile_dict['B']['tantile']+ self.broker_level_quantile_dict['C']['tantile']))
        self.broker_level_quantile_dict['D']['value'] = self.broker_entropy_value_df['总分'][
            'total_score_radar'].quantile(
            1 - (self.broker_level_quantile_dict['S']['tantile'] + self.broker_level_quantile_dict['A']['tantile']+ self.broker_level_quantile_dict['B']['tantile']+ self.broker_level_quantile_dict['C']['tantile']+ self.broker_level_quantile_dict['D']['tantile']))


        # for level_key in self.broker_level_quantile_dict:
        #     accumulate_tantile += self.broker_level_quantile_dict[level_key]['tantile']
        #     self.broker_level_quantile_dict[level_key]['value'] = self.broker_entropy_value_df['总分'][
        #         'total_score_radar'].quantile(1 - accumulate_tantile)
        # print('total_broker_score',self.broker_entropy_value_df['总分'])

    def part_zero_unzero_broker_cluster(self,):
        # 将总分分为全为0和非全为0两部分····
        score_part_unzero = copy.deepcopy(self.broker_entropy_value_df['总分'][
            (self.broker_entropy_value_df['总分']['CustomerDevelopmentScoreRadar'] != 10) | (
                        self.broker_entropy_value_df['总分']['PerformancedScoreRadar'] != 10) | (
                        self.broker_entropy_value_df['总分']['LivenessScoreRadar'] != 10)])

        score_part_zero = copy.deepcopy(self.broker_entropy_value_df['总分'][
            (self.broker_entropy_value_df['总分']['CustomerDevelopmentScoreRadar'] == 10) & (
                    self.broker_entropy_value_df['总分']['PerformancedScoreRadar'] == 10) & (
                    self.broker_entropy_value_df['总分']['LivenessScoreRadar'] == 10)])

        # 特征全为0的会员分为固定‘D’级别
        score_part_zero['BrokerLevelFlag'] = 'D'
        #非0值会员进行分群···
        score_part_unzero['BrokerLevelFlag'] = self.sklearn_broker_cluster(
            copy.deepcopy(score_part_unzero), broker_cluster_flag=['S', 'A', 'B', 'C'],
            cluster_model='Kmeans')  # 实现程序自动分群···

        combin_result = pd.concat([score_part_unzero,score_part_zero])
        return combin_result


    def null_broker_score(self, update_time):
        def replace_broker_level(row):
            if row['totalScoreRadar'] >= self.broker_level_quantile_dict['S']['value']:
                return 'S'
            elif row['totalScoreRadar'] < self.broker_level_quantile_dict['S']['value'] and row['totalScoreRadar'] >= \
                    self.broker_level_quantile_dict['A']['value']:
                return 'A'
            elif row['totalScoreRadar'] < self.broker_level_quantile_dict['A']['value'] and row['totalScoreRadar'] >= \
                    self.broker_level_quantile_dict['B']['value']:
                return 'B'
            elif row['totalScoreRadar'] < self.broker_level_quantile_dict['B']['value'] and row['totalScoreRadar'] >= \
                    self.broker_level_quantile_dict['C']['value']:
                return 'C'
            else:
                return 'D'

        for broker_feature in self.broker_entropy_value_df['总分']:

            if broker_feature != 'total_score_radar':
                self.broker_entropy_value_df['总分'][broker_feature] = self.broker_entropy_value_df['总分'][
                    broker_feature].round(2)
            self.broker_null_origin_df[broker_feature] = self.broker_entropy_value_df['总分'][broker_feature].min()
        self.broker_null_origin_df.set_index('guid', drop=True, inplace=True)

        try:
            self.broker_entropy_value_df['总分'] = pd.concat(
                [self.broker_entropy_value_df['总分'], self.broker_null_origin_df]).sort_values(by='total_score_radar',
                                                                                              ascending=False)
        except:
            pass
        # self.broker_entropy_value_df['总分']['total_score_radar'] = self.broker_entropy_value_df['总分'][
        #     'total_score_radar'].round(2)
        self.broker_entropy_value_df['总分'] = self.broker_entropy_value_df['总分'].fillna(0)

        self.broker_entropy_value_df['总分'] = self.broker_entropy_value_df['总分'].astype(int)
        guid = self.broker_entropy_value_df['总分'].index.astype(str)
        self.broker_entropy_value_df['总分'].insert(0, 'guid', guid)
        print(self.broker_entropy_value_df['总分'].columns)

        self.broker_entropy_value_df['总分'].columns = ['guid', 'CustomerDevelopmentScore', 'PerformancedScore',
                                                      'LivenessScore', 'totalScore', 'CustomerDevelopmentScoreRadar',
                                                      'PerformancedScoreRadar', 'LivenessScoreRadar', 'totalScoreRadar']
        self.broker_entropy_value_df['总分']['BrokerLevelFlag'] = self.broker_entropy_value_df['总分'].apply(
            replace_broker_level, axis=1)

        print(self.broker_entropy_value_df['总分'].groupby('BrokerLevelFlag').count())
        print(self.broker_level_quantile_dict)
        # 使用copy.deepcopy(self.broker_entropy_value_df['总分']) 防止前后数据干扰···
        # self.broker_entropy_value_df['总分']['BrokerLevelFlag'] = self.sklearn_broker_cluster(
        #     copy.deepcopy(self.broker_entropy_value_df['总分']), cluster_model='Kmeans')  # 实现程序自动分群···
        # self.broker_entropy_value_df['总分']['update_time'] = update_time  # 增加日期信息···
        print(self.broker_entropy_value_df['总分'].head())
        self.broker_entropy_value_df['总分'].to_csv(r'.//broker_level_score.csv', index=None)
        log_object = logs.logging()
        log_object.write_logs('存储结果数据至本地！')

    def save_pickle_entropy_info(self, filename='feature_weight', save_path='//home//bigdata_ai//MemberLevelEvaluation_V.1.0//scripts//'):
        save_path ='./'
        log_object = logs.logging()
        with open(save_path + filename + '.pickle', 'wb') as f:
            pickle.dump(self.entropy_parameter_dict, f)
        log_object = logs.logging()
        log_object.write_logs('特征权值及统计信息存储完成！')

    def print_feature_weight(self, ):
        for feature in self.entropy_parameter_dict:
            print(feature, self.entropy_parameter_dict[feature]['entropy_weight'])

    def sklearn_broker_cluster(self, broker_level_score_df, cluster_model='Kmeans'):
        log_object = logs.logging()
        score_columns = ['CustomerDevelopmentScoreRadar', 'PerformancedScoreRadar', 'LivenessScoreRadar']
        cluster_origin_data = broker_level_score_df[score_columns]
        if cluster_model == 'Kmeans':
            broker_cluster_flag = ['S', 'A', 'B', 'C', 'D']  # 会员等级分群数量···
            from sklearn.cluster import KMeans, MiniBatchKMeans
            # 保存Kmears模型

            if 'BrokerKmeansModel.pkl' not in os.listdir('//home//bigdata_ai//MemberLevelEvaluation_V.1.0//scripts//'): #如果模型不存在时···
                # 训练模型
                KM_model = KMeans(n_clusters=len(broker_cluster_flag), init='k-means++', n_init=10, random_state=28)
                cluster_pred = KM_model.fit_predict(cluster_origin_data)
                joblib.dump(KM_model, '//home//bigdata_ai//MemberLevelEvaluation_V.1.0//scripts//BrokerKmeansModel.pkl')  # 保存模型···
                log_object.write_logs('聚类模型训练完成！')
            else:
                KM_model = joblib.load('//home//bigdata_ai//MemberLevelEvaluation_V.1.0//scripts//BrokerKmeansModel.pkl')
                cluster_pred = KM_model.predict(cluster_origin_data)
                log_object.write_logs('成功加载，预先训练聚类模型！')
            print(cluster_pred,type(cluster_pred))
            cluster_center_matix = KM_model.cluster_centers_
            print('cluster_center_matix',cluster_center_matix)
            # print(cluster_center_matix[:,1])
            # cluster_center_sort_index = np.argsort(-np.sum(cluster_center_matix ** 2, axis=1))
            # 利用成交、推荐等规则排序，来排名；
            cluster_center_df = pd.DataFrame(cluster_center_matix,columns=score_columns)
            cluster_center_sort = cluster_center_df.sort_values(by=["PerformancedScoreRadar", "CustomerDevelopmentScoreRadar"], ascending=[False, False])
            cluster_center_sort_index=list(cluster_center_sort.index)
            print(cluster_center_sort_index)
            for cluster_flage_index in range(len(cluster_center_sort_index)):
                print()
                self.broker_cluster_info_dict[cluster_center_sort_index[cluster_flage_index]] = {
                    'center_info': cluster_center_matix[cluster_center_sort_index[cluster_flage_index]],
                    'cluster_map': broker_cluster_flag[cluster_flage_index]}

            # print(KM_model.cluster_centers_, type(KM_model.cluster_centers_))
            broker_level_score_df['cluster_flag'] = cluster_pred  # 会员聚类标志
            broker_level_score_df['BrokerLevelFlag'] = [self.broker_cluster_info_dict[cluster_value]['cluster_map'] for
                                                        cluster_value in cluster_pred]  # 会员聚类标志
            # broker_level_score_df.to_csv('sklearn_broker_cluster.csv', index=None)
            broker_level_score_df.groupby('BrokerLevelFlag').describe().to_excel(
                r'//home//bigdata_ai//MemberLevelEvaluation_V.1.0//scripts//sklearn_broker_cluster_describe_info.xlsx')  # 会员聚类分群数据分布特征···

            return [self.broker_cluster_info_dict[cluster_value]['cluster_map'] for cluster_value in
                    cluster_pred]  # 返回对应的分类特征字符数组S、A、B、C、D


if __name__ == '__main__':
    try:
        config_dict = get_broker_data.get_connect_config()
        #print config_dict
        hive_cur = get_broker_data.ConnectHive(config_dict)
        #print hive_cur
        get_broker_data.create_table(hive_cur)
    except:
        pass

    print(hive_cur)
    all_origin_data = get_broker_data.get_broker_data(hive_cur)
    date_info_flag = get_broker_data.check_datetime(all_origin_data) # 判断当天的数据日期特征

    # 类对象···
    deal_origin_broker_data = dealorigindata(all_origin_data)
    del all_origin_data
    # 类对象··

    broker_origin_df = evaluate_broker_data(deal_origin_broker_data.result_broker_df,
                                            deal_origin_broker_data.update_time)
    del deal_origin_broker_data
    # 删除类对象···
    broker_origin_df.deal_null_value()
    broker_origin_df.deal_abnormal_value(std_num=3)
    broker_origin_df.standard_data()
    delattr(broker_origin_df, 'broker_clean_df')
    broker_origin_df.get_entropy_parameter()
    # delattr(broker_origin_df, 'broker_data_df')
    broker_origin_df.get_target_coeff()
    delattr(broker_origin_df, 'broker_origin_df')

    # delattr(broker_origin_df, 'broker_data_std_df')
    # delattr(broker_origin_df, 'broker_data_p_df')
    broker_origin_df.get_broker_score()

    broker_origin_df.save_pickle_entropy_info()
    # print(broker_origin_df.entropy_parameter_dict)

    broker_origin_df.flex_broker_score()
    broker_origin_df.total_broker_score()
    broker_origin_df.null_broker_score(broker_origin_df.update_time)
    hiveclass = ControlHive.hiveConnect(config_dict)

    if date_info_flag['is_month_beginning']==True:
        hiveclass.check_table('label_broker_level_score_test')
        hiveclass.write_df('label_broker_level_score_test', broker_origin_df.broker_entropy_value_df['总分'],
                           ControlHive.get_cols_type(broker_origin_df.broker_entropy_value_df['总分']), once_cnt=30000)

    if date_info_flag['is_week_beginning']==True:
        hiveclass.check_table('label_broker_level_score_week')
        hiveclass.write_df('label_broker_level_score_week', broker_origin_df.broker_entropy_value_df['总分'],
                           ControlHive.get_cols_type(broker_origin_df.broker_entropy_value_df['总分']), is_week_beginning=True,once_cnt=30000)
