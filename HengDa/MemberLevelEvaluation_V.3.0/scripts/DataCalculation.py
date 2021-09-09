import pandas as pd

class DataCalculation():
    origin_feature_df = pd.DataFrame()
    broker_score_dict={'guid':[],'PerformancedScore':[],'CustomerDevelopmentScore':[],'LivenessScore':[],'TotalScore':[],}
    broker_score_result = pd.DataFrame()

    broker_feature_dict = {

        '业绩成交能力': {'feature_score_weight': 0.5,
                   'feature_columns':
                       {
                           'order_deal_cnt_d7':{'weight':0.24,'fullmark_time':3},
                           'order_deal_cnt_d15': {'weight': 0.24, 'fullmark_time': 6},
                           'order_deal_cnt_d30': {'weight': 0.16, 'fullmark_time': 16},
                           'order_deal_cnt_d60': {'weight': 0.1, 'fullmark_time': 10},
                           'order_deal_cnt_d90': {'weight': 0.14, 'fullmark_time': 14},
                           'order_deal_cnt_d180': {'weight': 0.06, 'fullmark_time': 12},
                           'order_deal_cnt_d360': {'weight': 0.06, 'fullmark_time': 12},
                       },

                   },

        '客户扩展能力': {'feature_score_weight': 0.35,
                   'feature_columns':
                       {
                       '推荐次数':
                           {'part_feature':{
                               'recommend_cnt_d7': {'weight': 0.18, 'fullmark_time': 36},
                               'recommend_cnt_d15': {'weight': 0.18, 'fullmark_time': 36},
                               'recommend_cnt_d30': {'weight': 0.16, 'fullmark_time': 80},
                               'recommend_cnt_d60': {'weight': 0.16, 'fullmark_time': 80},
                               'recommend_cnt_d90': {'weight': 0.14, 'fullmark_time': 70},
                               'recommend_cnt_d180': {'weight': 0.12, 'fullmark_time': 120},
                               'recommend_cnt_d360': {'weight': 0.6, 'fullmark_time': 120},
                           },
                               'part_feature_weight':0.35
                           },

                        '到访次数':
                            {'part_feature':
                                {
                                'visit_cnt_d7': {'weight': 0.22, 'fullmark_time': 36},
                                'visit_cnt_d15': {'weight': 0.2, 'fullmark_time': 40},
                                'visit_cnt_d30': {'weight': 0.18, 'fullmark_time': 60},
                                'visit_cnt_d60': {'weight': 0.14, 'fullmark_time': 70},
                                'visit_cnt_d90': {'weight': 0.12, 'fullmark_time': 60},
                                'visit_cnt_d180': {'weight': 0.08, 'fullmark_time': 100},
                                'visit_cnt_d360': {'weight': 0.06, 'fullmark_time': 150},
                            },
                                'part_feature_weight':0.65
                            }
                       }

                   },

        '平台活跃度': {'feature_score_weight': 0.15,
                  'feature_columns':
                      {
                          'app_user_cnt_d7': {'weight': 0.2, 'fullmark_time': 50},
                          'app_user_cnt_d15': {'weight': 0.18, 'fullmark_time': 60},
                          'app_user_cnt_d30': {'weight': 0.16, 'fullmark_time': 160},
                          'app_user_cnt_d60': {'weight': 0.14, 'fullmark_time': 175},
                          'app_user_cnt_d90': {'weight': 0.12, 'fullmark_time': 216},
                          'app_user_cnt_d180': {'weight': 0.12, 'fullmark_time': 400},
                          'app_user_cnt_d360': {'weight': 0.08, 'fullmark_time': 400},
                      },

                  }
    }
    # 获取得分基准数据
    ScoringCriteria_dict={'业绩成交能力':{},'客户扩展能力':{},'平台活跃度':{}}

    def __init__(self,origin_feature_df):
        self.origin_feature_df=origin_feature_df
        self.CalculateScoringCriteria()
        self.CalculateScore()

    def CalculateScoringCriteria(self):
        for broker_feature in self.broker_feature_dict:
            if broker_feature == '业绩成交能力':
                for feature_columns in self.broker_feature_dict[broker_feature]['feature_columns']:
                    ScoringCriteria = self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns][
                                          'weight'] * 100 / \
                                      self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns][
                                          'fullmark_time']
                    self.ScoringCriteria_dict[broker_feature][feature_columns]=ScoringCriteria

            if broker_feature == '平台活跃度':
                for feature_columns in self.broker_feature_dict[broker_feature]['feature_columns']:
                    ScoringCriteria = self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns][
                                          'weight'] * 100 / \
                                      self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns][
                                          'fullmark_time']
                    self.ScoringCriteria_dict[broker_feature][feature_columns] = ScoringCriteria

            if broker_feature == '客户扩展能力':
                part_ScoringCriteria = {}
                for part_feature in self.broker_feature_dict[broker_feature]['feature_columns']:
                    if part_feature not in part_ScoringCriteria:
                        part_ScoringCriteria[part_feature] = {}
                    for feature_columns in self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature']:
                        ScoringCriteria = \
                            self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature'][
                                feature_columns][
                                'weight'] * 100 / \
                            self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature'][
                                feature_columns][
                                'fullmark_time']
                        part_ScoringCriteria[part_feature][feature_columns]=ScoringCriteria

                self.ScoringCriteria_dict[broker_feature]=part_ScoringCriteria


    def CalculateScore(self,):
        featurename_and_featurescore_map={'业绩成交能力':'PerformancedScore','客户扩展能力':'CustomerDevelopmentScore','平台活跃度':'LivenessScore'}

        self.broker_score_result['guid']=self.origin_feature_df['guid']

        for broker_feature in self.broker_feature_dict:
            if broker_feature == '业绩成交能力':
                columns_list = list(self.ScoringCriteria_dict[broker_feature].keys())
                self.broker_score_result['PerformancedScore'] =(self.origin_feature_df[columns_list]*self.ScoringCriteria_dict[broker_feature]).sum(axis=1)

            if broker_feature == '平台活跃度':
                columns_list = list(self.ScoringCriteria_dict[broker_feature].keys())
                self.broker_score_result['LivenessScore'] =(self.origin_feature_df[columns_list]*self.ScoringCriteria_dict[broker_feature]).sum(axis=1)

            if broker_feature == '客户扩展能力':
                part_feature_score={}

                for part_feature in self.broker_feature_dict[broker_feature]['feature_columns']:
                    columns_list=list(self.ScoringCriteria_dict[broker_feature][part_feature].keys())
                    part_feature_score[part_feature]=(self.origin_feature_df[columns_list]*self.ScoringCriteria_dict[broker_feature][part_feature]).sum(axis=1)*self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature_weight']
                    if 'CustomerDevelopmentScore' not in self.broker_score_result:
                        self.broker_score_result['CustomerDevelopmentScore']=part_feature_score[part_feature]
                    else:
                        self.broker_score_result['CustomerDevelopmentScore'] += part_feature_score[part_feature]
                # self.broker_score_result['CustomerDevelopmentScore'] +=part_feature_score[part_feature]
            if 'TotalScore' not in self.broker_score_result:
                self.broker_score_result['TotalScore'] = self.broker_score_result[featurename_and_featurescore_map[broker_feature]]*self.broker_feature_dict[broker_feature]['feature_score_weight']
            else:
                self.broker_score_result['TotalScore'] += self.broker_score_result[featurename_and_featurescore_map[broker_feature]] * \
                                                         self.broker_feature_dict[broker_feature]['feature_score_weight']

        self.broker_score_result['update_time'] = self.origin_feature_df['update_time']
        self.broker_score_result = self.broker_score_result[['guid', 'PerformancedScore', 'CustomerDevelopmentScore', 'LivenessScore', 'TotalScore', 'update_time']]
            # for broker_feature in self.broker_feature_dict:
            #     feature_score = 0
            #     if broker_feature=='业绩成交能力':
            #         for feature_columns in self.broker_feature_dict[broker_feature]['feature_columns']:
            #             ScoringCriteria = self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns]['weight']*100/self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns]['fullmark_time']
            #             feature_score+=self.origin_feature_df.iloc[row_index][feature_columns]*ScoringCriteria
            #
            #         self.broker_score_dict['PerformancedScore'].append(feature_score)
            #
            #     if  broker_feature=='客户扩展能力':
            #         part_feature_score={}
            #         for part_feature in self.broker_feature_dict[broker_feature]['feature_columns']:
            #             if part_feature not in part_feature_score:
            #                 part_feature_score[part_feature]=0
            #             for feature_columns in self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature']:
            #                 ScoringCriteria = \
            #                 self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature'][feature_columns][
            #                     'weight'] * 100 / \
            #                 self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature'][feature_columns][
            #                     'fullmark_time']
            #                 part_feature_score[part_feature]+=self.origin_feature_df.iloc[row_index][feature_columns]*ScoringCriteria
            #             part_feature_score[part_feature]=part_feature_score[part_feature]*self.broker_feature_dict[broker_feature]['feature_columns'][part_feature]['part_feature_weight']
            #             feature_score +=part_feature_score[part_feature]
            #         self.broker_score_dict['CustomerDevelopmentScore'].append(feature_score)
            #
            #     if broker_feature=='平台活跃度':
            #         for feature_columns in self.broker_feature_dict[broker_feature]['feature_columns']:
            #             ScoringCriteria = self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns]['weight']*100/self.broker_feature_dict[broker_feature]['feature_columns'][feature_columns]['fullmark_time']
            #             feature_score+=self.origin_feature_df.iloc[row_index][feature_columns]*ScoringCriteria
            #
            #         self.broker_score_dict['LivenessScore'].append(feature_score)





