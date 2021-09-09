import Read_sample_data
import DataCalculation as DC
import ControlHive

ReadData = Read_sample_data.ReadFeatureValues()
FeatureValues_df = ReadData.get_broker_data(ReadData.hive_cur)
# FeatureValues_df.to_csv(r'./FeatureValuesData.txt',index=None)

datacalculation = DC.DataCalculation(FeatureValues_df)
print(datacalculation.ScoringCriteria_dict)
print(datacalculation.broker_score_result.head(),datacalculation.broker_score_result.columns)

hiveclass = ControlHive.hiveConnect(ReadData.config_dict)
hiveclass.check_table('label_broker_level_score_v3')
hiveclass.write_df('label_broker_level_score_v3', datacalculation.broker_score_result,datacalculation.broker_score_result['update_time'].unique()[0],
                   ControlHive.get_cols_type(datacalculation.broker_score_result), once_cnt=30000)
# print(FeatureValues_df.head())