# encoding: utf-8

import pandas as pd
import os
import logs
from datetime import datetime

os.system("kinit -kt /opt/kerberos/hiveops.keytab hiveops")

class ReadFeatureValues():

    def __init__(self):
        self.config_dict=self.get_connect_config()
        self.hive_cur=self.ConnectHive(self.config_dict)

    def get_connect_config(self,):
        config_dict={'host':'','port':25004,'password':'','user':'','database':'db_broker_ability',}
        configfile_cur=open('../conf/configfile','r')
        configdata=configfile_cur.readlines()
        for singleconfig in configdata:
            for key in config_dict:
                if key == singleconfig.split('=')[0]:
                    if key !='port':
                        config_dict[key]=singleconfig.split('=')[1].replace('\'','').strip()
                    else:
                        config_dict[key] = int(singleconfig.split('=')[1].replace('\'', '').strip())
        return config_dict

    def ConnectHive(self,config_dict):
        from impala.dbapi import connect
        log_object = logs.logging()
        try:
            connect = connect(host=config_dict['host'], port=config_dict['port'], password=config_dict['password'], user=config_dict['user'],database='default',kerberos_service_name='impala',auth_mechanism = 'GSSAPI')
            hive_cur = connect.cursor()
            print('Hive数据库连接成功！')
            log_object.write_logs('Hive数据库连接成功！')

            return hive_cur
        except Exception as e:
            log_object.write_logs(e)



    def get_broker_data(self,hive_cur,sql='select t.* from db_broker_ability.label_broker_Level_combine_data t  where t.update_time in (select max(update_time) from db_broker_ability.label_broker_Level_combine_data) and t.order_deal_cnt_d60>0 limit 10000;'):
        log_object = logs.logging()

        broker_data_columns = ['guid','res_city','register_city','working_city','is_manager',
                            'recommend_cnt_d7','recommend_cnt_d15','recommend_cnt_d30','recommend_cnt_d60','recommend_cnt_d90','recommend_cnt_d180','recommend_cnt_d360','recommend_cnt_d360_cusum','recommend_cnt_all',
                            'recommend_client_cnt_d7','recommend_client_cnt_d15','recommend_client_cnt_d30','recommend_client_cnt_d60','recommend_client_cnt_d90','recommend_client_cnt_d180','recommend_client_cnt_d360','recommend_client_cnt_d360_cusum','recommend_client_cnt_all',
                            'recommend_lday_cnt',
                            'visit_cnt_d7','visit_cnt_d15','visit_cnt_d30','visit_cnt_d60','visit_cnt_d90','visit_cnt_d180','visit_cnt_d360','visit_cnt_d360_cusum','visit_cnt_all',
                            'visit_client_cnt_d7','visit_client_cnt_d15','visit_client_cnt_d30','visit_client_cnt_d60','visit_client_cnt_d90','visit_client_cnt_d180','visit_client_cnt_d360','visit_client_cnt_d360_cusum','visit_client_cnt_all',
                            'visit_lday_cnt',
                            'order_deal_cnt_d7','order_deal_cnt_d15','order_deal_cnt_d30','order_deal_cnt_d60','order_deal_cnt_d90','order_deal_cnt_d180','order_deal_cnt_d360','order_deal_cnt_d360_cusum','order_deal_cnt_all',
                            'order_deal_lday_cnt',
                            'order_deal_amt_d7','order_deal_amt_d15','order_deal_amt_d30','order_deal_amt_d60','order_deal_amt_d90','order_deal_amt_d180','order_deal_amt_d360','order_deal_amt_d360_cusum','order_deal_amt_all',
                            'deal_client_cnt_d7','deal_client_cnt_d15','deal_client_cnt_d30','deal_client_cnt_d60','deal_client_cnt_d90','deal_client_cnt_d180','deal_client_cnt_d360','deal_client_cnt_d360_cusum','deal_client_cnt_all',
                            'app_user_len_d7','app_user_len_d15','app_user_len_d30','app_user_len_d60','app_user_len_d90','app_user_len_d180','app_user_len_d360','app_user_len_d360_cusum',
                            'app_user_cnt_d7','app_user_cnt_d15','app_user_cnt_d30','app_user_cnt_d60','app_user_cnt_d90','app_user_cnt_d180','app_user_cnt_d360','app_user_cnt_d360_cusum',
                            'app_user_hour_cnt_d7','app_user_hour_cnt_d15','app_user_hour_cnt_d30','app_user_hour_cnt_d60','app_user_hour_cnt_d90','app_user_hour_cnt_d180','app_user_hour_cnt_d360','app_user_hour_cnt_d360_cusum',
                            'app_user_day_cnt_d7','app_user_day_cnt_d15','app_user_day_cnt_d30','app_user_day_cnt_d60','app_user_day_cnt_d90','app_user_day_cnt_d180','app_user_day_cnt_d360','app_user_day_cnt_d360_cusum',
                            'app_user_len_avg_d7','app_user_len_avg_d15','app_user_len_avg_d30','app_user_len_avg_d60','app_user_len_avg_d90','app_user_len_avg_d180','app_user_len_avg_d360','app_user_len_avg_d360_cusum',
                            'app_user_cnt_avg_d7','app_user_cnt_avg_d15','app_user_cnt_avg_d30','app_user_cnt_avg_d60','app_user_cnt_avg_d90','app_user_cnt_avg_d180','app_user_cnt_avg_d360','app_user_cnt_avg_d360_cusum',
                            'app_user_hour_cnt_avg_d7','app_user_hour_cnt_avg_d15','app_user_hour_cnt_avg_d30','app_user_hour_cnt_avg_d60','app_user_hour_cnt_avg_d90','app_user_hour_cnt_avg_d180','app_user_hour_cnt_avg_d360','app_user_hour_cnt_avg_d360_cusum',
                            'app_user_lday_cnt_d360',
                            'update_time'
                           ]

        h_sql = sql
        try:
            hive_cur.execute(h_sql)
            all_data = hive_cur.fetchall()
            all_data_df=pd.DataFrame(all_data,columns=broker_data_columns)
            log_object.write_logs('读取数据记录成功!')
            hive_cur.close()
            return all_data_df

        except Exception as e:
            all_data_df = pd.read_csv(r'C:\Users\021206191\Desktop\FeatureValuesData.txt',sep=',',header=0, names=broker_data_columns)
            # all_data_df = pd.read_csv(r'/my_model/1.csv', sep='\t',header=0, names=broker_data_columns)
            log_object.write_logs('读取本地测试数据成功！')
            log_object.write_logs(e)
            return all_data_df

















