# _*_coding:utf-8_*_
#from krbcontext import krbcontext

import pandas as pd
import os
import logs
from datetime import datetime

os.system("kinit -kt /opt/kerberos/admin.keytab admin")

def ConnectHive(config_dict):
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


def get_connect_config():
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

def get_broker_data(hive_cur,sql='select t.* from db_broker_ability.label_broker_Level_combine_data_test t  where t.update_time in (select max(update_time) from db_broker_ability.label_broker_Level_combine_data_test);'):
    log_object = logs.logging()

    broker_data_columns = ['guid','res_city','register_city','working_city','is_manager',
                            'recommend_cnt_d7','recommend_cnt_d15','recommend_cnt_d30','recommend_cnt_d60','recommend_cnt_d90','recommend_cnt_d180','recommend_cnt_d360',
                            'recommend_client_cnt_d7','recommend_client_cnt_d15','recommend_client_cnt_d30','recommend_client_cnt_d60','recommend_client_cnt_d90','recommend_client_cnt_d180','recommend_client_cnt_d360',
                            'recommend_lday_cnt',
                            'visit_cnt_d7','visit_cnt_d15','visit_cnt_d30','visit_cnt_d60','visit_cnt_d90','visit_cnt_d180','visit_cnt_d360',
                            'visit_client_cnt_d7','visit_client_cnt_d15','visit_client_cnt_d30','visit_client_cnt_d60','visit_client_cnt_d90','visit_client_cnt_d180','visit_client_cnt_d360',
                            'visit_lday_cnt',
                            'order_deal_cnt_d7','order_deal_cnt_d15','order_deal_cnt_d30','order_deal_cnt_d60','order_deal_cnt_d90','order_deal_cnt_d180','order_deal_cnt_d360',
                            'order_deal_lday_cnt',
                            'order_deal_amt_d7','order_deal_amt_d15','order_deal_amt_d30','order_deal_amt_d60','order_deal_amt_d90','order_deal_amt_d180','order_deal_amt_d360',
                            'deal_client_cnt_d7','deal_client_cnt_d15','deal_client_cnt_d30','deal_client_cnt_d60','deal_client_cnt_d90','deal_client_cnt_d180','deal_client_cnt_d360',
                            'app_user_len_d7','app_user_len_d15','app_user_len_d30','app_user_len_d60','app_user_len_d90','app_user_len_d180','app_user_len_d360',
                            'app_user_cnt_d7','app_user_cnt_d15','app_user_cnt_d30','app_user_cnt_d60','app_user_cnt_d90','app_user_cnt_d180','app_user_cnt_d360',
                            'app_user_hour_cnt_d7','app_user_hour_cnt_d15','app_user_hour_cnt_d30','app_user_hour_cnt_d60','app_user_hour_cnt_d90','app_user_hour_cnt_d180','app_user_hour_cnt_d360',
                            'app_user_day_cnt_d7','app_user_day_cnt_d15','app_user_day_cnt_d30','app_user_day_cnt_d60','app_user_day_cnt_d90','app_user_day_cnt_d180','app_user_day_cnt_d360',
                            'app_user_len_avg_d7','app_user_len_avg_d15','app_user_len_avg_d30','app_user_len_avg_d60','app_user_len_avg_d90','app_user_len_avg_d180','app_user_len_avg_d360',
                            'app_user_cnt_avg_d7','app_user_cnt_avg_d15','app_user_cnt_avg_d30','app_user_cnt_avg_d60','app_user_cnt_avg_d90','app_user_cnt_avg_d180','app_user_cnt_avg_d360',
                            'app_user_hour_cnt_avg_d7','app_user_hour_cnt_avg_d15','app_user_hour_cnt_avg_d30','app_user_hour_cnt_avg_d60','app_user_hour_cnt_avg_d90','app_user_hour_cnt_avg_d180','app_user_hour_cnt_avg_d360',
                            'app_user_lday_cnt_d30',
                            'update_time'
                           ]

    h_sql = sql
    try:
        hive_cur.execute(h_sql)
        all_data = hive_cur.fetchall()
        all_data_df=pd.DataFrame(all_data,columns=broker_data_columns)
        all_data_df['recommend_cnt_d360'] = all_data_df['recommend_cnt_d360'] - all_data_df['recommend_cnt_d180']
        all_data_df['recommend_cnt_d180'] = all_data_df['recommend_cnt_d180'] - all_data_df['recommend_cnt_d90']
        all_data_df['recommend_cnt_d90'] = all_data_df['recommend_cnt_d90'] - all_data_df['recommend_cnt_d60']
        all_data_df['recommend_cnt_d60'] = all_data_df['recommend_cnt_d60'] - all_data_df['recommend_cnt_d30']
        all_data_df['recommend_cnt_d30'] = all_data_df['recommend_cnt_d30'] - all_data_df['recommend_cnt_d15']
        all_data_df['recommend_cnt_d15'] = all_data_df['recommend_cnt_d15'] - all_data_df['recommend_cnt_d7']
        all_data_df['recommend_client_cnt_d360'] = all_data_df['recommend_client_cnt_d360'] - all_data_df[
            'recommend_client_cnt_d180']
        all_data_df['recommend_client_cnt_d180'] = all_data_df['recommend_client_cnt_d180'] - all_data_df[
            'recommend_client_cnt_d90']
        all_data_df['recommend_client_cnt_d90'] = all_data_df['recommend_client_cnt_d90'] - all_data_df[
            'recommend_client_cnt_d60']
        all_data_df['recommend_client_cnt_d60'] = all_data_df['recommend_client_cnt_d60'] - all_data_df[
            'recommend_client_cnt_d30']
        all_data_df['recommend_client_cnt_d30'] = all_data_df['recommend_client_cnt_d30'] - all_data_df[
            'recommend_client_cnt_d15']
        all_data_df['recommend_client_cnt_d15'] = all_data_df['recommend_client_cnt_d15'] - all_data_df[
            'recommend_client_cnt_d7']
        all_data_df['visit_cnt_d360'] = all_data_df['visit_cnt_d360'] - all_data_df['visit_cnt_d180']
        all_data_df['visit_cnt_d180'] = all_data_df['visit_cnt_d180'] - all_data_df['visit_cnt_d90']
        all_data_df['visit_cnt_d90'] = all_data_df['visit_cnt_d90'] - all_data_df['visit_cnt_d60']
        all_data_df['visit_cnt_d60'] = all_data_df['visit_cnt_d60'] - all_data_df['visit_cnt_d30']
        all_data_df['visit_cnt_d30'] = all_data_df['visit_cnt_d30'] - all_data_df['visit_cnt_d15']
        all_data_df['visit_cnt_d15'] = all_data_df['visit_cnt_d15'] - all_data_df['visit_cnt_d7']
        all_data_df['visit_client_cnt_d360'] = all_data_df['visit_client_cnt_d360'] - all_data_df[
            'visit_client_cnt_d180']
        all_data_df['visit_client_cnt_d180'] = all_data_df['visit_client_cnt_d180'] - all_data_df[
            'visit_client_cnt_d90']
        all_data_df['visit_client_cnt_d90'] = all_data_df['visit_client_cnt_d90'] - all_data_df['visit_client_cnt_d60']
        all_data_df['visit_client_cnt_d60'] = all_data_df['visit_client_cnt_d60'] - all_data_df['visit_client_cnt_d30']
        all_data_df['visit_client_cnt_d30'] = all_data_df['visit_client_cnt_d30'] - all_data_df['visit_client_cnt_d15']
        all_data_df['visit_client_cnt_d15'] = all_data_df['visit_client_cnt_d15'] - all_data_df['visit_client_cnt_d7']
        all_data_df['order_deal_cnt_d360'] = all_data_df['order_deal_cnt_d360'] - all_data_df['order_deal_cnt_d180']
        all_data_df['order_deal_cnt_d180'] = all_data_df['order_deal_cnt_d180'] - all_data_df['order_deal_cnt_d90']
        all_data_df['order_deal_cnt_d90'] = all_data_df['order_deal_cnt_d90'] - all_data_df['order_deal_cnt_d60']
        all_data_df['order_deal_cnt_d60'] = all_data_df['order_deal_cnt_d60'] - all_data_df['order_deal_cnt_d30']
        all_data_df['order_deal_cnt_d30'] = all_data_df['order_deal_cnt_d30'] - all_data_df['order_deal_cnt_d15']
        all_data_df['order_deal_cnt_d15'] = all_data_df['order_deal_cnt_d15'] - all_data_df['order_deal_cnt_d7']
        all_data_df['order_deal_amt_d360'] = all_data_df['order_deal_amt_d360'] - all_data_df['order_deal_amt_d180']
        all_data_df['order_deal_amt_d180'] = all_data_df['order_deal_amt_d180'] - all_data_df['order_deal_amt_d90']
        all_data_df['order_deal_amt_d90'] = all_data_df['order_deal_amt_d90'] - all_data_df['order_deal_amt_d60']
        all_data_df['order_deal_amt_d60'] = all_data_df['order_deal_amt_d60'] - all_data_df['order_deal_amt_d30']
        all_data_df['order_deal_amt_d30'] = all_data_df['order_deal_amt_d30'] - all_data_df['order_deal_amt_d15']
        all_data_df['order_deal_amt_d15'] = all_data_df['order_deal_amt_d15'] - all_data_df['order_deal_amt_d7']
        all_data_df['deal_client_cnt_d360'] = all_data_df['deal_client_cnt_d360'] - all_data_df['deal_client_cnt_d180']
        all_data_df['deal_client_cnt_d180'] = all_data_df['deal_client_cnt_d180'] - all_data_df['deal_client_cnt_d90']
        all_data_df['deal_client_cnt_d90'] = all_data_df['deal_client_cnt_d90'] - all_data_df['deal_client_cnt_d60']
        all_data_df['deal_client_cnt_d60'] = all_data_df['deal_client_cnt_d60'] - all_data_df['deal_client_cnt_d30']
        all_data_df['deal_client_cnt_d30'] = all_data_df['deal_client_cnt_d30'] - all_data_df['deal_client_cnt_d15']
        all_data_df['deal_client_cnt_d15'] = all_data_df['deal_client_cnt_d15'] - all_data_df['deal_client_cnt_d7']
        all_data_df['app_user_len_d360'] = all_data_df['app_user_len_d360'] - all_data_df['app_user_len_d180']
        all_data_df['app_user_len_d180'] = all_data_df['app_user_len_d180'] - all_data_df['app_user_len_d90']
        all_data_df['app_user_len_d90'] = all_data_df['app_user_len_d90'] - all_data_df['app_user_len_d60']
        all_data_df['app_user_len_d60'] = all_data_df['app_user_len_d60'] - all_data_df['app_user_len_d30']
        all_data_df['app_user_len_d30'] = all_data_df['app_user_len_d30'] - all_data_df['app_user_len_d15']
        all_data_df['app_user_len_d15'] = all_data_df['app_user_len_d15'] - all_data_df['app_user_len_d7']
        all_data_df['app_user_cnt_d360'] = all_data_df['app_user_cnt_d360'] - all_data_df['app_user_cnt_d180']
        all_data_df['app_user_cnt_d180'] = all_data_df['app_user_cnt_d180'] - all_data_df['app_user_cnt_d90']
        all_data_df['app_user_cnt_d90'] = all_data_df['app_user_cnt_d90'] - all_data_df['app_user_cnt_d60']
        all_data_df['app_user_cnt_d60'] = all_data_df['app_user_cnt_d60'] - all_data_df['app_user_cnt_d30']
        all_data_df['app_user_cnt_d30'] = all_data_df['app_user_cnt_d30'] - all_data_df['app_user_cnt_d15']
        all_data_df['app_user_cnt_d15'] = all_data_df['app_user_cnt_d15'] - all_data_df['app_user_cnt_d7']
        all_data_df['app_user_hour_cnt_d360'] = all_data_df['app_user_hour_cnt_d360'] - all_data_df[
            'app_user_hour_cnt_d180']
        all_data_df['app_user_hour_cnt_d180'] = all_data_df['app_user_hour_cnt_d180'] - all_data_df[
            'app_user_hour_cnt_d90']
        all_data_df['app_user_hour_cnt_d90'] = all_data_df['app_user_hour_cnt_d90'] - all_data_df[
            'app_user_hour_cnt_d60']
        all_data_df['app_user_hour_cnt_d60'] = all_data_df['app_user_hour_cnt_d60'] - all_data_df[
            'app_user_hour_cnt_d30']
        all_data_df['app_user_hour_cnt_d30'] = all_data_df['app_user_hour_cnt_d30'] - all_data_df[
            'app_user_hour_cnt_d15']
        all_data_df['app_user_hour_cnt_d15'] = all_data_df['app_user_hour_cnt_d15'] - all_data_df[
            'app_user_hour_cnt_d7']
        all_data_df['app_user_day_cnt_d360'] = all_data_df['app_user_day_cnt_d360'] - all_data_df[
            'app_user_day_cnt_d180']
        all_data_df['app_user_day_cnt_d180'] = all_data_df['app_user_day_cnt_d180'] - all_data_df[
            'app_user_day_cnt_d90']
        all_data_df['app_user_day_cnt_d90'] = all_data_df['app_user_day_cnt_d90'] - all_data_df['app_user_day_cnt_d60']
        all_data_df['app_user_day_cnt_d60'] = all_data_df['app_user_day_cnt_d60'] - all_data_df['app_user_day_cnt_d30']
        all_data_df['app_user_day_cnt_d30'] = all_data_df['app_user_day_cnt_d30'] - all_data_df['app_user_day_cnt_d15']
        all_data_df['app_user_day_cnt_d15'] = all_data_df['app_user_day_cnt_d15'] - all_data_df['app_user_day_cnt_d7']
        all_data_df['app_user_len_avg_d360'] = all_data_df['app_user_len_avg_d360'] - all_data_df[
            'app_user_len_avg_d180']
        all_data_df['app_user_len_avg_d180'] = all_data_df['app_user_len_avg_d180'] - all_data_df[
            'app_user_len_avg_d90']
        all_data_df['app_user_len_avg_d90'] = all_data_df['app_user_len_avg_d90'] - all_data_df['app_user_len_avg_d60']
        all_data_df['app_user_len_avg_d60'] = all_data_df['app_user_len_avg_d60'] - all_data_df['app_user_len_avg_d30']
        all_data_df['app_user_len_avg_d30'] = all_data_df['app_user_len_avg_d30'] - all_data_df['app_user_len_avg_d15']
        all_data_df['app_user_len_avg_d15'] = all_data_df['app_user_len_avg_d15'] - all_data_df['app_user_len_avg_d7']
        all_data_df['app_user_cnt_avg_d360'] = all_data_df['app_user_cnt_avg_d360'] - all_data_df[
            'app_user_cnt_avg_d180']
        all_data_df['app_user_cnt_avg_d180'] = all_data_df['app_user_cnt_avg_d180'] - all_data_df[
            'app_user_cnt_avg_d90']
        all_data_df['app_user_cnt_avg_d90'] = all_data_df['app_user_cnt_avg_d90'] - all_data_df['app_user_cnt_avg_d60']
        all_data_df['app_user_cnt_avg_d60'] = all_data_df['app_user_cnt_avg_d60'] - all_data_df['app_user_cnt_avg_d30']
        all_data_df['app_user_cnt_avg_d30'] = all_data_df['app_user_cnt_avg_d30'] - all_data_df['app_user_cnt_avg_d15']
        all_data_df['app_user_cnt_avg_d15'] = all_data_df['app_user_cnt_avg_d15'] - all_data_df['app_user_cnt_avg_d7']
        all_data_df['app_user_hour_cnt_avg_d360'] = all_data_df['app_user_hour_cnt_avg_d360'] - all_data_df[
            'app_user_hour_cnt_avg_d180']
        all_data_df['app_user_hour_cnt_avg_d180'] = all_data_df['app_user_hour_cnt_avg_d180'] - all_data_df[
            'app_user_hour_cnt_avg_d90']
        all_data_df['app_user_hour_cnt_avg_d90'] = all_data_df['app_user_hour_cnt_avg_d90'] - all_data_df[
            'app_user_hour_cnt_avg_d60']
        all_data_df['app_user_hour_cnt_avg_d60'] = all_data_df['app_user_hour_cnt_avg_d60'] - all_data_df[
            'app_user_hour_cnt_avg_d30']
        all_data_df['app_user_hour_cnt_avg_d30'] = all_data_df['app_user_hour_cnt_avg_d30'] - all_data_df[
            'app_user_hour_cnt_avg_d15']
        all_data_df['app_user_hour_cnt_avg_d15'] = all_data_df['app_user_hour_cnt_avg_d15'] - all_data_df[
            'app_user_hour_cnt_avg_d7']

        log_object.write_logs('读取数据记录成功!')
        hive_cur.close()
        return all_data_df

    except Exception as e:
        all_data_df = pd.read_csv(r'C:\Users\Public\Documents\KK6 Files\KK6.0\Account\966207@kk5.evergrande.com\file_cache\1.test.csv',sep='\t',header=0, names=broker_data_columns)
        # all_data_df = pd.read_csv(r'/my_model/1.csv', sep='\t',header=0, names=broker_data_columns)

        all_data_df['recommend_cnt_d360'] = all_data_df['recommend_cnt_d360'] - all_data_df['recommend_cnt_d180']
        all_data_df['recommend_cnt_d180'] = all_data_df['recommend_cnt_d180'] - all_data_df['recommend_cnt_d90']
        all_data_df['recommend_cnt_d90'] = all_data_df['recommend_cnt_d90'] - all_data_df['recommend_cnt_d60']
        all_data_df['recommend_cnt_d60'] = all_data_df['recommend_cnt_d60'] - all_data_df['recommend_cnt_d30']
        all_data_df['recommend_cnt_d30'] = all_data_df['recommend_cnt_d30'] - all_data_df['recommend_cnt_d15']
        all_data_df['recommend_cnt_d15'] = all_data_df['recommend_cnt_d15'] - all_data_df['recommend_cnt_d7']
        all_data_df['recommend_client_cnt_d360'] = all_data_df['recommend_client_cnt_d360'] - all_data_df[
            'recommend_client_cnt_d180']
        all_data_df['recommend_client_cnt_d180'] = all_data_df['recommend_client_cnt_d180'] - all_data_df[
            'recommend_client_cnt_d90']
        all_data_df['recommend_client_cnt_d90'] = all_data_df['recommend_client_cnt_d90'] - all_data_df[
            'recommend_client_cnt_d60']
        all_data_df['recommend_client_cnt_d60'] = all_data_df['recommend_client_cnt_d60'] - all_data_df[
            'recommend_client_cnt_d30']
        all_data_df['recommend_client_cnt_d30'] = all_data_df['recommend_client_cnt_d30'] - all_data_df[
            'recommend_client_cnt_d15']
        all_data_df['recommend_client_cnt_d15'] = all_data_df['recommend_client_cnt_d15'] - all_data_df[
            'recommend_client_cnt_d7']
        all_data_df['visit_cnt_d360'] = all_data_df['visit_cnt_d360'] - all_data_df['visit_cnt_d180']
        all_data_df['visit_cnt_d180'] = all_data_df['visit_cnt_d180'] - all_data_df['visit_cnt_d90']
        all_data_df['visit_cnt_d90'] = all_data_df['visit_cnt_d90'] - all_data_df['visit_cnt_d60']
        all_data_df['visit_cnt_d60'] = all_data_df['visit_cnt_d60'] - all_data_df['visit_cnt_d30']
        all_data_df['visit_cnt_d30'] = all_data_df['visit_cnt_d30'] - all_data_df['visit_cnt_d15']
        all_data_df['visit_cnt_d15'] = all_data_df['visit_cnt_d15'] - all_data_df['visit_cnt_d7']
        all_data_df['visit_client_cnt_d360'] = all_data_df['visit_client_cnt_d360'] - all_data_df[
            'visit_client_cnt_d180']
        all_data_df['visit_client_cnt_d180'] = all_data_df['visit_client_cnt_d180'] - all_data_df[
            'visit_client_cnt_d90']
        all_data_df['visit_client_cnt_d90'] = all_data_df['visit_client_cnt_d90'] - all_data_df['visit_client_cnt_d60']
        all_data_df['visit_client_cnt_d60'] = all_data_df['visit_client_cnt_d60'] - all_data_df['visit_client_cnt_d30']
        all_data_df['visit_client_cnt_d30'] = all_data_df['visit_client_cnt_d30'] - all_data_df['visit_client_cnt_d15']
        all_data_df['visit_client_cnt_d15'] = all_data_df['visit_client_cnt_d15'] - all_data_df['visit_client_cnt_d7']
        all_data_df['order_deal_cnt_d360'] = all_data_df['order_deal_cnt_d360'] - all_data_df['order_deal_cnt_d180']
        all_data_df['order_deal_cnt_d180'] = all_data_df['order_deal_cnt_d180'] - all_data_df['order_deal_cnt_d90']
        all_data_df['order_deal_cnt_d90'] = all_data_df['order_deal_cnt_d90'] - all_data_df['order_deal_cnt_d60']
        all_data_df['order_deal_cnt_d60'] = all_data_df['order_deal_cnt_d60'] - all_data_df['order_deal_cnt_d30']
        all_data_df['order_deal_cnt_d30'] = all_data_df['order_deal_cnt_d30'] - all_data_df['order_deal_cnt_d15']
        all_data_df['order_deal_cnt_d15'] = all_data_df['order_deal_cnt_d15'] - all_data_df['order_deal_cnt_d7']
        all_data_df['order_deal_amt_d360'] = all_data_df['order_deal_amt_d360'] - all_data_df['order_deal_amt_d180']
        all_data_df['order_deal_amt_d180'] = all_data_df['order_deal_amt_d180'] - all_data_df['order_deal_amt_d90']
        all_data_df['order_deal_amt_d90'] = all_data_df['order_deal_amt_d90'] - all_data_df['order_deal_amt_d60']
        all_data_df['order_deal_amt_d60'] = all_data_df['order_deal_amt_d60'] - all_data_df['order_deal_amt_d30']
        all_data_df['order_deal_amt_d30'] = all_data_df['order_deal_amt_d30'] - all_data_df['order_deal_amt_d15']
        all_data_df['order_deal_amt_d15'] = all_data_df['order_deal_amt_d15'] - all_data_df['order_deal_amt_d7']
        all_data_df['deal_client_cnt_d360'] = all_data_df['deal_client_cnt_d360'] - all_data_df['deal_client_cnt_d180']
        all_data_df['deal_client_cnt_d180'] = all_data_df['deal_client_cnt_d180'] - all_data_df['deal_client_cnt_d90']
        all_data_df['deal_client_cnt_d90'] = all_data_df['deal_client_cnt_d90'] - all_data_df['deal_client_cnt_d60']
        all_data_df['deal_client_cnt_d60'] = all_data_df['deal_client_cnt_d60'] - all_data_df['deal_client_cnt_d30']
        all_data_df['deal_client_cnt_d30'] = all_data_df['deal_client_cnt_d30'] - all_data_df['deal_client_cnt_d15']
        all_data_df['deal_client_cnt_d15'] = all_data_df['deal_client_cnt_d15'] - all_data_df['deal_client_cnt_d7']
        all_data_df['app_user_len_d360'] = all_data_df['app_user_len_d360'] - all_data_df['app_user_len_d180']
        all_data_df['app_user_len_d180'] = all_data_df['app_user_len_d180'] - all_data_df['app_user_len_d90']
        all_data_df['app_user_len_d90'] = all_data_df['app_user_len_d90'] - all_data_df['app_user_len_d60']
        all_data_df['app_user_len_d60'] = all_data_df['app_user_len_d60'] - all_data_df['app_user_len_d30']
        all_data_df['app_user_len_d30'] = all_data_df['app_user_len_d30'] - all_data_df['app_user_len_d15']
        all_data_df['app_user_len_d15'] = all_data_df['app_user_len_d15'] - all_data_df['app_user_len_d7']
        all_data_df['app_user_cnt_d360'] = all_data_df['app_user_cnt_d360'] - all_data_df['app_user_cnt_d180']
        all_data_df['app_user_cnt_d180'] = all_data_df['app_user_cnt_d180'] - all_data_df['app_user_cnt_d90']
        all_data_df['app_user_cnt_d90'] = all_data_df['app_user_cnt_d90'] - all_data_df['app_user_cnt_d60']
        all_data_df['app_user_cnt_d60'] = all_data_df['app_user_cnt_d60'] - all_data_df['app_user_cnt_d30']
        all_data_df['app_user_cnt_d30'] = all_data_df['app_user_cnt_d30'] - all_data_df['app_user_cnt_d15']
        all_data_df['app_user_cnt_d15'] = all_data_df['app_user_cnt_d15'] - all_data_df['app_user_cnt_d7']
        all_data_df['app_user_hour_cnt_d360'] = all_data_df['app_user_hour_cnt_d360'] - all_data_df[
            'app_user_hour_cnt_d180']
        all_data_df['app_user_hour_cnt_d180'] = all_data_df['app_user_hour_cnt_d180'] - all_data_df[
            'app_user_hour_cnt_d90']
        all_data_df['app_user_hour_cnt_d90'] = all_data_df['app_user_hour_cnt_d90'] - all_data_df[
            'app_user_hour_cnt_d60']
        all_data_df['app_user_hour_cnt_d60'] = all_data_df['app_user_hour_cnt_d60'] - all_data_df[
            'app_user_hour_cnt_d30']
        all_data_df['app_user_hour_cnt_d30'] = all_data_df['app_user_hour_cnt_d30'] - all_data_df[
            'app_user_hour_cnt_d15']
        all_data_df['app_user_hour_cnt_d15'] = all_data_df['app_user_hour_cnt_d15'] - all_data_df[
            'app_user_hour_cnt_d7']
        all_data_df['app_user_day_cnt_d360'] = all_data_df['app_user_day_cnt_d360'] - all_data_df[
            'app_user_day_cnt_d180']
        all_data_df['app_user_day_cnt_d180'] = all_data_df['app_user_day_cnt_d180'] - all_data_df[
            'app_user_day_cnt_d90']
        all_data_df['app_user_day_cnt_d90'] = all_data_df['app_user_day_cnt_d90'] - all_data_df['app_user_day_cnt_d60']
        all_data_df['app_user_day_cnt_d60'] = all_data_df['app_user_day_cnt_d60'] - all_data_df['app_user_day_cnt_d30']
        all_data_df['app_user_day_cnt_d30'] = all_data_df['app_user_day_cnt_d30'] - all_data_df['app_user_day_cnt_d15']
        all_data_df['app_user_day_cnt_d15'] = all_data_df['app_user_day_cnt_d15'] - all_data_df['app_user_day_cnt_d7']
        all_data_df['app_user_len_avg_d360'] = all_data_df['app_user_len_avg_d360'] - all_data_df[
            'app_user_len_avg_d180']
        all_data_df['app_user_len_avg_d180'] = all_data_df['app_user_len_avg_d180'] - all_data_df[
            'app_user_len_avg_d90']
        all_data_df['app_user_len_avg_d90'] = all_data_df['app_user_len_avg_d90'] - all_data_df['app_user_len_avg_d60']
        all_data_df['app_user_len_avg_d60'] = all_data_df['app_user_len_avg_d60'] - all_data_df['app_user_len_avg_d30']
        all_data_df['app_user_len_avg_d30'] = all_data_df['app_user_len_avg_d30'] - all_data_df['app_user_len_avg_d15']
        all_data_df['app_user_len_avg_d15'] = all_data_df['app_user_len_avg_d15'] - all_data_df['app_user_len_avg_d7']
        all_data_df['app_user_cnt_avg_d360'] = all_data_df['app_user_cnt_avg_d360'] - all_data_df[
            'app_user_cnt_avg_d180']
        all_data_df['app_user_cnt_avg_d180'] = all_data_df['app_user_cnt_avg_d180'] - all_data_df[
            'app_user_cnt_avg_d90']
        all_data_df['app_user_cnt_avg_d90'] = all_data_df['app_user_cnt_avg_d90'] - all_data_df['app_user_cnt_avg_d60']
        all_data_df['app_user_cnt_avg_d60'] = all_data_df['app_user_cnt_avg_d60'] - all_data_df['app_user_cnt_avg_d30']
        all_data_df['app_user_cnt_avg_d30'] = all_data_df['app_user_cnt_avg_d30'] - all_data_df['app_user_cnt_avg_d15']
        all_data_df['app_user_cnt_avg_d15'] = all_data_df['app_user_cnt_avg_d15'] - all_data_df['app_user_cnt_avg_d7']
        all_data_df['app_user_hour_cnt_avg_d360'] = all_data_df['app_user_hour_cnt_avg_d360'] - all_data_df[
            'app_user_hour_cnt_avg_d180']
        all_data_df['app_user_hour_cnt_avg_d180'] = all_data_df['app_user_hour_cnt_avg_d180'] - all_data_df[
            'app_user_hour_cnt_avg_d90']
        all_data_df['app_user_hour_cnt_avg_d90'] = all_data_df['app_user_hour_cnt_avg_d90'] - all_data_df[
            'app_user_hour_cnt_avg_d60']
        all_data_df['app_user_hour_cnt_avg_d60'] = all_data_df['app_user_hour_cnt_avg_d60'] - all_data_df[
            'app_user_hour_cnt_avg_d30']
        all_data_df['app_user_hour_cnt_avg_d30'] = all_data_df['app_user_hour_cnt_avg_d30'] - all_data_df[
            'app_user_hour_cnt_avg_d15']
        all_data_df['app_user_hour_cnt_avg_d15'] = all_data_df['app_user_hour_cnt_avg_d15'] - all_data_df[
            'app_user_hour_cnt_avg_d7']
        print(all_data_df.head(),all_data_df.head().shape)
        log_object.write_logs('读取本地测试数据成功！')
        log_object.write_logs(e)
        return all_data_df.head(200)

def get_history_broker_ability_info(hive_cur,sql='select t.guid,t.BrokerLevelFlag from db_broker_ability.label_broker_level_score_test t where t.update_time in (select  max(update_time) from label_broker_level_score);'):
    log_object = logs.logging()
    h_sql = sql
    history_broker_ability_columns = ['guid', 'CustomerDevelopmentScore', 'PerformancedScore', 'LivenessScore', 'totalScore',
     'CustomerDevelopmentScoreRadar', 'PerformancedScoreRadar', 'LivenessScoreRadar', 'totalScoreRadar','BrokerLevelFlag','group_level_rank_percent','update_time']
    try:
        hive_cur.execute(h_sql)
        all_data = hive_cur.fetchall()
        all_data_df = pd.DataFrame(all_data, columns=history_broker_ability_columns)
        all_data_df = all_data_df[['guid','BrokerLevelFlag']] # 仅选取上个月会员的等级信息即可···
        log_object.write_logs('读取会员等级得分数据成功!')
        hive_cur.close()
        return all_data_df
    except Exception as e:
        all_data_df = pd.read_csv(
            r'./broker_level_score_history.csv', sep=',',
            header=0, names=history_broker_ability_columns,dtype={'guid':object})
        all_data_df = all_data_df[['guid', 'BrokerLevelFlag']]  # 仅选取上个月会员的等级信息即可···
        log_object.write_logs('读取本地测试会员等级得分数据成功！')
        log_object.write_logs(e)
        return all_data_df

def save_broker_score_file(all_data_df):
    all_data_df.to_csv('./OrigineBrokerData/broker_score.txt',header=None,sep=',')

def create_table(hive_cur):
    #检查是否已创建数据库···
    hive_cur.execute('show databases;')
    all_databases_buff = hive_cur.fetchall()
    all_databases=[database[0] for database in all_databases_buff]
    if 'db_broker_ability' not in all_databases:
        print('the database db_broker_ability is not existing,create databse db_broker_ability!')
        hive_cur.execute('create database db_broker_ability;')
        # hive_cur.execute('create table db_broker_ability.label_broker_ability_score(guid string,res_city string,register_city string,working_city string,is_manager string,recommend_cnt_d7 int,recommend_cnt_d15 int,recommend_cnt_d30 int,recommend_cnt_d90 int,recommend_cnt_d180 int,recommend_client_cnt_d7 int,recommend_client_cnt_d15 int,recommend_client_cnt_d30 int,recommend_client_cnt_d90 int,recommend_client_cnt_d180 int,visit_cnt_d7 int,visit_cnt_d15 int,visit_cnt_d30 int,visit_cnt_d90 int,visit_cnt_d180 int,visit_client_cnt_d7 int,visit_client_cnt_d15 int,visit_client_cnt_d30 int,visit_client_cnt_d90 int,visit_client_cnt_d180 int,online_order_cnt_d7 int,online_order_cnt_d15 int,online_order_cnt_d30 int,online_order_cnt_d90 int,online_order_cnt_d180 int,order_deal_cnt_d7 int,order_deal_cnt_d15 int,order_deal_cnt_d30 int,order_deal_cnt_d90 int,order_deal_cnt_d180 int,order_deal_amt_d7 int,order_deal_amt_d15 int,order_deal_amt_d30 int,order_deal_amt_d90 int,order_deal_amt_d180 int,deal_client_cnt_d7 int,deal_client_cnt_d15 int,deal_client_cnt_d30 int,deal_client_cnt_d90 int,deal_client_cnt_d180 int,update_time string) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE; ')
        hive_cur.execute('create table db_broker_ability.label_broker_level_score_test(guid string,CustomerDevelopmentScore int,PerformancedScore int,LivenessScore int,totalScore int,CustomerDevelopmentScoreRadar int,PerformancedScoreRadar int,LivenessScoreRadar int,totalScoreRadar int,BrokerLevelFlag string,update_time string) ROW FORMAT DELIMITED FIELDS TERMINATED BY \',\' STORED AS TEXTFILE;')
    else:
        hive_cur.execute('use db_broker_ability;')
        hive_cur.execute('show tables;')
        all_table_buff = hive_cur.fetchall()
        all_table = [table[0] for table in all_table_buff]
        if 'label_broker_level_score_test' not in all_table:
            print('the table label_broker_level_score is not existing,create table label_broker_level_score!')
            hive_cur.execute('create table db_broker_ability.label_broker_level_score_test(guid string,CustomerDevelopmentScore int,PerformancedScore int,LivenessScore int,totalScore int,CustomerDevelopmentScoreRadar int,PerformancedScoreRadar int,LivenessScoreRadar int,totalScoreRadar int,BrokerLevelFlag string,update_time string) ROW FORMAT DELIMITED FIELDS TERMINATED BY \',\' STORED AS TEXTFILE;')


def check_datetime(data_df):
    date_info_flag = {'is_month_beginning':False,'is_week_beginning':False}

    date_time = data_df.iloc[0]['update_time'][:10]
    date_week_num = datetime.strptime(date_time, '%Y-%m-%d').weekday()
    print('check_datetime:',date_week_num)

    # if date_time[-2:] == '01':
    #     date_info_flag['is_month_beginning']=True
    # if date_week_num==0:
    #     date_info_flag['is_week_beginning'] = True

    if date_time[-2:] in ['30','31','01']:
        date_info_flag['is_month_beginning']=True  #月底更新标志···
    if date_time[-2:] in ['14','15','16']:
        date_info_flag['is_week_beginning'] = True #15号更新标志···

    return date_info_flag