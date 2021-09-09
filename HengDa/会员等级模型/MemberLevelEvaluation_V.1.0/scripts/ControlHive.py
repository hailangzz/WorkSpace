#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# from krbcontext import krbcontext
import logs
from impala.dbapi import connect
import pandas as pd
import re
from decimal import Decimal

class hiveConnect():

    def __init__(self,config_dict,
                 kerberos_service_name='impala',
                 auth_mechanism='GSSAPI'):

        self.user = config_dict['user']
        self.password = config_dict['password']
        self.host = config_dict['host']
        self.port = config_dict['port']
        self.database = config_dict['database']
        self.kerberos_service_name = kerberos_service_name
        self.auth_mechanism = auth_mechanism

        self.create_connect()

    def check_table(self,table_name):
        log_object = logs.logging()
        try:
            self.cursor.execute('show tables;')
            all_table_buff = self.cursor.fetchall()
            all_table = [table[0] for table in all_table_buff]
            if table_name not in all_table:
                print('the table %s is not existing,create table %s!'%(table_name,table_name))
                sql_command = 'create table db_broker_ability.%s(guid string,CustomerDevelopmentScore int,PerformancedScore int,LivenessScore int,totalScore int,CustomerDevelopmentScoreRadar int,PerformancedScoreRadar int,LivenessScoreRadar int,totalScoreRadar int,BrokerLevelFlag string,update_time string) ROW FORMAT DELIMITED FIELDS TERMINATED BY \',\' STORED AS TEXTFILE;'%table_name
                self.cursor.execute()
            log_object.write_logs('检查评分表成功！')
        except Exception as e:
            log_object.write_logs(e)

    def create_connect(self):
        self.conn = connect(host=self.host, port=self.port, password=self.password, user=self.user,database=self.database,kerberos_service_name=self.kerberos_service_name,auth_mechanism=self.auth_mechanism )
        self.cursor = self.conn.cursor()

    def close_connect(self):

        self.cursor.close()
        self.conn.close()

    def read(self, statement):

        self.create_connect()
        self.cursor.execute(statement)
        dat = self.cursor.fetchall()
        self.close_connect()
        return dat


    def write_df(self, table_name, df, cols_type=dict(), is_week_beginning=False,once_cnt=100):
        log_object = logs.logging()
        try:
            if is_week_beginning==True and table_name=='label_broker_level_score_week':#如果数据为周更新数据则，先清空周数据更新表
                self.cursor.execute('TRUNCATE db_broker_ability.%s' % table_name)

            self.create_connect()
            if not cols_type:
                cols_type = get_cols_type(df)

            data_cnt = df.shape[0]
            iters = int(data_cnt / once_cnt) if data_cnt % once_cnt == 0 else int(data_cnt / once_cnt) + 1
            sql_insert_sub = "insert into {} values({})"

            for i in range(iters):
                start = i * once_cnt
                end = data_cnt if i == iters - 1 else (i + 1) * once_cnt
                df_sub = df.iloc[start:end]
                dat_lst = ["(" + self._concat_cols_to_string(i, cols_type) + ")" for i in df_sub.to_dict("records")]
                sql_insert = sql_insert_sub.format(table_name, ",".join(dat_lst))
                print(sql_insert)
                self.cursor.execute(sql_insert)

            log_object.write_logs('插入打分结果成功!')
            self.close_connect()
        except Exception as e:
            log_object.write_logs(e)

    def delete(self, table_name):
        self.create_connect()
        sql_delete = "truncate table {}".format(table_name)
        self.cursor.execute(sql_delete)
        self.close_connect()

    def _get_val_by_type(self, col_name, col_value, col_type_dic):

        if pd.isnull(col_value):
            res = 'null'
        elif col_type_dic[col_name] in ['string', 'str'] and len(re.findall('^[0-9.]+$', str(col_value))) > 0:
            res = Decimal(str(col_value)).normalize()
            res = "'" + str(res) + "'"
        elif col_type_dic[col_name] in ['string', 'str']:
            res = "'" + str(col_value) + "'"
        elif col_type_dic[col_name] in ['bigint', 'int']:
            res = int(col_value)
        elif col_type_dic[col_name] in ['float']:
            res = float(col_value)
        else:
            raise ValueError("{}的类型未找到".format(col_name))

        return res

    def _concat_cols_to_string(self, val_dic, col_type_dic):

        columns_list = ['guid', 'CustomerDevelopmentScore', 'PerformancedScore', 'LivenessScore', 'totalScore',
                        'CustomerDevelopmentScoreRadar', 'PerformancedScoreRadar', 'LivenessScoreRadar',
                        'totalScoreRadar',
                        'BrokerLevelFlag', 'update_time']
        val_str = ''
        for colums_key in columns_list:
            if 'str' in col_type_dic[colums_key]:
                columns_string = "'" + val_dic[colums_key] + "'"
            else:
                columns_string = str(val_dic[colums_key])
            val_str += columns_string + ','
        val_str = val_str[0:-1]
        return val_str


    def _concat_cols_to_string_old(self, val_dic, col_type_dic):
        columns_list = ['guid','CustomerDevelopmentScore','PerformancedScore','LivenessScore','totalScore','CustomerDevelopmentScoreRadar','PerformancedScoreRadar','LivenessScoreRadar','totalScoreRadar','BrokerLevelFlag','update_time']
        val_str = ''
        for key, val in val_dic.items():
            res = self._get_val_by_type(key, val, col_type_dic)
            val_str += str(res) + ','

        val_str = val_str[0:-1]
        return val_str

def get_cols_type(df):
    cols_type = dict(df.dtypes.apply(lambda x: str(x)))
    f_types = {}
    for k, v in cols_type.items():
        if len(re.findall('int', v)) > 0:
            f_type = 'int'
        elif len(re.findall('float', v)) > 0:
            f_type = 'float'
        elif len(re.findall('object', v)) > 0:
            f_type = 'str'
        elif len(re.findall('bool', v)) > 0:
            f_type = 'bool'
        elif len(re.findall('datetime', v)) > 0:
            f_type = 'time'
        else:
            f_type = 'unknown'

        f_types[k] = f_type

    return f_types



