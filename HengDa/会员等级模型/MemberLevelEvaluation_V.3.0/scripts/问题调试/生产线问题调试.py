# 1.生产上插入顺序颠倒的问题

import pandas as pd
import re



origin_df = pd.read_csv(r'F:\工作文档\会员等级模型\MemberLevelEvaluation_V.1.0-生产上反馈\MemberLevelEvaluation_V.1.0\scripts\broker_level_score.csv')


def write_df(table_name, df, cols_type=dict(), once_cnt=1000):

    try:
        if not cols_type:
            cols_type = get_cols_type(df)

        data_cnt = df.shape[0]
        iters = int(data_cnt / once_cnt) if data_cnt % once_cnt == 0 else int(data_cnt / once_cnt) + 1
        sql_insert_sub = "insert into {} values({})"

        for i in range(iters):
            start = i * once_cnt
            end = data_cnt if i == iters - 1 else (i + 1) * once_cnt
            df_sub = df.iloc[start:end]
            dat_lst = ["(" + _concat_cols_to_string(i, cols_type) + ")" for i in df_sub.to_dict("records")]
            sql_insert = sql_insert_sub.format(table_name, ",".join(dat_lst))
            print(sql_insert)
    except Exception as e:
        pass

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
    # print(f_types)
    return f_types


def _concat_cols_to_string(val_dic, col_type_dic):
    columns_list = ['guid', 'CustomerDevelopmentScore', 'PerformancedScore', 'LivenessScore', 'totalScore',
                    'CustomerDevelopmentScoreRadar', 'PerformancedScoreRadar', 'LivenessScoreRadar', 'totalScoreRadar',
                    'BrokerLevelFlag', 'update_time']
    val_str = ''
    for colums_key in columns_list:
        if 'str' in col_type_dic[colums_key]:
            columns_string = "'" + val_dic[colums_key] + "'"
        else:
            columns_string=str(val_dic[colums_key])
        val_str += columns_string + ','
    val_str = val_str[0:-1]
    print(val_str)
    return val_str


    # for  key, val in col_type_dic.items():
    #     if 'str' in val:
    #         columns_string = "'" + val_dic[key] + "'"
    #     else:
    #         columns_string=str(val_dic[key])
    #
    #     val_str += columns_string + ','
    #
    # val_str = val_str[0:-1]
    #
    # print col_type_dic.items()
    # print val_str
    # return val_str
    # val_str = val_str[0:-1]

    # for key, val in val_dic.items():
    #
    #     if 'str' in columns_type_list[column_id]:
    #         val = "'" + val + "'"
    #     val_str += str(val) + ','
    #     column_id += 1

    # print(val_str)



write_df('label_broker_level_score', origin_df,get_cols_type(origin_df), once_cnt=1000)