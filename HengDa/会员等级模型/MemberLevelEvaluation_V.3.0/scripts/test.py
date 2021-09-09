# _*_coding:utf-8_*_

import os
from krbcontext import krbcontext
from impala.dbapi import connect
os.system("kinit -kt /opt/kerberos/admin.keytab admin")
conn = connect(host='bigdata-prd-cdh-zk-01', port=25004,user='impala',password='hive',database='default',kerberos_service_name='impala',auth_mechanism = 'GSSAPI')

cur = conn.cursor()
cur.execute('use db_broker_ability;')
cur.execute('show tables;')
result=cur.fetchall()
for data in result:
    print (data)

cur.execute('insert into db_broker_ability.test_part_table PARTITION (dt=\'2019-03-21\') values ("555","helloworld");;')


import pickle
df=open(r'F:\工作文档\会员等级模型\MemberLevelEvaluation_V.1.0-生产上反馈\MemberLevelEvaluation_V.1.0 - 营销测试修改版-特征百分化\MemberLevelEvaluation_V.1.0\scripts\feature_weight.pickle','rb')#注意此处是rb

weight=pickle.load(df)
print(weight)
