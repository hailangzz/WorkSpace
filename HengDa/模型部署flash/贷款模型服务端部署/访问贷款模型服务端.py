import json
import requests
import pandas as  pd
"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

"""Reading test batch
"""
df = pd.read_csv(r'F:\\工作文档\\tensorflow服务模型\\贷款风控模型训练\\data\\test.csv', encoding="utf-8-sig")
df = df.head()

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')
# print(data)

print(json.dumps(data))
"""POST <url>/predict
"""
resp = requests.post("http://127.0.0.1:5000/predict", \
                    data = json.dumps(data),\
                    headers= header)
# json.dumps() 是把python对象转换成json对象的一个过程，生成的是字符串。
print(resp.status_code)
print(resp.json())