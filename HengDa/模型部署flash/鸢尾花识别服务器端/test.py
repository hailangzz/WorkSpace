import json
from flask import Flask,request,jsonify
data = {
    'name' : 'myname',
    'age' : 100,
}
json_str = json.dumps(data)

json_str2 = jsonify(json_str)
print(json_str[0])