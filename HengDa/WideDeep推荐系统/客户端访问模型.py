import requests
data = {'age': 50,
        'workclass': 'Self-emp-not-inc',
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 13
        }
res = requests.post("http://192.168.234.132:8503/v1/models/wd_tfserving:predict",
                    json={"instances": [data], "signature_name": "predict"})
print(res.text)