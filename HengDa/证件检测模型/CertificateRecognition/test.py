import os

def get_class_weight(origin_path = r'F:\工作文档\keras迁移学习\CertificateRecognition\imbalance_train'):
    class_weight={}
    sample_class_num={}
    sample_directory = os.listdir(origin_path)
    print(sample_directory)
    for dir_index in range(len(sample_directory)):
        sample_class_num[dir_index]=len(os.listdir(os.path.join(origin_path,sample_directory[dir_index])))
        class_weight[dir_index] = 1

    for class_key in class_weight:
        class_weight[class_key]=(1/sample_class_num[class_key]*sum(sample_class_num.values()))

    return class_weight

class_weight=get_class_weight(origin_path = r'F:\工作文档\keras迁移学习\CertificateRecognition\imbalance_train')
print(class_weight)
