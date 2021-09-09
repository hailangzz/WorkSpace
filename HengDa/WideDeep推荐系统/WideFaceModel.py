import os
import argparse
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



# 设置模型基本参数···
def args_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--train_epochs", "-te", type=int, default=10,
        help="[default %(default)s] the train epochs of model training.",
        metavar="<TE>")
    parse.add_argument(
        "--batch_size", "-bs", type=int, default=32,
        help="[default: %(default)s] Batch size for training and evaluation.",
        metavar="<BS>")
    parse.add_argument(
        "--model_dir", "-mr", type=str, default="tmp",
        help="[default: %(default)s] The location of the model checkpoint files",
        metavar="<MD>")
    parse.add_argument(
        "--model_type", "-mt", type=str, default="wide_deep",
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar="<MT>")

    parse.set_defaults(
        train_epochs=10,
        batch_size=128,
        model_dir="widedeep_pandas_model",
        model_type="wide_deep")

    flags = parse.parse_args()

    return flags


def build_model_columns():
    # 定义连续值列
    age = tf.feature_column.numeric_column('age', normalizer_fn=lambda x: (x - 17) / 90, dtype=tf.float32)
    education_num = tf.feature_column.numeric_column('education-num', normalizer_fn=lambda x: (x - 1) / 16,
                                                     dtype=tf.int64)
    capital_gain = tf.feature_column.numeric_column('capital-gain', normalizer_fn=lambda x: (x - 0) / 99999,
                                                    dtype=tf.int64)
    capital_loss = tf.feature_column.numeric_column('capital-loss', normalizer_fn=lambda x: (x - 0) / 4356,
                                                    dtype=tf.float32)
    hours_per_week = tf.feature_column.numeric_column('hours-per-week', normalizer_fn=lambda x: (x - 1) / 99,
                                                      dtype=tf.int64)

    age_raw = tf.feature_column.numeric_column('age')
    capital_gain_raw = tf.feature_column.numeric_column('capital-gain')
    capital_loss_raw = tf.feature_column.numeric_column('capital-loss')
    education_num_raw = tf.feature_column.numeric_column('education-num')
    hours_per_week_raw = tf.feature_column.numeric_column('hours-per-week')

    # 定义离散值列
    # 分类列名称列表
    category_names = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]

    workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass',
                                                                          ['State-gov', 'Self-emp-not-inc', 'Private',
                                                                           'Federal-gov', 'Local-gov', '?',
                                                                           'Self-emp-inc', 'Without-pay',
                                                                           'Never-worked'], dtype=tf.string)
    education = tf.feature_column.categorical_column_with_vocabulary_list('education',
                                                                          ['Bachelors', 'HS-grad', '11th', 'Masters',
                                                                           '9th', 'Some-college', 'Assoc-acdm',
                                                                           'Assoc-voc', '7th-8th', 'Doctorate',
                                                                           'Prof-school', '5th-6th', '10th', '1st-4th',
                                                                           'Preschool', '12th'], dtype=tf.string)
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list('marital-status',
                                                                               ['Never-married', 'Married-civ-spouse',
                                                                                'Divorced', 'Married-spouse-absent',
                                                                                'Separated', 'Married-AF-spouse',
                                                                                'Widowed'], dtype=tf.string)
    occupation = tf.feature_column.categorical_column_with_vocabulary_list('occupation',
                                                                           ['Adm-clerical', 'Exec-managerial',
                                                                            'Handlers-cleaners', 'Prof-specialty',
                                                                            'Other-service', 'Sales', 'Craft-repair',
                                                                            'Transport-moving', 'Farming-fishing',
                                                                            'Machine-op-inspct', 'Tech-support', '?',
                                                                            'Protective-serv', 'Armed-Forces',
                                                                            'Priv-house-serv'], dtype=tf.string)
    relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship',
                                                                             ['Not-in-family', 'Husband', 'Wife',
                                                                              'Own-child', 'Unmarried',
                                                                              'Other-relative'], dtype=tf.string)
    race = tf.feature_column.categorical_column_with_vocabulary_list('race', ['White', 'Black', 'Asian-Pac-Islander',
                                                                              'Amer-Indian-Eskimo', 'Other'],
                                                                     dtype=tf.string)
    sex = tf.feature_column.categorical_column_with_vocabulary_list('sex', ['Male', 'Female'], dtype=tf.string)

    #     house = tf.feature_column.categorical_column_with_identity(
    #         'House', 2)
    #     if_milk = tf.feature_column.categorical_column_with_identity(
    #         'if_milk', 2)

    # 对购买总金额和最大一次购买inx进行分箱
    age_bin = tf.feature_column.bucketized_column(age_raw, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    caption_gain_bin = tf.feature_column.bucketized_column(capital_gain_raw, boundaries=[0, 1000, 2000, 3000, 10000])
    caption_loss_bin = tf.feature_column.bucketized_column(capital_loss_raw, boundaries=[0, 1000, 2000, 3000, 5000])
    education_num_bin = tf.feature_column.bucketized_column(education_num_raw, boundaries=[1, 9, 10, 12])
    hours_per_week_bin = tf.feature_column.bucketized_column(hours_per_week_raw, boundaries=[1, 40, 45])

    # 定义基础离散特征
    base_columns = [workclass, education, marital_status, occupation, relationship, race, sex]

    # 定义交叉组合特征
    cross_columns = [
        tf.feature_column.crossed_column([age_bin, caption_gain_bin], hash_bucket_size=10),
        tf.feature_column.crossed_column([age_bin, education_num_bin], hash_bucket_size=10),
        tf.feature_column.crossed_column([caption_loss_bin, hours_per_week_bin], hash_bucket_size=10),
        tf.feature_column.crossed_column([education_num_bin, hours_per_week_bin], hash_bucket_size=25)
    ]

    # wide部分的特征是0 1稀疏向量, 走LR, 采用全部离散特征和某些离散特征的交叉
    wide_columns = base_columns + cross_columns

    # 所有特征都走deep部分, 连续特征+离散特征onehot或者embedding
    deep_columns = [
        age, education_num, capital_gain, capital_loss, hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(occupation),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(race),
        tf.feature_column.indicator_column(sex)
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type, warm_start_from=None):
    """按照指定的模型生成估算器对象."""
    # 特征工程后的列对象组成的list
    wide_columns, deep_columns = build_model_columns()
    # deep 每一层全连接隐藏层单元个数, 4层每一层的激活函数是relu
    hidden_units = [100, 75, 50, 25]

    run_config = tf.estimator.RunConfig().replace(  # 将GPU个数设为0，关闭GPU运算。因为该模型在CPU上速度更快
        session_config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}),
        save_checkpoints_steps=100,
        keep_checkpoint_max=2)

    if model_type == 'wide':  # 生成带有wide模型的估算器对象
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':  # 生成带有deep模型的估算器对象
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(  # 生成带有wide和deep模型的估算器对象
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config,
            warm_start_from=warm_start_from)


def read_pandas(data_file):
    """pandas将数据读取内存"""
    assert os.path.exists(data_file), ("%s not found." % data_file)
    df = pd.read_csv(data_file).dropna()
    df['label'] = df['label'].map({'<=50K': 0, '>50K': 1})

    train, test = train_test_split(df, test_size=0.15, random_state=1)
    y_train = train.pop("label")
    y_test = test.pop("label")

    return train, test, y_train, y_test


def input_fn(X, y, shuffle, batch_size, predict=False):  # 定义估算器输入函数
    """估算器的输入函数."""
    if predict == True:
        # from_tensor_slices 从内存引入数据
        dataset = tf.data.Dataset.from_tensor_slices(X.to_dict(orient='list'))  # 创建dataset数据集
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))  # 创建dataset数据集

    if shuffle:  # 对数据进行乱序操作
        dataset = dataset.shuffle(buffer_size=64)  # 越大shuffle程度越大

    dataset = dataset.batch(batch_size)  # 将数据集按照batch_size划分
    dataset = dataset.prefetch(1)  # 预取数据,buffer_size=1 在多数情况下就足够了
    return dataset


def trainmain(train, y_train, test, y_test):
    flags = args_parse()

    shutil.rmtree(flags.model_dir, ignore_errors=True)  # 读取参数配置信息···
    model = build_estimator(flags.model_dir, flags.model_type)  # 生成估算器对象

    def train_input_fn():
        return input_fn(train, y_train, True, flags.batch_size, predict=False)

    def eval_input_fn():
        return input_fn(test, y_test, False, flags.batch_size, predict=False)

    # 在外部指定repeat 不在dataset中
    for n in range(flags.train_epochs):
        model.train(input_fn=train_input_fn)
        results = model.evaluate(input_fn=eval_input_fn)

        print('{0:-^30}'.format('evaluate at epoch %d' % ((n + 1))))
        # results 是一个字典
        print(pd.Series(results).to_frame('values'))

    # 导出模型
    export_model(model, "wd_tfserving")


def premain(predict_data):
    flags = args_parse()

    def predict_input_fn():  # 定义预测集样本输入函数
        return input_fn(predict_data, None, False, flags.batch_size, predict=True)  # 该输入函数按照batch_size批次，不使用乱序处理

    model2 = build_estimator(flags.model_dir, flags.model_type)  # 从检查点载入模型

    predictions = model2.predict(input_fn=predict_input_fn)

    # 数据下载
    predict_proba = list(map(lambda x: x['logistic'][0], predictions))
    predict_data['predict_proba'] = predict_proba
    predict_data.to_csv("milk_widedeep_res.csv", index=False)


def export_model(model, export_dir):
    features = {
        "age": tf.placeholder(dtype=tf.int32, shape=1, name='age'),
        "workclass": tf.placeholder(dtype=tf.string, shape=1, name='workclass'),
        "education": tf.placeholder(dtype=tf.string, shape=1, name='education'),
        "education-num": tf.placeholder(dtype=tf.int32, shape=1, name='education-num'),
        "marital-status": tf.placeholder(dtype=tf.string, shape=1, name='marital-status'),
        "occupation": tf.placeholder(dtype=tf.string, shape=1, name='occupation'),
        "relationship": tf.placeholder(dtype=tf.string, shape=1, name='relationship'),
        "race": tf.placeholder(dtype=tf.string, shape=1, name='race'),
        "sex": tf.placeholder(dtype=tf.string, shape=1, name='sex'),
        "capital-gain": tf.placeholder(dtype=tf.float64, shape=1, name='capital-gain'),
        "capital-loss": tf.placeholder(dtype=tf.int64, shape=1, name='capital-loss'),
        "hours-per-week": tf.placeholder(dtype=tf.int32, shape=1, name='hours-per-week')
    }

    example_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

    model.export_saved_model(export_dir, example_input_fn, as_text=True)

train, test, y_train, y_test=read_pandas(r'./data//train.txt')
trainmain(train, y_train, test, y_test)
