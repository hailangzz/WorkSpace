import os
import argparse
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


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
        train_epochs=5,
        batch_size=12,
        model_dir="widedeep_pandas_model",
        model_type="wide_deep")

    flags = parse.parse_args()

    return flags


def build_model_columns():
    # 定义连续值列
    area = tf.feature_column.numeric_column('jzmj', normalizer_fn=lambda x: x / 150, dtype=tf.float32)

    #     area_raw = tf.feature_column.numeric_column('建筑面积')

    # 定义离散值列
    # 分类列名称列表
    category_names = ["hxjg", "cjsj", "szlc", "zlc", "lx", "dt", "zx", "csid", "qy", "sq"]

    hxjg_class = tf.feature_column.categorical_column_with_hash_bucket("hxjg",
                                                                       hash_bucket_size=8)  # 因为离散特征取值总的数目，不会超过512
    cjsj_class = tf.feature_column.categorical_column_with_hash_bucket("cjsj",
                                                                       hash_bucket_size=12)  # 因为离散特征取值总的数目，不会超过512
    szlc_class = tf.feature_column.categorical_column_with_hash_bucket("szlc",
                                                                       hash_bucket_size=14)  # 因为离散特征取值总的数目，不会超过512
    zlc_class = tf.feature_column.categorical_column_with_hash_bucket("zlc",
                                                                      hash_bucket_size=14)  # 因为离散特征取值总的数目，不会超过512
    lx_class = tf.feature_column.categorical_column_with_hash_bucket("lx", hash_bucket_size=6)  # 因为离散特征取值总的数目，不会超过512
    dt_class = tf.feature_column.categorical_column_with_hash_bucket("dt", hash_bucket_size=3)  # 因为离散特征取值总的数目，不会超过512
    zx_class = tf.feature_column.categorical_column_with_hash_bucket("zx", hash_bucket_size=3)  # 因为离散特征取值总的数目，不会超过512
    csid_class = tf.feature_column.categorical_column_with_hash_bucket("csid",
                                                                       hash_bucket_size=14)  # 因为离散特征取值总的数目，不会超过512
    qy_class = tf.feature_column.categorical_column_with_hash_bucket("qy", hash_bucket_size=36)  # 因为离散特征取值总的数目，不会超过512
    sq_class = tf.feature_column.categorical_column_with_hash_bucket("sq",
                                                                     hash_bucket_size=48)  # 因为离散特征取值总的数目，不会超过512

    #     # 对购买总金额和最大一次购买inx进行分箱
    #     area_bin = tf.feature_column.bucketized_column(age_raw, boundaries=[55,75,125,150])

    # 定义基础离散特征
    base_columns = [hxjg_class, cjsj_class,szlc_class, zlc_class, lx_class, dt_class, zx_class, csid_class,
                    qy_class, sq_class]

    #     # 定义交叉组合特征
    #     cross_columns = [
    #             tf.feature_column.crossed_column([age_bin, caption_gain_bin], hash_bucket_size=10),
    #             tf.feature_column.crossed_column([age_bin, education_num_bin], hash_bucket_size=10),
    #             tf.feature_column.crossed_column([caption_loss_bin, hours_per_week_bin], hash_bucket_size=10),
    #             tf.feature_column.crossed_column([education_num_bin, hours_per_week_bin], hash_bucket_size=25)
    #     ]

    # wide部分的特征是0 1稀疏向量, 走LR, 采用全部离散特征和某些离散特征的交叉
    #     wide_columns = base_columns + cross_columns
    wide_columns = base_columns

    # 所有特征都走deep部分, 连续特征+离散特征onehot或者embedding
    deep_columns = [area,
                    tf.feature_column.embedding_column(hxjg_class, 4),
                    tf.feature_column.embedding_column(cjsj_class, 6),
                    tf.feature_column.embedding_column(szlc_class, 12),
                    tf.feature_column.embedding_column(zlc_class, 12),
                    tf.feature_column.embedding_column(lx_class, 3),
                    tf.feature_column.embedding_column(dt_class, 2),
                    tf.feature_column.embedding_column(zx_class, 2),
                    tf.feature_column.embedding_column(csid_class, 14),
                    tf.feature_column.embedding_column(qy_class, 24),
                    tf.feature_column.embedding_column(sq_class, 36)
                    ]

    return wide_columns, deep_columns



def build_estimator(model_dir, model_type, warm_start_from=None):
    """按照指定的模型生成估算器对象."""
    # 特征工程后的列对象组成的list
    wide_columns, deep_columns = build_model_columns()
    # deep 每一层全连接隐藏层单元个数, 4层每一层的激活函数是relu
    hidden_units = [100, 75, 50, 25]

    run_config = tf.estimator.RunConfig().replace(  # 将GPU个数设为0，关闭GPU运算。因为该模型在CPU上速度更快
        session_config=tf.ConfigProto(device_count={'GPU': 0}),
        save_checkpoints_steps=100,
        keep_checkpoint_max=2)

    if model_type == 'wide':  # 生成带有wide模型的估算器对象
        return tf.estimator.LinearRegressor(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':  # 生成带有deep模型的估算器对象
        return tf.estimator.DNNRegressor(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedRegressor(  # 生成带有wide和deep模型的估算器对象
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config,
            warm_start_from=warm_start_from)


def read_pandas(data_file):
    """pandas将数据读取内存"""
    assert os.path.exists(data_file), ("%s not found." % data_file)
    df = pd.read_csv(r'./房屋有效特征数据.csv',dtype={'cjsj':object,'szlc':object,'zlc':object,'csid':object}).dropna()
    df['label'] = df['cjzj'] / 2000000
    df.drop(['xqid', 'cjzj'], axis=1, inplace=True)
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
    export_model(model, "tfserving")


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
        "hxjg": tf.placeholder(dtype=tf.string, shape=1, name='hxjg'),
        "jzmj": tf.placeholder(dtype=tf.int32, shape=1, name='jzmj'),
        "cjsj": tf.placeholder(dtype=tf.string, shape=1, name='cjsj'),
        "szlc": tf.placeholder(dtype=tf.string, shape=1, name='szlc'),
        "zlc": tf.placeholder(dtype=tf.string, shape=1, name='zlc'),
        "lx": tf.placeholder(dtype=tf.string, shape=1, name='lx'),
        "dt": tf.placeholder(dtype=tf.string, shape=1, name='dt'),
        "zx": tf.placeholder(dtype=tf.string, shape=1, name='zx'),
        "csid": tf.placeholder(dtype=tf.string, shape=1, name='csid'),
        "qy": tf.placeholder(dtype=tf.string, shape=1, name='qy'),
        "sq": tf.placeholder(dtype=tf.string, shape=1, name='sq')
    }

    example_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

    model.export_saved_model(export_dir, example_input_fn, as_text=True)

train, test, y_train, y_test=read_pandas(r'./房屋有效特征数据.csv')
trainmain(train, y_train, test, y_test)