import functools
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

train_file_path = "./titanic/train.csv"
test_file_path = "./titanic/test.csv"

# 标签列
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

def get_dataset(file_path):
    """
    构建tensorflow的数据集格式
    """
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
    return dataset

# 将train和test的csv，分别加载成tensorflow的对象的格式
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

# 测试一个批次
examples, labels = next(iter(raw_test_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

categorical_columns = []
numerical_columns = []

# 分类数据的码表
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

#标准化连续数据···
def process_continuous_data(mean, data):
    # 标准化数据的函数
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])

# 提前算好的均值
MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

#序列化分类数据···
for feature, vocab in CATEGORIES.items():
    # 提供码表的特征输入
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(
        feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)


preprocessing_layer = tf.keras.layers.DenseFeatures(
    categorical_columns+numerical_columns)

# 构建一个DNN模型h(g(f(x)))
model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae'])

train_data = raw_train_data.shuffle(500)
test_data = raw_test_data

model.fit(train_data, epochs=200)

# model.save('./my_model.h5')

predictions = model.predict(test_data)
predictions[:10]









