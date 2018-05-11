import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib.request
import tensorflow as tf

# 下载数据
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?'
        )
    return filename

filename = maybe_download('text8.zip', 31344016)

# 读取数据
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size', len(words))

# 创建vocabulary词汇表
vocabulary_size = 50000

# dic,reverse_dic:字符和数字的字典和反字典
# count: 单词的出现频率
# data: 单词编号
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))     # 记录出现次数前50000的单词
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print("Most common words (+UNK)", count[:5])
print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
# batch_size: batch的大小  num_skips: 对每个单词生成多少样本  skip_window: 单词最远可联系的距离
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):                               # 填充buffer数据，后续数据会替换前面数据
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):            # 第一层循环，每次对一个单词生成样本
        target = skip_window
        target_to_avoid = [skip_window]                 # 存放已使用过的target
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span-1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])