# coding: UTF-8
#%%
import time
import torch
import numpy as np
from importlib import import_module
from train import train, init_network
from predict import load_dataset,final_predict
from utils import build_dataset, build_iterator, get_time_dif
import random
import csv  # 导入 csv 模块


def load_random_test_samples(file_path, sample_num=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去除空行、首尾空格，并过滤掉太短的行
    lines = [line.strip() for line in lines if line.strip()]
    return random.sample(lines, sample_num)


class TransformerPredict:
    def __init__(self, config, model, vocab):
        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, text):
        content = load_dataset(text, self.vocab)
        predict_iter = build_iterator(content, self.config, predict=True)
        self.config.n_vocab = len(self.vocab)
        result = final_predict(self.config, self.model, predict_iter)
        return result


def predict_all(config, model, vocab, test_file_path, output_file_path):
    """
    对所有测试集样本进行预测，并保存预测错误的样本。
    """
    model.eval()  # 设置为评估模式
    correct_count = 0
    total_count = 0
    error_list = []

    with open(test_file_path, 'r', encoding='utf-8') as f, \
            open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['text', 'true_label', 'predicted_label'])  # 写入表头

        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')  # 假设数据以制表符分隔，第一列是文本，第二列是标签
            true_label = int(label)  # 将 true_label 转换为整数

            # 预测
            tp = TransformerPredict(config, model, vocab)
            predicted_label_text = tp.predict([text])[0]  # 预测的标签是文本
            predicted_label = config.class_list.index(predicted_label_text)  # 转换为数字

            total_count += 1
            if predicted_label == true_label:
                correct_count += 1
            else:
                error_list.append((text, true_label, predicted_label_text))
                writer.writerow([text, true_label, predicted_label_text])  # 写入错误样本

    accuracy = correct_count / total_count
    print(f"Total samples: {total_count}, Correct: {correct_count}, Accuracy: {accuracy:.4f}")
    print(f"Error samples saved to {output_file_path}")


#%%
dataset = 'THUCNews'  # 数据集
embedding = 'random'
model_name = 'Transformer'

x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
vocab, train_data, dev_data, test_data = build_dataset(config, False)
train_iter = build_iterator(train_data, config, False)
dev_iter = build_iterator(dev_data, config, False)
test_iter = build_iterator(test_data, config, False)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# 训练
config.n_vocab = len(vocab)
model = x.Model(config).to(config.device)
if model_name != 'Transformer':
    init_network(model)
# print(model.parameters)
train(config, model, train_iter, dev_iter, test_iter)

# 分类（预测）的例子
tp = TransformerPredict(config, model, vocab)
test_file_path = r"./THUCNews/data/test.txt"  # 你的测试文本路径
test = load_random_test_samples(test_file_path, 5)
results = tp.predict(test)
for i, j in enumerate(results):
    print('text:{}'.format(test[i]), '\t', 'label:{}'.format(j))

# 对所有测试集样本进行预测
test_file_path = r"./THUCNews/data/test.txt"  # 你的测试文本路径
output_file_path = r"./THUCNews/data/error_samples.csv"  # 错误样本保存路径
predict_all(config, model, vocab, test_file_path, output_file_path)

# %%