#%%
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import evaluate
import matplotlib.pyplot as plt

#%%
import os

os.environ['http_proxy'] = 'http://127.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


# 加载训练数据、分词器、预训练模型以及评价方法
dataset = load_dataset('glue', 'sst2')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-mini', return_dict=True)
metric = evaluate.load('glue', 'sst2')
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')

#%%
# 对训练集进行分词
def tokenize(examples):
    return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length = 64)
dataset = dataset.map(tokenize, batched=True)
encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

# 将数据集格式化为torch.Tensor类型以训练PyTorch模型
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)

# 定义评价指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    accuracy_results = accuracy_metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)
    f1_results = f1_metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)
    return {'accuracy': accuracy_results['accuracy'], 'f1': f1_results['f1']}

#%%
# 定义一个回调函数来记录 loss 和最终 epoch 的验证集指标
class LossCallback(TrainerCallback):
    def __init__(self, num_epochs):
        self.losses = []
        self.steps = []
        self.final_metrics = {}
        self.num_epochs = num_epochs  # 存储训练的总 epoch 数
        self.last_epoch = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])
            self.steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 记录最后一个 epoch 的 metrics
        if state.epoch > self.last_epoch:
            self.final_metrics = metrics
            self.last_epoch = state.epoch

    def on_train_end(self, args, state, control, **kwargs):
        # 绘制 loss 曲线
        plt.plot(self.steps, self.losses)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('training_loss.png')  # 保存图像
        plt.show()  # 显示图像

        # 打印最终 epoch 的验证集指标
        print("\nFinal Epoch Validation Metrics:")
        for key, value in self.final_metrics.items():
            print(f"{key}: {value}")

# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    "no-need-to-care-about-this-folder",
    eval_steps=100,        # 定义每轮结束后进行评价
    learning_rate=3e-5,                 # 定义初始学习率
    per_device_train_batch_size=32,     # 定义训练批次大小
    per_device_eval_batch_size=32,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
    warmup_ratio=0.1,
    logging_steps=10,      # 每10步记录一次 loss
    logging_dir="./logs",   # 日志保存路径
)

# 创建 LossCallback 实例
loss_callback = LossCallback(args.num_train_epochs)

# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[loss_callback]  # 添加回调函数
)

trainer.train()

print("\nSaving model to bert-for-sst2")
model.save_pretrained("./bert-for-sst2")
model = BertForSequenceClassification.from_pretrained("./bert-for-sst2", return_dict=True)

preds = []
print("\nPractical task examples:\n")
example = dataset["train"][:5]
example_loader = DataLoader(encoded_dataset["train"], 5)
with torch.no_grad():
    for i, batch in enumerate(example_loader):
        if i == 0:
            pred = model(**batch)["logits"]
            preds.extend(pred.squeeze().argmax(-1).tolist())
            break

texts = []
for key in example:
    if type(example[key][0]) == str:
        texts.append(example[key])
for i in range(len(preds)):
    for text in texts:
        print(text[i], end = '     ')
    print(f"\nPrediction: {preds[i]}, {'which is correct.' if preds[i] == example['label'][i] else 'which is wrong.'}\n")
# %%
# 在训练结束后手动调用 evaluate 方法
metrics = trainer.evaluate()
loss_callback.final_metrics = metrics
print("\nFinal Evaluation Metrics:")
for key, value in loss_callback.final_metrics.items():
    print(f"{key}: {value}")
# %%
