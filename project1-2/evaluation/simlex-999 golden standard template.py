import torch
from scipy import stats
import os, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 创建输出文件
output_file = "./results/simlex_result.txt"  # 文件保存路径

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

# spearman correlation using function from scipy library. you can also implement by yourself.
def spearman(W, q):
    return torch.tensor(stats.spearmanr(q, W)[0])

def cos(vec1, vec2):
    vec1 = vec1 / vec1.norm(p=2)
    vec2 = vec2 / vec2.norm(p=2)
    raw_score = torch.dot(vec1, vec2).item()
    
    # 正则化到0-10范围（原始范围[-1,1] -> [0,10]）
    normalized_score = (raw_score + 1) * 5  # 等价于 ((x - (-1)) / (1 - (-1))) * 10
    return normalized_score

# Load embeddings from file
def load_embeddings(filepath):
    embeddings = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            # Skip the first line if it contains metadata
            if i == 0 and len(line.split()) == 2:
                continue
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]])
            embeddings[word] = vector
    return embeddings

# simlex-999 golden standard
with open('./evaluation/simlex-999.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 使用with语句自动管理文件资源
with open(output_file, 'w', encoding='utf-8') as f:
    # 重定向打印到文件和屏幕的辅助函数
    def dual_print(content, file=f):
        print(content)          # 输出到控制台
        print(content, file=file)  # 写入文件

    dual_print("SimLex-999 Evaluation Results\n" + "="*40)

    # Load pre-trained embeddings
    embeddings = load_embeddings('./results/cbow.vec')
    embed_vocabulary_set = set(embeddings.keys())  # 改用集合直接检查存在性

    standard = []
    calculated = []

    random.seed(42)
    random_lines = random.sample(lines, 100)  # 随机选择 20 对单词对

    for line in random_lines:
        word1, word2, std = tuple(line.strip().split('\t'))  # 注意：SimLex-999使用制表符分隔
        word1, word2 = word1.lower(), word2.lower()  # 统一小写处理
        std = float(std)  # 直接转为浮点数
        
        dual_print(f"\nWord pair: {word1} and {word2}")

        # 直接检查词是否在嵌入集合中
        if word1 in embed_vocabulary_set and word2 in embed_vocabulary_set:
            vec1 = embeddings[word1]
            vec2 = embeddings[word2]
            cal = cos(vec1, vec2)
            dual_print(f"Standard similarity: {std:.2f}, Calculated: {cal:.4f}")
            standard.append(std)
            calculated.append(cal)
        else:
            dual_print("At least one of the words is Out-of-Vocabulary.")

    # Calculate Spearman correlation
    if len(standard) > 0 and len(calculated) > 0:
        correlation = spearman(standard, calculated).item()
        dual_print("\nThe spearman correlation between the standard and calculated similarity is: {:.4f}".format(correlation))
    else:
        dual_print("\nNo valid word pairs found in the vocabulary.")