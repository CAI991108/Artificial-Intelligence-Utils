import torch
import random
import matplotlib.pyplot as plt  # 导入绘图库
import os

# 创建保存图片的目录
image_dir = "./results/images/skipgram_plots"
os.makedirs(image_dir, exist_ok=True)

# analogical reasoning task dataset
with open('./evaluation/analogical reasoning task.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Calculate cosine similarity
def cos(vec1, vec2):
    vec1 = vec1 / vec1.norm(p=2)
    vec2 = vec2 / vec2.norm(p=2)
    return torch.dot(vec1, vec2).item()

# Find the nearest neighbor (excluding given words)
def knn(target_vec, embeddings, exclude_words, k=1):
    similarities = []
    for word, vec in embeddings.items():
        if word not in exclude_words:
            similarity = cos(target_vec, vec)
            similarities.append((word, similarity))
    # Sort by similarity in descending order
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:k]

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

# 绘制向量图并保存
def plot_vectors(word1, vec1, word2, vec2, filename):
    plt.figure(figsize=(10, 6))
    dimensions = range(len(vec1))  # 假设向量的维度是连续的
    plt.plot(dimensions, vec1.numpy(), label=f"{word1}", marker='o')
    plt.plot(dimensions, vec2.numpy(), label=f"{word2}", marker='x')
    plt.xlabel("Dimensions")
    plt.ylabel("Values")
    plt.title(f"Comparison of {word1} and {word2}")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Load pre-trained embeddings
embeddings = load_embeddings('./results/skipgram.vec')
embed_vocabulary = set(embeddings.keys())

# Output file
output_file = "./results/analogical_reasoning_results.txt"

# Randomly pick 10 examples
num_exist = 0
correct = []
random.seed(42)  # Set random seed for reproducibility
random_lines = random.sample(lines, 100)  # Randomly pick 10 examples

with open(output_file, 'w', encoding='utf-8') as f:
    def dual_print(content, file=f):
        print(content)          # Output to console
        print(content, file=file)  # Write to file

    dual_print("Analogical Reasoning Evaluation Results\n" + "="*40)

    for idx, line in enumerate(random_lines):
        words = line.strip().split()
        if len(words) != 4:
            continue  # Skip invalid lines
        word_a, word_b, word_c, word_d = words
        word_a, word_b, word_c, word_d = word_a.lower(), word_b.lower(), word_c.lower(), word_d.lower()

        # Check if all words exist in the vocabulary
        if all(word in embed_vocabulary for word in [word_a, word_b, word_c, word_d]):
            num_exist += 1
            vec_a, vec_b, vec_c = embeddings[word_a], embeddings[word_b], embeddings[word_c]
            # Calculate the target vector: vec_d = vec_b - vec_a + vec_c
            target_vec = vec_b - vec_a + vec_c
            # Find the nearest neighbor excluding the given words
            nearest_neighbors = knn(target_vec, embeddings, exclude_words={word_a, word_b, word_c, word_d}, k=1)
            predicted_word = nearest_neighbors[0][0] if nearest_neighbors else None

            # Check if the prediction is correct
            if predicted_word == word_d:
                correct.append((word_a, word_b, word_c, word_d, predicted_word))

            dual_print(f"Question: {word_a}:{word_b}::{word_c}:{word_d}")
            dual_print(f"Predicted: {predicted_word}, Correct: {word_d}")

            # 绘制并保存图像
            if predicted_word and predicted_word in embeddings:
                vec_predicted = embeddings[predicted_word]
                vec_correct = embeddings[word_d]
                plot_filename = os.path.join(image_dir, f"pair_{idx + 1}_{predicted_word}_vs_{word_d}.png")
                plot_vectors(predicted_word, vec_predicted, word_d, vec_correct, plot_filename)
        # Skip questions where any word is out of vocabulary
        else:
            continue

    # Print final results
    dual_print(f"\nFor {num_exist} analogy questions with all words presented in the vocabulary, {len(correct)} of them are correctly answered.")