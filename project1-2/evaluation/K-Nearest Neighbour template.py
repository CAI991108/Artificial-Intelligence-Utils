import torch

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''
# %%
# query word list
word_list = ["july", "reliable", "play", "willing", "good", "very", "patient", "concerned", "important", "powerful", "quickly", "generally", "gradually", "happy", "able", "close", "near",
             "saturday", "friend", "company", "road", "plane", "war", "politics", "building", "student", "university", "realm", "china", "experience", "police",
             "give", "create", "tell", "become", "lack", "win", "help", "gain", "get", "take", "use", "set", "find", "increase",
             "difficult", "go", "man", "ten", "year"]

# Load embeddings from cbow.vec
def load_embeddings(filepath):
    embeddings = {}
    expected_dim = None  # 嵌入向量的维度
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            # 跳过第一行
            if i == 0:
                _, expected_dim = map(int, line.strip().split())
                continue
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]])
            # 检查向量维度是否匹配
            if vector.size(0) != expected_dim:
                print(f"Skipping word '{word}' due to mismatched dimensions: {vector.size(0)}")
                continue
            embeddings[word] = vector
    return embeddings

# Calculate cosine similarity
def cos(vec1, vec2):
    vec1 = vec1 / vec1.norm(p=2)
    vec2 = vec2 / vec2.norm(p=2)
    return torch.dot(vec1, vec2).item()

# Find k-nearest neighbors
def knn(word, embeddings, k=5):
    if word not in embeddings:
        return [], []
    
    query_vec = embeddings[word]
    similarities = []
    
    for other_word, other_vec in embeddings.items():
        if other_word != word:
            similarity = cos(query_vec, other_vec)
            similarities.append((other_word, similarity))
    
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    knn_words = [item[0] for item in similarities[:k]]
    knn_sims = [item[1] for item in similarities[:k]]
    return knn_words, knn_sims

# Evaluation process for each word
def evaluate(word, embeddings, k=5):
    knn_words, knn_sims = knn(word, embeddings, k)
    result = f"\nQuery: {word}\n"
    for i in range(len(knn_words)):
        result += f"The similarity of {knn_words[i]} is {knn_sims[i]:.4f}.\n"
    return result

# %%
# Main process
def main():
    # Load embeddings
    embeddings = load_embeddings('./results/cbow.vec')
    
    # Open a file to save results
    with open('./results/evaluation_results.txt', 'w') as f:
        for word in word_list:
            result = evaluate(word, embeddings, k=5)
            print(result)  # Print to console
            f.write(result)  # Save to file

if __name__ == "__main__":
    main()
