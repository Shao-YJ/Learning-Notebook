import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np

# 数据
text = "the quick brown fox jumps over the lazy dog the cat sleeps"
words = text.split()
vocab = set(words)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocab)

# 生成训练数据
def create_skipgram_data(words, window_size=2):
    data = []
    for i, word in enumerate(words):
        # 确定上下文窗口边界
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_word = words[j]
                data.append((word, context_word))
    return data

window_size = 2
training_data = create_skipgram_data(words, window_size=window_size)
print("训练样本(中心词->上下文词):")
for pair in training_data[:10]:
    print(f"{pair[0]} -> {pair[1]}")
    
# 定义Skip-gram模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, center_words):
        embeds = self.embeddings(center_words)
        out = self.out(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
embedding_dim = 10
learning_rate = 0.01
epochs = 500

model = SkipGramModel(vocab_size, embedding_dim)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(epochs):
    total_loss = 0
    for center_word, context_word in training_data:
        center_idx = torch.tensor([word_to_idx[center_word]], dtype=torch.long)
        context_idx = torch.tensor([word_to_idx[context_word]], dtype=torch.long)
        
        model.zero_grad()
        log_probs = model(center_idx)
        loss = loss_function(log_probs, context_idx)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    losses.append(total_loss)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss}")
        
print("\n学习到的词向量:")
with torch.no_grad():
    for word in vocab:
        idx = word_to_idx[word]
        vector = model.embeddings(torch.tensor(idx)).numpy()
        print(f"{word}: {vector}")
        

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print("\n计算相似度：")
test_word = "the"
with torch.no_grad():
    test_embedding = model.embeddings(torch.tensor(word_to_idx[test_word])).numpy()

    similarities = {}
    for word in vocab:
        if word != test_word:
            other_embedding = model.embeddings(torch.tensor(word_to_idx[word])).numpy()
            sim = cosine_similarity(test_embedding, other_embedding)
            similarities[word] = sim

    # 按相似度排序
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print(f"与 '{test_word}' 最相似的词：")
    for word, sim in sorted_sims:
        print(f"  {word}: {sim:.4f}")       