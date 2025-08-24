import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------- 1. 读取语料 ----------------
with open("爱情诗.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("语料长度：", len(text))
print("示例前200字：\n", text[:200])

# ---------------- 2. 构建字典 ----------------
chars = sorted(list(set(text)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

print("字典大小：", vocab_size)

# ---------------- 3. 构建训练数据 ----------------
seq_length = 40  # 输入序列长度（原来20，现在加长到40）
X, y = [], []

for i in range(0, len(text) - seq_length):
    X.append([char2idx[c] for c in text[i:i + seq_length]])
    y.append(char2idx[text[i + seq_length]])

X = np.array(X)
y = to_categorical(y, num_classes=vocab_size)

print("X 形状：", X.shape)
print("y 形状：", y.shape)

# ---------------- 4. 构建模型 ----------------
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=seq_length))
model.add(LSTM(256, return_sequences=False))   # 单元数增加
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# ---------------- 5. 训练模型 ----------------
model.fit(X, y, batch_size=128, epochs=20)   # epochs 增加

# ---------------- 6. 保存模型和字典 ----------------
model.save("爱情诗.h5")
with open("爱情诗char2idx.json", "w", encoding="utf-8") as f:
    json.dump(char2idx, f, ensure_ascii=False)
with open("爱情诗idx2char.json", "w", encoding="utf-8") as f:
    json.dump(idx2char, f, ensure_ascii=False)

print("✅ 模型和字典已保存")