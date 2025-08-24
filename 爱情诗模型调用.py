from tensorflow.keras.models import load_model
import numpy as np
import json

# ---------------- 1. 加载模型和字典 ----------------
model = load_model("爱情诗")

with open("爱情诗char2idx.json", "r", encoding="utf-8") as f:
    char2idx = json.load(f)
with open("爱情诗idx2char.json", "r", encoding="utf-8") as f:
    idx2char = json.load(f)

vocab_size = len(char2idx)
seq_length = 20  # 必须和训练时一致

# ---------------- 2. 定义生成函数 ----------------
def generate_text(model, seed_text, length=100, temperature=1.0):
    """
    model        : 训练好的LSTM模型
    seed_text    : 初始种子文本 (最好 >= seq_length)
    length       : 要生成的字符数量
    temperature  : 控制随机性 (0.5更保守, 1.0正常, >1更随机)
    """
    text = seed_text

    for _ in range(length):
        # 取最后 seq_length 个字符
        input_seq = text[-seq_length:]
        input_idx = [char2idx.get(c, 0) for c in input_seq]

        # 补齐到固定长度
        input_idx = np.pad(input_idx, (seq_length - len(input_idx), 0))

        # 转换为模型输入格式 (1, seq_length)
        x_pred = np.array([input_idx])

        # 预测下一个字符的概率分布
        preds = model.predict(x_pred, verbose=0)[0]

        # 加入温度系数，控制随机性
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # 按概率采样一个字符
        next_idx = np.random.choice(range(vocab_size), p=preds)
        next_char = idx2char[str(next_idx)]

        text += next_char

    return text

# ---------------- 3. 示例调用 ----------------
seed = "爱"  # 种子文本
result = generate_text(model, seed, length=100, temperature=0.8)

print("【种子】:", seed)
print("【生成】:", result.replace(" ", "").replace("\n", ""))