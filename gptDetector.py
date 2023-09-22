import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# 读取CSV文件
test_data = pd.read_csv('/home/xjyin/workspace/chatgpt-comparison-detection/data/test_data_2k.csv')

tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta-chinese")
model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta-chinese")

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型移到设备上
model.to(device)

# 对文本进行预处理和预测
def predict0(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(input_ids)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

def predict1(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=None, truncation=False)
    logits = []
    for i in range(0, input_ids.size(1), 512):
        input_segment = input_ids[:, i:i+512]
        outputs = model(input_segment)
        logits.append(outputs.logits)
    combined_logits = torch.sum(torch.stack(logits), dim=0)
    predictions = torch.argmax(combined_logits, dim=-1)
    return predictions.item()

def predict(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=None, truncation=False)
    logits = []
    for i in range(0, input_ids.size(1), 512):
        input_segment = input_ids[:, i:i+512].to(device)
        outputs = model(input_segment)
        logits.append(outputs.logits.squeeze(0))

    # 计算权重
    weight_last_segment = input_ids.size(1) % 512 / 512
    weights = [1.0] * (len(logits) - 1) + [weight_last_segment]
    logits_tensor = torch.stack(logits)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1).to(logits_tensor.device)

    combined_logits = torch.sum(logits_tensor * weights_tensor, dim=0) / torch.sum(weights_tensor)
    predictions = torch.argmax(combined_logits)
    return predictions.item()

test_data['predictions'] = test_data['text'].apply(predict)
test_data.to_csv('/home/xjyin/workspace/chatgpt-comparison-detection/data/test_data_2k_pred.csv')
# 计算模型预测精度
accuracy = accuracy_score(test_data['label'], test_data['predictions'])
print("模型预测精度：", accuracy)
