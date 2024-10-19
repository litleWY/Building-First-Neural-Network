# predict.py - 预测模型
import torch

def predict(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            print(f'Predicted class: {pred.view(-1).tolist()}')
            break  # 只预测一个批次
