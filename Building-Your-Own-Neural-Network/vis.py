# visualization.py - 可视化工具
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('./loss', exist_ok=True)


def show_images(images, labels, classes):
    """
    展示一组图像及其标签。
    :param images: 图像张量列表
    :param labels: 图像对应的标签列表
    :param classes: 标签类别名称列表
    """
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image / 2 + 0.5  # 反归一化
        npimg = image.numpy()
        axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    plt.show()

def plot_training_curves(train_losses, train_accuracies):
    """
    绘制训练过程中的损失曲线和准确率曲线。
    :param train_losses: 训练过程中的损失列表
    :param train_accuracies: 训练过程中的准确率列表
    """
    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration (per 100 steps)')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # 绘制训练准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Curve')
    plt.legend()

    # 保存训练损失和准确率曲像
    plt.savefig('./loss/training_curves.png')
    
    plt.show()
