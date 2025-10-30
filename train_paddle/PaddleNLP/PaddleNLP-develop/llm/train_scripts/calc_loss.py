import math
import numpy as np 
import matplotlib.pyplot as plt 
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def get_paddle_loss(data):
    loss_item_paddle = []
    for item in data:
        if "loss: " in item:
            try:
                loss = float(item.split("loss: ")[1].split(",")[0])
                loss_item_paddle.append(loss)
            except:
                continue
    return loss_item_paddle

def get_torch_loss(data):
    loss_item_torch = []
    for item in data:
        if "loss\': " in item:
            item = eval(item)
            loss_item_torch.append(item['loss'])
    return loss_item_torch


if __name__ == '__main__':
    with open("/data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/new-cpt-YuLan_Base_12B-pile-lr_1e6-bs_device_1-acc_1-gpu_16-warmup_0-len_1024.log","r") as f:
        data = f.readlines()
    loss_item_paddle = get_paddle_loss(data)

    with open("/data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/new-cpt-YuLan_Base_12B-pile-lr_1e6-bs_device_1-acc_1-gpu_16-warmup_0-len_1024-test.log","r") as f:
        data = f.readlines()
    # loss_item_torch = get_torch_loss(data)
    loss_item_torch = get_paddle_loss(data)

    n = 100
    loss_item_paddle = loss_item_paddle[:n]
    loss_item_torch = loss_item_torch[:n]
    paddle_avg_loss = sum(loss_item_paddle) / n
    torch_avg_loss = sum(loss_item_torch) / n
    print('Paddle Loss: {}'.format(paddle_avg_loss))
    print('Torch Loss: {}'.format(torch_avg_loss))
    print('Related Delta: {}'.format(math.fabs(paddle_avg_loss - torch_avg_loss)))



    # 绘制数据点
    x = list(range(len(loss_item_paddle)))
    plt.figure(figsize=(10, 6))  # 设定图形大小
    plt.plot(x, loss_item_paddle, label="Paddle", color='skyblue', linewidth=2, marker='o', markersize=4)
    plt.plot(x, loss_item_torch, label="PyTorch", color='lightcoral', linewidth=2, marker='x', markersize=4)
    plt.xlabel('Training Steps', fontsize=18)  # 调整字体大小
    plt.ylabel('Training Loss', fontsize=18)  # 调整字体大小
    plt.legend(fontsize=16)  # 调整字体大小
    plt.tick_params(axis='x', labelsize=16)  # x轴刻度标签字体大小
    plt.tick_params(axis='y', labelsize=16)  # y轴刻度标签字体大小
    plt.grid(True)  # 添加网格线
    # plt.savefig('tianshu_torch-tianshu_paddle.jpg')
    # plt.savefig('tianshu_torch-kunlun_paddle.jpg')
    plt.savefig('loss_test.jpg')