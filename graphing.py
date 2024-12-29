import matplotlib.pyplot as plt
import pandas

# 2-species heatmap
def heatmap(data,title,xlabel,ylabel,path):
    plt.imshow(data, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
    plt.colorbar()                                                                  # 添加颜色条
    plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
    plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置y轴的刻度
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()