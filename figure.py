import numpy as np
import random as rd
import os
import time
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import rc
from datetime import datetime

# Function to plot the dynamics of x and y.
def xydynamics(x,y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('time')
    ax.set_ylabel('conc')
    # ax.set_title('Dynamics of x and y')
    plt.show()
    
# naming by date and number
def figurename(path, date_format="%y%m%d"):
    """
    :param folder_path: 存储图片的文件夹路径
    :param date_format: 日期格式
    :return: 新的文件名
    """
    today = datetime.now().strftime(date_format)
    existing_files = os.listdir(path)
    today_files = [f for f in existing_files if f.startswith(today)]
    numbers = [int(f.split("-")[1].split("_")[0]) for f in today_files if "-" in f and "_" in f]
    new_number = max(numbers) + 1 if numbers else 1
    new_filename = f"{today}-{new_number}"
    return new_filename

def savefigure(figure, path, filename, format='png'):
    """
    :param figure: 需要保存的图片
    :param path: 图片存储路径
    :param filename: 图片名称
    :param format: 图片格式，默认是png
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    figure.savefig(os.path.join(path, filename), dpi=300, bbox_inches='tight')
    plt.close(figure)  # 关闭图形以释放内存