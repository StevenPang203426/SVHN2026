"""
数据探索与可视化工具
"""

import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import data_dir


def look_train_json():
    """查看训练集第一张图片的标注信息"""
    with open(data_dir['train_label'], 'r', encoding='utf-8') as f:
        content = json.loads(f.read())
    print(content['000000.png'])


def look_submit():
    """查看提交文件格式"""
    df = pd.read_csv(data_dir['submit_file'], sep=',')
    print(df.head(5))


def img_size_summary():
    """统计训练集图片尺寸分布"""
    sizes = []
    for img_path in glob(data_dir['train_data'] + '*.png'):
        img = Image.open(img_path)
        sizes.append(img.size)

    sizes = np.array(sizes)
    plt.figure(figsize=(10, 8))
    plt.scatter(sizes[:, 0], sizes[:, 1])
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Width-Height Summary')
    plt.show()


def bbox_summary():
    """统计训练集 bbox 尺寸分布"""
    marks = json.loads(open(data_dir['train_label'], 'r').read())
    bboxes = []

    for img, mark in marks.items():
        for i in range(len(mark['label'])):
            bboxes.append([
                mark['left'][i], mark['top'][i],
                mark['width'][i], mark['height'][i]
            ])

    bboxes = np.array(bboxes)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(bboxes[:, 2], bboxes[:, 3])
    ax.set_title('Bbox Width-Height Summary')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.show()


def label_summary():
    """统计训练集中各数字位数的图片数量"""
    marks = json.load(open(data_dir['train_label'], 'r'))
    dicts = {}

    for img, mark in marks.items():
        n = len(mark['label'])
        dicts[n] = dicts.get(n, 0) + 1

    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k, v in dicts:
        print(f'{k}个数字的图片数目: {v}')
