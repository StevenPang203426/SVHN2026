"""
数据集下载与解压
"""

import os
import zipfile

import pandas as pd
import requests


def download_and_extract(csv_path='./mchar_data_list_0515.csv', dataset_path='./data'):
    """
    根据 CSV 中的链接下载数据集并解压。

    Args:
        csv_path: 包含下载链接的 CSV 文件路径
        dataset_path: 数据集存放目录
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] 未找到链接文件 {csv_path}，跳过下载。")
        print("请手动下载数据并放入 data/ 目录。")
        return

    links = pd.read_csv(csv_path)
    print(f"数据集目录：{dataset_path}")

    os.makedirs(dataset_path, exist_ok=True)

    # 下载文件
    for i, link in enumerate(links['link']):
        file_name = links['file'][i]
        file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_path):
            print(f"正在下载 {file_name} ...")
            try:
                response = requests.get(link, stream=True, timeout=30)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"  ✓ {file_name} 下载完成")
            except Exception as e:
                print(f"  ✗ {file_name} 下载失败: {e}")
        else:
            print(f"  - {file_name} 已存在，跳过")

    # 解压文件
    zip_list = ['mchar_train', 'mchar_test_a', 'mchar_val']
    for name in zip_list:
        target = os.path.join(dataset_path, name)
        zip_path = os.path.join(dataset_path, f"{name}.zip")
        if not os.path.exists(target) and os.path.exists(zip_path):
            print(f"正在解压 {name}.zip ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=dataset_path)
            print(f"  ✓ {name} 解压完成")

    print("数据准备完成。")


if __name__ == '__main__':
    download_and_extract()
