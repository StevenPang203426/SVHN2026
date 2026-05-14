"""
修复脚本：重新下载被覆盖的 JSON 标签文件
在你的本机运行（非沙盒环境）：
    cd Street_Character_Recognition
    python scripts/fix_labels.py
"""
import os
import requests

# 定位到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FILES = {
    "mchar_train.json": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json",
    "mchar_val.json": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json",
    "mchar_sample_submit_A.csv": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv",
}

for name, url in FILES.items():
    path = os.path.join(DATA_DIR, name)
    # 检查是否为空文件或不存在
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("Downloading %s ..." % name)
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)
            print("  OK (%d bytes)" % len(r.content))
        except Exception as e:
            print("  FAILED: %s" % e)
    else:
        print("  SKIP %s (already exists, %d bytes)" % (name, os.path.getsize(path)))

print("\nDone. You can now run: python train.py --config baseline")
