import tarfile
from pathlib import Path

tar_path = Path(r"C:\Users\qiyin\supervised\Agriculture-Vision-2021.tar.gz")
extract_dir = Path(r"C:\Users\qiyin\supervised\Agriculture-Vision-2021")

if not extract_dir.exists():
    print("开始解压数据...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=tar_path.parent)
    print("解压完成")
else:
    print("数据已解压，跳过")


