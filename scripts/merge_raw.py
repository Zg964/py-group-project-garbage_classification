import os
import shutil
from pathlib import Path

src1 = Path(r"d:\claude_code\teamwork--copilot\data\raw-data\Garbage classification\Garbage classification")
src2 = Path(r"d:\claude_code\teamwork--copilot\data\raw-data\trashnet-master\data")
dst = Path(r"d:\claude_code\teamwork--copilot\garbage_classification\data\processed")
classes = ['cardboard','glass','metal','paper','plastic','trash']

os.makedirs(dst, exist_ok=True)
for c in classes:
    td = dst / c
    td.mkdir(parents=True, exist_ok=True)
    for s in (src1, src2):
        sd = s / c
        if sd.exists():
            for p in sd.rglob('*'):
                if p.is_file():
                    if p.suffix.lower() in ['.jpg','.jpeg','.png']:
                        try:
                            shutil.copy2(p, td)
                        except Exception as e:
                            print('跳过文件', p, e)

# 输出每类文件数量
for c in classes:
    td = dst / c
    if td.exists():
        print(f"{c}: {len(list(td.glob('*')))} files")
    else:
        print(f"{c}: 0 files")
