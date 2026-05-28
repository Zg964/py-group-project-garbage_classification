"""
v1.06: 数据集扩展脚本（git clone 版）
使用 git sparse clone 批量拉取 Hugging Face 数据集仓库，
提取需要的类别图片，去重后合并到现有数据集中。

数据源:
  1. omasteam/waste-garbage-management-dataset (MIT, 19,762 张, 10 类)
  2. shahzaibvohra/realwaste (CC BY 4.0, 4,752 张, 9 类)
"""

import os
import sys
import hashlib
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
from PIL import Image
import logging

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUR_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
TARGET_SIZE = (224, 224)

CLASS_MAPPING_OMASTEAM = {
    'cardboard': 'cardboard', 'glass': 'glass', 'metal': 'metal',
    'paper': 'paper', 'plastic': 'plastic', 'trash': 'trash',
}

CLASS_MAPPING_REALWASTE = {
    'Cardboard': 'cardboard', 'Glass': 'glass', 'Metal': 'metal',
    'Paper': 'paper', 'Plastic': 'plastic',
    'Miscellaneous Trash': 'trash', 'Textile Trash': 'trash',
    'Food Organics': 'trash', 'Vegetation': 'trash',
}


def compute_md5(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False


def build_hash_index(data_dir):
    """建立现有数据集哈希索引（用于去重）"""
    data_dir = Path(data_dir)
    hashes = set()
    for cls in OUR_CLASSES:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            for f in tqdm(list(cls_dir.iterdir()), desc=f"索引 {cls}"):
                if f.is_file():
                    h = compute_md5(f)
                    if h:
                        hashes.add(h)
    logger.info(f"现有数据集: {len(hashes)} 个唯一哈希")
    return hashes


def git_sparse_clone(repo_url, repo_dir, include_dirs):
    """用 git sparse clone 只下载指定目录

    Args:
        repo_url: git 仓库 URL
        repo_dir: 本地目标目录
        include_dirs: 要下载的目录列表
    """
    repo_dir = Path(repo_dir)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    # 初始化仓库并设置 sparse checkout
    cmds = [
        ['git', 'init'],
        ['git', 'remote', 'add', 'origin', repo_url],
        ['git', 'config', 'core.sparseCheckout', 'true'],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd=repo_dir, check=True, capture_output=True)

    # 设置 sparse-checkout 路径
    sparse_path = repo_dir / '.git/info/sparse-checkout'
    with open(sparse_path, 'w') as f:
        for d in include_dirs:
            f.write(f'{d}/*\n')

    # 拉取（深度 1，不下载文件内容历史）
    logger.info(f"  拉取仓库（仅 {', '.join(include_dirs)}）...")
    result = subprocess.run(
        ['git', 'pull', '--depth', '1', 'origin', 'main'],
        cwd=repo_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        # 尝试 master 分支
        result = subprocess.run(
            ['git', 'pull', '--depth', '1', 'origin', 'master'],
            cwd=repo_dir, capture_output=True, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(f"git pull 失败: {result.stderr}")

    # 检查下载结果
    total_files = sum(1 for _ in repo_dir.rglob('*') if _.is_file())
    logger.info(f"  下载完成: {total_files} 个文件")


def extract_images(repo_dir, class_mapping, output_dir, existing_hashes,
                    max_per_class=None, trash_only=False, source_name=''):
    """从仓库目录提取图片"""
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    added = Counter()
    skipped_hash = Counter()
    skipped_invalid = Counter()

    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        all_images.extend(repo_dir.rglob(ext))

    # 排除 .git
    all_images = [f for f in all_images if '.git' not in f.parts]

    for img_path in tqdm(all_images, desc=f"提取 {source_name}"):
        rel = img_path.relative_to(repo_dir)
        parts = rel.parts

        # 推断外部类别名
        ext_class = None
        for part in parts[:-1]:
            for ek in class_mapping:
                if part.lower() == ek.lower():
                    ext_class = ek
                    break
            if ext_class:
                break

        if ext_class is None:
            for ek in class_mapping:
                norm_ek = ek.lower().replace(' ', '_')
                norm_stem = img_path.stem.lower().replace(' ', '_')
                if norm_stem.startswith(norm_ek.lower()):
                    ext_class = ek
                    break

        if ext_class is None:
            continue

        our_class = class_mapping.get(ext_class)
        if our_class not in OUR_CLASSES:
            continue
        if trash_only and our_class != 'trash':
            continue
        if max_per_class and added[our_class] >= max_per_class.get(our_class, float('inf')):
            continue

        if not is_valid_image(img_path):
            skipped_invalid[our_class] += 1
            continue

        md5 = compute_md5(img_path)
        if md5 is None or md5 in existing_hashes:
            skipped_hash[our_class] += 1
            continue

        cls_dir = output_dir / our_class
        cls_dir.mkdir(parents=True, exist_ok=True)
        dest = cls_dir / f"{md5[:16]}.jpg"
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB').resize(TARGET_SIZE, Image.LANCZOS)
                img.save(dest, 'JPEG', quality=95)
            existing_hashes.add(md5)
            added[our_class] += 1
        except:
            if dest.exists():
                dest.unlink()

    return added, skipped_hash, skipped_invalid


def print_stats(data_dir):
    total = 0
    for cls in OUR_CLASSES:
        cls_dir = Path(data_dir) / cls
        count = len(list(cls_dir.glob('*.*'))) if cls_dir.exists() else 0
        total += count
        logger.info(f"  {cls}: {count:>5d} 张")
    logger.info(f"  总计: {total} 张")
    return total


def main():
    parser = argparse.ArgumentParser(description='数据集扩展 — git clone 方式')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--max-trash', type=int, default=1000, help='trash 类最大补充量')
    parser.add_argument('--trash-only', action='store_true', help='仅补充 trash')
    parser.add_argument('--temp-dir', default=None, help='临时下载目录')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tmp_base = Path(args.temp_dir) if args.temp_dir else Path(tempfile.mkdtemp())
    tmp_base.mkdir(parents=True, exist_ok=True)

    max_per_class = {cls: 0 for cls in OUR_CLASSES}
    max_per_class['trash'] = args.max_trash

    logger.info("=" * 60)
    logger.info("当前数据集:")
    print_stats(data_dir)

    existing_hashes = build_hash_index(data_dir)

    datasets = [
        {
            'url': 'https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset',
            'dirs': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
            'mapping': CLASS_MAPPING_OMASTEAM,
            'name': 'omasteam',
        },
        {
            'url': 'https://huggingface.co/datasets/shahzaibvohra/realwaste',
            'dirs': [],
            'mapping': CLASS_MAPPING_REALWASTE,
            'name': 'realwaste',
        },
    ]

    all_added = Counter()

    for ds in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"下载 {ds['name']}...")
        logger.info(f"{'='*60}")

        # 如果只有 trash-only 且数据集没有 trash 类别，跳过
        if args.trash_only and ds['name'] == 'omasteam':
            ds['dirs'] = ['trash']  # 只下载 trash 目录

        if not ds['dirs']:
            # realwaste 没有子目录，直接 clone 全部
            ds['dirs'] = ['.']

        clone_dir = tmp_base / ds['name']
        try:
            git_sparse_clone(ds['url'], clone_dir, ds['dirs'])
            added, skipped_h, skipped_i = extract_images(
                clone_dir, ds['mapping'], data_dir, existing_hashes,
                max_per_class=max_per_class if args.trash_only else None,
                trash_only=args.trash_only,
                source_name=ds['name']
            )
            all_added += added
            for cls in OUR_CLASSES:
                if added.get(cls, 0) > 0 or skipped_h.get(cls, 0) > 0:
                    logger.info(f"  {cls}: +{added.get(cls, 0)} (重复 {skipped_h.get(cls, 0)}, 无效 {skipped_i.get(cls, 0)})")

            # 清理
            shutil.rmtree(clone_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"  下载/提取 {ds['name']} 失败: {e}")

    total_added = sum(all_added.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"完成！新增 {total_added} 张")
    print_stats(data_dir)

    # 清理临时目录
    if not args.temp_dir and tmp_base.exists():
        shutil.rmtree(tmp_base, ignore_errors=True)


if __name__ == '__main__':
    main()
