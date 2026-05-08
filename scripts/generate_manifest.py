import json
from pathlib import Path

dst = Path('data/processed')
classes = [p.name for p in dst.iterdir() if p.is_dir()]
images = []
class_count = {}
for c in classes:
    files = [str(p.as_posix()) for p in (dst / c).glob('*') if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png']]
    class_count[c] = len(files)
    for f in files:
        images.append({'path': f, 'class': c})
manifest = {
    'classes': classes,
    'class_count': class_count,
    'images': images,
    'total_images': len(images)
}
with open(dst / 'manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print('manifest saved, total_images=', manifest['total_images'])
