import json
from pathlib import Path

logs = Path('logs')
metrics_file = logs / 'evaluation_results.json'
report_file = Path('中期报告.md')

if not metrics_file.exists():
    print('找不到评估结果文件:', metrics_file)
    raise SystemExit(1)

with open(metrics_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
lines.append('# 中期报告（基于真实数据的快速试跑结果）\n')
lines.append('本次运行使用合并的原始数据集（见 `data/processed/manifest.json`），对三个基线模型进行了快速试跑（每个模型 5 epochs），用于生成初步对比与下一步调优依据。\n')

# 总结表
lines.append('**快速试跑摘要**\n')
for m, v in data.items():
    lines.append(f'- **{m}**: Accuracy={v.get("accuracy"):.3f}, Macro-F1={v.get("macro_f1"):.3f}, Inference={v.get("inference_time_ms"):.1f}ms')
lines.append('\n')

# 详细每模型
for m, v in data.items():
    lines.append(f'## 模型: {m}\n')
    lines.append(f'- 准确率: {v.get("accuracy"):.3f}')
    lines.append(f'- Macro-F1: {v.get("macro_f1"):.4f}')
    lines.append(f'- 加权 F1: {v.get("weighted_f1"):.4f}')
    lines.append(f'- 平均推理时间: {v.get("inference_time_ms"):.2f} ms')
    lines.append('\n')
    lines.append('### 分类报告（摘要）\n')
    for cls, stats in v.get('classification_report', {}).items():
        if cls in ['accuracy','macro avg','weighted avg']:
            continue
        p = stats.get('precision', 0)
        r = stats.get('recall', 0)
        f1 = stats.get('f1-score', 0)
        lines.append(f'- {cls}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}')
    lines.append('\n')

# 结论与下一步
lines.append('## 初步结论与下一步计划\n')
lines.append('- 当前快速试跑显示模型在部分类别（例如 `plastic`）表现较好，但总体准确率偏低（约 20%），说明数据分布或标签存在不平衡或模型训练未充分。')
lines.append('- 下一步推荐：\n  1. 检查并修正标签噪声与不平衡（过采样、类别权重、更严格的清洗）；\n  2. 增加训练 epoch（如 50+）、使用更强的数据增强；\n  3. 超参数搜索（学习率、权重衰减、优化器）；\n  4. 尝试迁移学习微调与模型集成。')
lines.append('\n')
lines.append('## 产物位置\n')
lines.append('- 训练模型: `models/`（包含 `*_best.pth`）\n')
lines.append('- 训练历史: `models/training_history.json`\n')
lines.append('- 评估结果: `logs/evaluation_results.json`，混淆矩阵与可视化保存在 `logs/` 下（如有）\n')

report_text = '\n'.join(lines)
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)
print('中期报告已更新：', report_file)
