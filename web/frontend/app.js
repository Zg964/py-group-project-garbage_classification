/**
 * 智能垃圾分类 - 前端交互逻辑
 */

const API_BASE = '';

// DOM 元素
const elements = {
    modelSelect: document.getElementById('model-select'),
    modelInfo: document.getElementById('model-info'),
    uploadArea: document.getElementById('upload-area'),
    uploadContent: document.getElementById('upload-content'),
    previewArea: document.getElementById('preview-area'),
    previewImage: document.getElementById('preview-image'),
    fileInput: document.getElementById('file-input'),
    uploadBtn: document.getElementById('upload-btn'),
    changeBtn: document.getElementById('change-btn'),
    predictBtn: document.getElementById('predict-btn'),
    loadingArea: document.getElementById('loading-area'),
    resultArea: document.getElementById('result-area'),
    emptyState: document.getElementById('empty-state'),
    resultBadge: document.getElementById('result-badge'),
    resultClassCn: document.getElementById('result-class-cn'),
    resultClassEn: document.getElementById('result-class-en'),
    confidenceValue: document.getElementById('confidence-value'),
    barChart: document.getElementById('bar-chart'),
    adviceText: document.getElementById('advice-text'),
    infoModel: document.getElementById('info-model'),
    infoTime: document.getElementById('info-time'),
};

let selectedFile = null;
let modelData = [];

// ========== 颜色映射 ==========
const CLASS_COLORS = {
    'cardboard': { bg: '#fff3e0', bar: '#ff9800' },
    'glass': { bg: '#e3f2fd', bar: '#2196f3' },
    'metal': { bg: '#f3e5f5', bar: '#9c27b0' },
    'paper': { bg: '#e8f5e9', bar: '#4caf50' },
    'plastic': { bg: '#fff8e1', bar: '#ffc107' },
    'trash': { bg: '#eceff1', bar: '#607d8b' },
};

const CATEGORY_COLORS = {
    '可回收物': 'recyclable',
    '其他垃圾': 'other',
};

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupEventListeners();
});

// ========== 加载模型列表 ==========
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        modelData = data.models;

        // 构建选项
        elements.modelSelect.innerHTML = '<option value="">-- 请选择模型 --</option>';

        // 按模型类型分组
        const groups = {};
        for (const model of modelData) {
            if (!groups[model.display_name]) {
                groups[model.display_name] = [];
            }
            groups[model.display_name].push(model);
        }

        // 添加分组选项
        let optionIndex = 0;
        for (const [displayName, variants] of Object.entries(groups)) {
            for (const variant of variants) {
                const option = document.createElement('option');
                option.value = `${variant.name}|${variant.variant}`;
                // 默认选中最好的模型 (efficientnetv2s)
                const isDefault = variant.name === 'efficientnetv2s' && variant.variant === 'best';
                option.text = `${variant.display_name}`;
                if (variants.length > 1) {
                    option.text += ` (${variant.variant.toUpperCase()})`;
                }
                if (variant.accuracy) {
                    option.text += ` [${variant.accuracy}%]`;
                }
                if (isDefault) {
                    option.selected = true;
                }
                elements.modelSelect.appendChild(option);
                optionIndex++;
            }
        }

        // 触发 change 显示默认模型信息
        elements.modelSelect.dispatchEvent(new Event('change'));
    } catch (error) {
        console.error('加载模型列表失败:', error);
        elements.modelSelect.innerHTML = '<option value="">加载失败，请刷新重试</option>';
    }
}

// ========== 事件监听 ==========
function setupEventListeners() {
    // 模型选择变更
    elements.modelSelect.addEventListener('change', onModelChange);

    // 文件上传
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());
    elements.changeBtn.addEventListener('click', () => elements.fileInput.click());
    elements.uploadArea.addEventListener('click', (e) => {
        if (e.target === elements.uploadArea || e.target.closest('.upload-content')) {
            elements.fileInput.click();
        }
    });
    elements.fileInput.addEventListener('change', onFileSelected);

    // 拖拽上传
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('drag-over');
    });
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('drag-over');
    });
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // 预测按钮
    elements.predictBtn.addEventListener('click', onPredict);
}

// ========== 模型变更 ==========
function onModelChange() {
    const value = elements.modelSelect.value;
    if (!value) {
        elements.modelInfo.innerHTML = '<span class="model-info-placeholder">请选择模型以查看详情</span>';
        return;
    }

    const [name, variant] = value.split('|');
    const model = modelData.find(m => m.name === name && m.variant === variant);
    if (!model) return;

    // 构建标签
    const tags = [
        { cls: 'size', icon: '📦', text: `${model.size_mb} MB` },
        { cls: 'speed', icon: '⚡', text: model.speed_label },
    ];

    if (model.accuracy) {
        tags.push({ cls: 'accuracy', icon: '🎯', text: `${model.accuracy}% 准确率` });
    }

    elements.modelInfo.innerHTML = `
        <div class="model-info-content">
            ${tags.map(t => `<span class="model-info-tag ${t.cls}">${t.icon} ${t.text}</span>`).join('')}
        </div>
    `;
}

// ========== 文件选择 ==========
function onFileSelected(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    // 验证文件类型
    if (!file.type.startsWith('image/')) {
        alert('请选择图片文件');
        return;
    }

    // 验证文件大小 (最大 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('图片文件过大，请选择小于 10MB 的图片');
        return;
    }

    selectedFile = file;

    // 预览
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadContent.classList.add('hidden');
        elements.previewArea.classList.remove('hidden');
        elements.predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ========== 预测 ==========
async function onPredict() {
    if (!selectedFile) return;

    // 获取选中的模型
    const modelValue = elements.modelSelect.value;
    if (!modelValue) {
        alert('请先选择模型');
        return;
    }
    const [modelName, variant] = modelValue.split('|');

    // 显示加载状态
    elements.loadingArea.classList.remove('hidden');
    elements.resultArea.classList.add('hidden');
    elements.emptyState.classList.add('hidden');
    elements.predictBtn.disabled = true;
    elements.predictBtn.innerHTML = '<span class="btn-icon">⏳</span> 识别中...';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_name', modelName);
        formData.append('variant', variant);
        formData.append('topk', '6');

        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error('预测失败:', error);
        alert('预测失败：' + error.message);
        elements.emptyState.classList.remove('hidden');
    } finally {
        elements.loadingArea.classList.add('hidden');
        elements.predictBtn.disabled = false;
        elements.predictBtn.innerHTML = '<span class="btn-icon">🔍</span> 开始识别';
    }
}

// ========== 显示结果 ==========
function displayResult(result) {
    // 顶部主要预测
    const badgeClass = CATEGORY_COLORS[result.category] || 'other';
    elements.resultBadge.textContent = result.category;
    elements.resultBadge.className = `result-badge ${badgeClass}`;
    elements.resultClassCn.textContent = result.top_class_cn;
    elements.resultClassEn.textContent = result.top_class;
    elements.confidenceValue.textContent = `${(result.top_probability * 100).toFixed(1)}%`;

    // 概率柱状图
    elements.barChart.innerHTML = '';
    const maxProb = result.predictions[0].probability;

    result.predictions.forEach((pred, i) => {
        const colorInfo = CLASS_COLORS[pred.class_name] || { bar: '#9e9e9e' };
        const isTop = pred === result.predictions[0];
        const barWidth = maxProb > 0 ? (pred.probability / maxProb) * 100 : 0;

        const barItem = document.createElement('div');
        barItem.className = 'bar-item';
        barItem.innerHTML = `
            <span class="bar-label">${pred.class_cn}</span>
            <div class="bar-track">
                <div class="bar-fill ${isTop ? 'top' : 'other'}"
                     style="width: 0%; background: ${colorInfo.bar};">
                </div>
            </div>
            <span class="bar-value">${(pred.probability * 100).toFixed(1)}%</span>
        `;

        // 延迟触发动画展开
        const fill = barItem.querySelector('.bar-fill');
        setTimeout(() => {
            fill.style.width = `${barWidth}%`;
        }, 50 * (i + 1));

        elements.barChart.appendChild(barItem);
    });

    // 分类建议
    elements.adviceText.textContent = result.advice;

    // 推理信息
    elements.infoModel.textContent = result.model_used;
    elements.infoTime.textContent = `${result.inference_time_ms} ms`;

    // 显示结果
    elements.resultArea.classList.remove('hidden');
    elements.emptyState.classList.add('hidden');

    // 滚动到结果区域
    elements.resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
