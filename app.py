"""
Streamlit 演示应用
垃圾分类系统的交互式界面
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import json
from src.models import create_model
from src.data_loader import GARBAGE_CLASSES, get_transforms

# 页面配置
st.set_page_config(
    page_title="智能垃圾分类系统",
    page_icon="♻️",
    layout="wide"
)

# CSS 样式
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .block-container {
        padding: 2rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name, device):
    """加载模型"""
    model = create_model(model_name, num_classes=len(GARBAGE_CLASSES), pretrained=True)
    model_path = f'models/{model_name}_best.pth'
    
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_evaluation_results():
    """加载评估结果"""
    results_path = 'logs/evaluation_results.json'
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def predict_image(image, model, device):
    """预测图像"""
    _, val_transform = get_transforms()
    
    # 预处理
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = GARBAGE_CLASSES[predicted_idx]
        confidence = probabilities[0, predicted_idx].item()
    
    # 获取所有类别的概率
    all_probs = {GARBAGE_CLASSES[i]: float(probabilities[0, i].item()) 
                 for i in range(len(GARBAGE_CLASSES))}
    
    return predicted_class, confidence, all_probs


def main():
    st.title("♻️ 智能垃圾分类系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 选择模型
        model_options = ['simple_cnn', 'mobilenetv2', 'resnet18']
        selected_model = st.selectbox(
            "选择模型",
            model_options,
            help="选择用于预测的模型"
        )
        
        st.markdown("---")
        
        # 关于
        st.subheader("📖 关于")
        st.markdown("""
        这是一个基于深度学习的垃圾分类系统。
        
        **支持的垃圾类别：**
        - 🗂️ Cardboard (纸板)
        - 🥫 Glass (玻璃)
        - 🔧 Metal (金属)
        - 📄 Paper (纸张)
        - 🛢️ Plastic (塑料)
        - 🗑️ Trash (其他垃圾)
        """)
    
    # 主要内容
    tab1, tab2, tab3 = st.tabs(["🎯 预测", "📊 模型对比", "📈 详细信息"])
    
    with tab1:
        st.header("图像分类预测")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("上传或选择图像")
            
            # 文件上传
            uploaded_file = st.file_uploader(
                "选择一张垃圾图像",
                type=['jpg', 'jpeg', 'png'],
                help="支持 JPG、JPEG 和 PNG 格式"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="上传的图像", use_column_width=True)
                
                # 预测
                if st.button("🔍 开始分类", use_container_width=True):
                    with st.spinner("正在分类..."):
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = load_model(selected_model, device)
                        
                        predicted_class, confidence, all_probs = predict_image(image, model, device)
                    
                    st.success("分类完成！")
        
        with col2:
            st.subheader("分类结果")
            
            if uploaded_file is not None:
                # 显示预测结果
                col2_1, col2_2 = st.columns([1, 1])
                
                with col2_1:
                    st.metric("预测类别", f"**{predicted_class.upper()}**")
                    st.metric("置信度", f"{confidence:.1%}")
                
                with col2_2:
                    st.metric("使用模型", selected_model)
                    device_type = "GPU 🚀" if torch.cuda.is_available() else "CPU"
                    st.metric("计算设备", device_type)
                
                st.markdown("---")
                
                # 所有类别的概率
                st.subheader("各类别预测概率")
                
                # 转换为列表以便排序
                probs_list = [(cls, prob) for cls, prob in all_probs.items()]
                probs_list.sort(key=lambda x: x[1], reverse=True)
                
                # 绘制概率分布
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 5))
                classes = [p[0] for p in probs_list]
                probs = [p[1] for p in probs_list]
                
                colors = ['#FF6B6B' if c == predicted_class else '#4ECDC4' for c in classes]
                ax.barh(classes, probs, color=colors)
                ax.set_xlabel('概率')
                ax.set_title('各类别预测概率')
                ax.set_xlim([0, 1])
                
                # 添加概率标签
                for i, (cls, prob) in enumerate(probs_list):
                    ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
                
                st.pyplot(fig, use_container_width=True)
                
                # 建议
                st.markdown("---")
                st.subheader("♻️ 分类建议")
                
                garbage_tips = {
                    'cardboard': '📦 **纸板类垃圾** - 请拆解后进行回收处理，避免受潮。',
                    'glass': '🥫 **玻璃类垃圾** - 请小心处理，避免划伤。可回收利用。',
                    'metal': '🔧 **金属类垃圾** - 请清洁后进行回收，很有回收价值。',
                    'paper': '📄 **纸质垃圾** - 请保持干燥，集中回收处理。',
                    'plastic': '🛢️ **塑料类垃圾** - 请清洁后分类回收，避免污染。',
                    'trash': '🗑️ **其他垃圾** - 无法回收的垃圾，请正确投放。'
                }
                
                st.info(garbage_tips.get(predicted_class.lower(), "无法识别的垃圾类型"))
            else:
                st.info("👈 请先上传一张图像")
    
    with tab2:
        st.header("模型性能对比")
        
        results = load_evaluation_results()
        
        if results:
            # 创建对比表格
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    '模型': model_name,
                    '准确率': f"{metrics['accuracy']:.1%}",
                    'Macro-F1': f"{metrics['macro_f1']:.4f}",
                    '平均推理时间': f"{metrics['inference_time_ms']:.2f}ms"
                })
            
            st.dataframe(comparison_data, use_container_width=True)
            
            st.markdown("---")
            
            # 绘制对比图
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("准确率对比")
                accuracies = {m: results[m]['accuracy'] for m in results}
                st.bar_chart(accuracies)
            
            with col2:
                st.subheader("Macro-F1 对比")
                f1_scores = {m: results[m]['macro_f1'] for m in results}
                st.bar_chart(f1_scores)
            
            with col3:
                st.subheader("推理速度对比")
                inference_times = {m: results[m]['inference_time_ms'] for m in results}
                st.bar_chart(inference_times)
        else:
            st.warning("⚠️ 评估结果未找到，请先运行评估脚本。")
    
    with tab3:
        st.header("详细信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("系统信息")
            st.markdown(f"""
            - **PyTorch 版本**: {torch.__version__}
            - **计算设备**: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
            - **GPU 可用**: {'✓ 是' if torch.cuda.is_available() else '✗ 否'}
            - **支持的类别**: {len(GARBAGE_CLASSES)}
            - **垃圾类别**: {', '.join(GARBAGE_CLASSES)}
            """)
        
        with col2:
            st.subheader("模型信息")
            st.markdown(f"""
            - **选择的模型**: {selected_model}
            - **预训练**: {'✓ 是' if selected_model in ['mobilenetv2', 'resnet18'] else '✗ 否'}
            - **输入尺寸**: 224x224
            - **输出类别数**: {len(GARBAGE_CLASSES)}
            """)
        
        st.markdown("---")
        
        results = load_evaluation_results()
        if results and selected_model in results:
            st.subheader(f"{selected_model} 的评估指标")
            
            metrics = results[selected_model]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("准确率", f"{metrics['accuracy']:.1%}")
            col2.metric("Macro-F1", f"{metrics['macro_f1']:.4f}")
            col3.metric("推理时间", f"{metrics['inference_time_ms']:.2f}ms")
            
            st.markdown("---")
            
            st.subheader("分类详细报告")
            
            report = metrics['classification_report']
            
            report_data = []
            for cls in GARBAGE_CLASSES:
                if cls in report:
                    cls_metrics = report[cls]
                    report_data.append({
                        '类别': cls,
                        'Precision': f"{cls_metrics['precision']:.3f}",
                        'Recall': f"{cls_metrics['recall']:.3f}",
                        'F1-Score': f"{cls_metrics['f1-score']:.3f}",
                        'Support': int(cls_metrics['support'])
                    })
            
            st.dataframe(report_data, use_container_width=True)
            
            # 显示混淆矩阵
            st.subheader("混淆矩阵")
            confusion_img_path = f'logs/{selected_model}_confusion_matrix.png'
            if Path(confusion_img_path).exists():
                st.image(confusion_img_path, use_column_width=True)
            else:
                st.info("混淆矩阵图像未找到")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>智能垃圾分类系统 | 基于深度学习 | ♻️ 保护环境，从分类开始</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
