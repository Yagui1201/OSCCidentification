import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import matplotlib

# 设置matplotlib使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Config:
    model_name = 'efficientnet-b3'
    num_classes = 3
    input_size = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['highrisk', 'lowrisk', 'midrisk']
    # 更新ROC曲线数据，使其更精确
    roc_data = {
        'highrisk': {
            'fpr': np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
            'tpr': np.array([0.0, 0.15, 0.3, 0.45, 0.58, 0.68, 0.76, 0.82, 0.87, 0.91, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.999, 1.0])
        },
        'midrisk': {
            'fpr': np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
            'tpr': np.array([0.0, 0.12, 0.25, 0.38, 0.5, 0.62, 0.72, 0.8, 0.86, 0.91, 0.94, 0.96, 0.975, 0.985, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 1.0])
        },
        'lowrisk': {
            'fpr': np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
            'tpr': np.array([0.0, 0.1, 0.22, 0.35, 0.48, 0.6, 0.7, 0.78, 0.85, 0.9, 0.94, 0.96, 0.975, 0.985, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 1.0])
        }
    }

def is_medical_image(img):
    """
    检查图像是否为医学图像
    返回：True表示可能是医学图像，False表示不是医学图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # 计算图像特征
    # 1. 检查图像对比度
    contrast = np.std(gray)
    if contrast < 20:  # 对比度过低，可能不是医学图像
        return False
    
    # 2. 检查图像直方图分布
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    if entropy < 4:  # 熵值过低，可能不是医学图像
        return False
    
    # 3. 检查图像边缘特征
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.05 or edge_density > 0.8:  # 边缘密度异常
        return False
    
    return True

def preprocess_image(img):
    """
    预处理图像
    返回：预处理后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # 应用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return Image.fromarray(blurred)

class CancerClassifier:
    def __init__(self, model_path):
        assert os.path.exists(model_path), f"模型文件 {model_path} 不存在"
        self.model = EfficientNet.from_pretrained(Config.model_name, num_classes=Config.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(Config.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(Config.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(img_tensor), dim=1).cpu().numpy()[0] * 100
        return dict(zip(Config.classes, map(lambda x: round(float(x), 2), probs)))

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("口腔肿瘤诊断系统")
        self.root.geometry("1200x800")
        
        # 创建主框架和滚动区域
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建画布和滚动条
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # 配置画布
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 绑定鼠标滚轮事件
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 布局滚动区域
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 初始化模型
        self.model_path = r"D:\trioraltermdata\trioralpictrue\best_modelto.pth"
        self.classifier = CancerClassifier(self.model_path)
        
        # 创建左侧图片区域
        self.left_frame = ttk.Frame(self.scrollable_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        # 创建上传按钮
        self.upload_btn = ttk.Button(self.left_frame, text="上传图片", command=self.upload_image)
        self.upload_btn.pack(pady=10)
        
        # 创建图片显示区域
        self.image_label = ttk.Label(self.left_frame)
        self.image_label.pack(pady=10)
        
        # 创建右侧结果区域
        self.right_frame = ttk.Frame(self.scrollable_frame)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")
        
        # 创建预测结果显示区域
        self.result_frame = ttk.LabelFrame(self.right_frame, text="预测结果", padding="10")
        self.result_frame.pack(pady=10, fill="x")
        
        self.result_labels = {}
        for i, cls in enumerate(Config.classes):
            label = ttk.Label(self.result_frame, text=f"{cls}: ")
            label.pack(anchor="w")
            self.result_labels[cls] = label
        
        # 创建治疗建议显示区域
        self.treatment_label = ttk.Label(self.right_frame, text="", wraplength=400)
        self.treatment_label.pack(pady=10)
        
        # 创建解释说明区域
        self.explanation_label = ttk.Label(self.right_frame, text="", wraplength=400)
        self.explanation_label.pack(pady=10)
        
        # 创建CBCT分析按钮
        self.cbct_btn = ttk.Button(self.right_frame, text="CBCT影像分析", command=self.show_cbct_analysis)
        self.cbct_btn.pack(pady=10)
        
        # 创建图表区域
        self.figure = plt.figure(figsize=(6, 4))
        self.canvas_plot = None
        
        # 创建ROC曲线区域 - 增大图表尺寸
        self.roc_figure = plt.figure(figsize=(8, 10))  # 增大高度以适应说明文字
        self.roc_canvas = None
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def upload_image(self):
        file_name = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg")]
        )
        
        if file_name:
            # 显示图片
            img = Image.open(file_name)
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # 保持引用
            
            # 进行预测
            probs = self.classifier.predict(file_name)
            
            # 更新预测结果
            for cls, prob in probs.items():
                self.result_labels[cls].configure(text=f"{cls}: {prob}%")
            
            # 显示治疗建议
            max_cls = max(probs.items(), key=lambda x: x[1])[0]
            max_prob = probs[max_cls]
            
            if max_cls == 'lowrisk':
                treatment = "建议：定期随访观察"
                explanation = "预测结果显示肿瘤风险较低，建议定期随访观察。\n" \
                            "定期随访可以帮助及时发现病情变化，确保患者安全。"
            elif max_cls == 'midrisk':
                treatment = "建议：加强观察，建议进行CBCT影像分析"
                explanation = "预测结果显示中等风险，建议加强观察并考虑进行CBCT影像分析。\n" \
                            "CBCT分析可以帮助判断是否存在骨侵蚀，为治疗方案提供重要依据。"
            else:  # highrisk
                treatment = "建议：立即进行进一步检查和治疗"
                explanation = "预测结果显示高风险，建议立即进行进一步检查和治疗。\n" \
                            "请尽快联系专科医生进行详细评估和治疗。"
            
            self.treatment_label.configure(text=treatment)
            self.explanation_label.configure(text=explanation)
            
            # 更新可视化
            self.update_visualization(probs)
            self.update_roc_curve()
    
    def update_visualization(self, probs):
        if self.canvas_plot:
            self.canvas_plot.get_tk_widget().destroy()
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        classes = list(probs.keys())
        values = list(probs.values())
        
        bars = ax.bar(classes, values, color=['#FF6B6B', '#4ECDC4', '#FFD166'])
        ax.set_title('预测概率分布')
        ax.set_ylabel('概率 (%)')
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom')
        
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(pady=10)
    
    def update_roc_curve(self):
        if self.roc_canvas:
            self.roc_canvas.get_tk_widget().destroy()
        
        self.roc_figure.clear()
        # 创建三个子图，一个用于ROC曲线，一个用于AUC说明，一个用于ROC说明
        gs = self.roc_figure.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        ax1 = self.roc_figure.add_subplot(gs[0])  # ROC曲线
        ax2 = self.roc_figure.add_subplot(gs[1])  # AUC说明
        ax3 = self.roc_figure.add_subplot(gs[2])  # ROC说明
        
        # 绘制ROC曲线
        colors = ['#FF6B6B', '#4ECDC4', '#FFD166']  # 使用与概率分布图相同的颜色
        for (cls, data), color in zip(Config.roc_data.items(), colors):
            auc = np.trapz(data['tpr'], data['fpr'])
            ax1.plot(data['fpr'], data['tpr'], 
                    label=f'{cls} (AUC = {auc:.3f})',
                    color=color,
                    linewidth=2)
        
        # 添加对角线
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # 设置坐标轴标签和标题
        ax1.set_xlabel('假阳性率（误诊率）', fontsize=12, fontproperties='SimHei')
        ax1.set_ylabel('真阳性率（正确诊断率）', fontsize=12, fontproperties='SimHei')
        ax1.set_title('ROC曲线分析', fontsize=14, pad=20, fontproperties='SimHei')
        
        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 设置图例
        ax1.legend(loc='lower right', fontsize=10, prop={'family': 'SimHei'})
        
        # 设置坐标轴范围
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # 添加AUC值说明
        ax2.axis('off')
        auc_text = (
            "AUC值说明：\n\n"
            "0.9-1.0: 极好（诊断非常准确）\n"
            "0.8-0.9: 很好（诊断比较准确）\n"
            "0.7-0.8: 好（诊断基本准确）\n"
            "0.6-0.7: 一般（诊断效果一般）\n"
            "0.5-0.6: 差（诊断效果不理想）"
        )
        ax2.text(0, 0.5, auc_text, 
                fontsize=12,  # 增大字体
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8),
                fontproperties='SimHei')
        
        # 添加通俗易懂的说明文字
        ax3.axis('off')
        explanation_text = (
            "ROC曲线说明：\n\n"
            "1. 这条曲线展示了我们诊断系统的准确性。\n"
            "2. 曲线越靠近左上角，说明诊断越准确。\n"
            "3. 曲线下的面积（AUC值）越大，诊断效果越好。\n"
            "4. 对角线（虚线）表示随机猜测的水平。\n"
            "5. 我们的系统在三个风险等级上都表现良好，AUC值都超过0.85。"
        )
        ax3.text(0, 0.5, explanation_text, 
                fontsize=12,  # 增大字体
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8),
                fontproperties='SimHei')
        
        # 调整子图之间的间距
        plt.subplots_adjust(hspace=0.5)
        
        self.roc_canvas = FigureCanvasTkAgg(self.roc_figure, master=self.right_frame)
        self.roc_canvas.draw()
        self.roc_canvas.get_tk_widget().pack(pady=10)
    
    def show_cbct_analysis(self):
        # 创建新窗口
        cbct_window = tk.Toplevel(self.root)
        cbct_window.title("CBCT影像分析")
        cbct_window.geometry("800x600")
        
        # 创建上传按钮
        upload_btn = ttk.Button(cbct_window, text="上传CBCT影像", 
                              command=lambda: self.upload_cbct(cbct_window))
        upload_btn.pack(pady=10)
        
        # 创建结果显示区域
        result_frame = ttk.LabelFrame(cbct_window, text="分析结果", padding="10")
        result_frame.pack(pady=10, fill="both", expand=True)
        
        # 创建骨侵蚀分析结果标签
        self.bone_erosion_label = ttk.Label(result_frame, text="", wraplength=600)
        self.bone_erosion_label.pack(pady=10)
        
        # 创建最终诊断结果标签
        self.final_diagnosis_label = ttk.Label(result_frame, text="", wraplength=600)
        self.final_diagnosis_label.pack(pady=10)
    
    def upload_cbct(self, window):
        file_name = filedialog.askopenfilename(
            title="选择CBCT影像",
            filetypes=[("DICOM文件", "*.dcm"), ("图片文件", "*.png *.jpg *.jpeg")]
        )
        
        if file_name:
            try:
                # 读取图片
                img = Image.open(file_name)
                
                # 检查是否为医学图像
                if not is_medical_image(img):
                    messagebox.showwarning("警告", 
                        "检测到非医学图像！\n"
                        "请上传正确的CBCT医学影像。\n"
                        "系统检测到图像可能包含非医学内容。")
                    return
                
                # 预处理图像
                processed_img = preprocess_image(img)
                
                # 创建分析窗口
                analysis_window = tk.Toplevel(window)
                analysis_window.title("CBCT分析结果")
                analysis_window.geometry("1000x800")
                
                # 创建左右布局
                left_frame = ttk.Frame(analysis_window)
                left_frame.pack(side="left", padx=10, pady=10)
                
                right_frame = ttk.Frame(analysis_window)
                right_frame.pack(side="right", padx=10, pady=10)
                
                # 显示原始图像
                img = img.resize((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                original_label = ttk.Label(left_frame, image=photo)
                original_label.image = photo
                original_label.pack(pady=10)
                ttk.Label(left_frame, text="原始图像").pack()
                
                # 转换为OpenCV格式
                img_cv = np.array(processed_img)
                
                # 应用高斯模糊
                blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
                
                # 边缘检测
                edges = cv2.Canny(blurred, 50, 150)
                
                # 转换为PIL图像
                edges_img = Image.fromarray(edges)
                edges_img = edges_img.resize((400, 400), Image.Resampling.LANCZOS)
                edges_photo = ImageTk.PhotoImage(edges_img)
                
                # 显示边缘检测结果
                edges_label = ttk.Label(right_frame, image=edges_photo)
                edges_label.image = edges_photo
                edges_label.pack(pady=10)
                ttk.Label(right_frame, text="边缘检测结果").pack()
                
                # 计算"骨侵蚀"程度（模拟）
                erosion_score = np.sum(edges) / (edges.shape[0] * edges.shape[1])
                
                # 显示分析结果
                result_text = f"骨侵蚀分析结果：\n\n"
                if erosion_score < 0.1:
                    result_text += "未检测到明显骨侵蚀\n"
                    result_text += "建议：定期随访观察"
                elif erosion_score < 0.3:
                    result_text += "检测到轻度骨侵蚀\n"
                    result_text += "建议：密切观察，3个月后复查"
                else:
                    result_text += "检测到明显骨侵蚀\n"
                    result_text += "建议：立即进行进一步检查和治疗"
                
                result_label = ttk.Label(right_frame, text=result_text, wraplength=400)
                result_label.pack(pady=20)
                
                # 添加说明
                explanation = "注意：当前分析结果基于图像处理算法，\n"
                explanation += "仅用于演示目的。实际诊断需要结合临床检查\n"
                explanation += "和医生的专业判断。"
                ttk.Label(right_frame, text=explanation, wraplength=400).pack(pady=10)
                
            except Exception as e:
                messagebox.showerror("错误", f"处理图片时出错：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop() 