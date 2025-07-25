import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import pandas as pd

'''
这个文件可以说是主体了，将你想要预测的图片（装文件夹里），现在这里使用的是随机的几张照片，
我装在左侧的testpictrue文件夹中。
放在这（见下文MODEL_PATH和IMAGE_PATH），
建议用绝对路径，使用你想要的模型，可以得到结果，

这个文件运行会得到一张图，prediction_report.png
和一个xlsx表格
'''


#这素模型
class Config:
    model_name = 'efficientnet-b3'
    num_classes = 3
    input_size = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['highrisk', 'lowrisk', 'midrisk']  
#这素数据
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
        """ 返回标准化到0-100%的概率 """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(Config.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(img_tensor), dim=1).cpu().numpy()[0] * 100  # 转为百分比
        return dict(zip(Config.classes, map(lambda x: round(float(x), 2), probs)))  # 保留2位小数

    def batch_predict(self, image_dir):
        """ 批量预测并返回DataFrame """
        results = []
        valid_extensions = ['.jpg', '.jpeg', '.png']
        
        for img_name in os.listdir(image_dir):
            if not any(img_name.lower().endswith(ext) for ext in valid_extensions):
                continue
            img_path = os.path.join(image_dir, img_name)
            try:
                probs = self.predict(img_path)
                probs['image'] = img_name
                results.append(probs)
            except Exception as e:
                print(f"跳过 {img_name} (错误: {str(e)}")
        
        return pd.DataFrame(results)

def plot_probs(df, save_path="prediction_report.png"):
    """ 生成"专业级"非常专业！！！可视化报告 """
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 3]})
    #这素一个图表
    sample = df.iloc[0]
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166']
    bars = ax1.bar(Config.classes, sample[Config.classes], color=colors)
    ax1.set_title(f"预测结果 - {sample['image']}", fontsize=14, pad=20)
    ax1.set_ylabel('Probability (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    #这素图表上的标注
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12)

    #excel
    cell_data = df[Config.classes].applymap(lambda x: f"{x:.1f}%").values
    table = ax2.table(
        cellText=cell_data,
        rowLabels=df['image'],
        colLabels=[cls.upper() for cls in Config.classes],
        loc='center',
        cellLoc='center',
        colColours=['#FF6B6B', '#4ECDC4', '#FFD166']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # 扩大单元格
    
    ax2.axis('off')
    ax2.set_title("批量预测结果汇总", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化报告已保存至 {save_path}")

if __name__ == "__main__":
    MODEL_PATH = r"D:\trioraltermdata\trioralpictrue\best_modelto.pth"  # 替换为你的模型路径
    IMAGE_DIR = r"D:\trioraltermdata\trioralpictrue\testpictrue"# 图片文件夹路径
    
    classifier = CancerClassifier(MODEL_PATH)
    
    result_df = classifier.batch_predict(IMAGE_DIR)
    
    if not result_df.empty:
        print("\n预测结果汇总 (0-100%):")
        print(result_df)
        
        plot_probs(result_df)
        
        result_df.to_excel("predictions.xlsx", index=False, float_format="%.2f")
        print("Excel文件已保存至 predictions.xlsx")
    else:
        print("错误：未找到可处理的图片文件！")