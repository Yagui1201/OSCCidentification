import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
'''
这个文件只是用于测定一下咱模型的一个准确度，不在目标使用范围内
这个文件运行后，会得到一张图片，如validation_result.png
表示的是你测试的这个模型它的精确度，他的对角线颜色越深，越精确
'''
#配置
class Config:

    val_dir = r"D:\trioraltermdata\trioralpictrue\data\val" 
    

    model_path = r"D:\trioraltermdata\trioralpictrue\best_modelto.pth"  
    

    val_classes = ['val_highrisk', 'val_lowrisk', 'val_midrisk']  
    

    model_name = 'efficientnet-b3'
    input_size = 300
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据集
class ValDataset(Dataset):
    def __init__(self, root_dir, class_subdirs, transform=None):
        self.samples = []
        self.class_to_idx = {'val_highrisk': 0, 'val_lowrisk': 1, 'val_midrisk': 2}  # 映射到0,1,2
        
        for class_name in class_subdirs:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.samples.append((
                    os.path.join(class_dir, img_name),
                    self.class_to_idx[class_name]  
                 ))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

#val预处理
val_transform = transforms.Compose([
    transforms.Resize((Config.input_size, Config.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#验证
def validate():

    model = EfficientNet.from_pretrained(Config.model_name, num_classes=3)
    model.load_state_dict(torch.load(Config.model_path))
    model = model.to(Config.device)
    model.eval()
    
    val_data = ValDataset(Config.val_dir, Config.val_classes, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ['highrisk', 'lowrisk', 'midrisk']
    print("\n=== 验证结果 ===")
    print(f"加权F1分数: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"准确率: {np.mean(np.array(all_labels) == np.array(all_preds)):.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig('validation_result.png', bbox_inches='tight', dpi=300)
    plt.show()

    print("\n=== 各类别表现 ===")
    for i, name in enumerate(class_names):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        print(f"{name}: 精确率={precision:.3f}, 召回率={recall:.3f}, F1={2*(precision*recall)/(precision+recall+1e-6):.3f}")

if __name__ == "__main__":

    assert os.path.exists(Config.model_path), f"模型文件不存在: {Config.model_path}"
    assert os.path.exists(Config.val_dir), f"验证集路径不存在: {Config.val_dir}"
    
    print("开始验证...")
    print(f"模型位置: {Config.model_path}")
    print(f"验证数据: {Config.val_dir}")
    validate()