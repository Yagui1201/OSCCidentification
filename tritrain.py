import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from tqdm import tqdm
'''
这个是训练模型使用的文件
运行这个文件后，将得到一个pth文件，如左边我已经运行好的best_model.pth&best_modelto.pth

'''
# 配置
class Config:
    base_dir = r"D:\MedicalData\trioralpictrue\data"  # 你放train和val的根目录（即上一级目录）
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    class_folders = {
        'train': ['highrisk', 'lowrisk', 'midrisk'],
        'val': ['val_highrisk', 'val_lowrisk', 'val_midrisk']
    }

    model_name = 'efficientnet-b3'
    num_classes = 3
    input_size = 300
    
    batch_size = 16
    epochs = 30
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据加载
class OralCancerDataset(Dataset):
    def __init__(self, root_dir, class_subdirs, transform=None):
        self.samples = []
        self.class_to_idx = {'highrisk': 0, 'lowrisk': 1, 'midrisk': 2}  # 统一映射

        for class_name in class_subdirs:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            true_class = class_name.replace('val_', '')  # 去除val_前缀
            class_idx = self.class_to_idx[true_class]
            
            for img_name in os.listdir(class_dir):
                self.samples.append((
                    os.path.join(class_dir, img_name),
                    class_idx
                ))
        
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)
        img = Image.fromarray(enhanced)
        
        if self.transform:
            img = self.transform(img)
        return img, label

#训练部分
def train():
    train_transform = transforms.Compose([
        transforms.Resize((Config.input_size, Config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.input_size, Config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_data = OralCancerDataset(
        Config.train_dir, 
        Config.class_folders['train'],  
        transform=train_transform
    )
    val_data = OralCancerDataset(
        Config.val_dir,
        Config.class_folders['val'],  
        transform=val_transform
    )
    
    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size, shuffle=False)

    model = EfficientNet.from_pretrained(Config.model_name, num_classes=Config.num_classes)
    model = model.to(Config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
 
    best_acc = 0.0
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        
        for images, labels in progress_bar:
            images, labels = images.to(Config.device), labels.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{train_loss/len(progress_bar):.4f}'})

        val_acc = validate(model, val_loader)
        print(f"Val Accuracy: {val_acc:.2f}%")
  
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with acc: {best_acc:.2f}%")

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

#接口
class CancerClassifier:
    def __init__(self, model_path='best_model.pth'):
        self.model = EfficientNet.from_pretrained(Config.model_name, num_classes=Config.num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(Config.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(Config.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
        
        classes = ['highrisk', 'lowrisk', 'midrisk']
        return {
            'class': classes[pred.item()],
            'confidence': conf.item()
        }

#执行入口
if __name__ == "__main__":
    print("Starting training with your directory structure...")
    print(f"Training data from: {Config.train_dir}")
    print(f"Validation data from: {Config.val_dir}")
    train()
    classifier = CancerClassifier()
    sample_image = os.path.join(Config.val_dir, "val_highrisk/example.jpg")  # 替换为实际存在的图片
    if os.path.exists(sample_image):
        result = classifier.predict(sample_image)
        print(f"\nSample Prediction: {result}")