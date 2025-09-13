import torch
import math
import cv2
import os
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class PH2SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        image_dir: 图像文件夹
        mask_dir: mask 文件夹
        transform: albumentations 或 torchvision.transforms
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)  # e.g., IMD002.bmp
        mask_name = img_name.replace('.bmp', '_lesion.bmp')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 应用 transform
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 转为 tensor
        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask
# ====================
# 1. 加载数据集
# ====================
def prepare_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_set = PH2SegmentationDataset(
        image_dir='../Dataset/PH2_Dataset/train',  # 图像文件夹
        mask_dir='../Dataset/PH2_Dataset/test',  # mask 文件夹
        transform=transform)
    test_set = PH2SegmentationDataset(
        image_dir='../Dataset/PH2_Dataset/val_images',
        mask_dir='../Dataset/PH2_Dataset/val_masks',
        transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=0)
    test_loader = DataLoader(
        test_set,
        batch_size=2,
        shuffle=False,
        num_workers=0)
    return train_loader, test_loader


# ====================
# 2. 模型模块
# ====================
class ECA_GMP_module(nn.Module):
    """高效注意力通道"""
    def __init__(self,in_channel,out_channel,k_size=None,gamma=2,b=1):
        '''
        :param channel: 根据输入通道数设置
        :param k_size: 卷积核不固定,与输入通道数有关
        :param gamma: 控制 核大小随通道数增长的“斜率”
        :param b: 控制 整体偏移量，相当于“基准核大小”
        '''
        super().__init__()
        # 动态设置 kernel size
        if k_size is None:
            t = int(abs((math.log2(in_channel) / gamma) + b))
            k = t if t % 2 else t + 1
        else:
            k = k_size if (k_size % 2 == 1) else k_size + 1
        #-------特征提取层--------
        self.conv5x5=nn.Conv2d(in_channel,out_channel,kernel_size=5,stride=2,padding=2)
        self.relu=nn.ReLU(inplace=True)
        self.conv3x3=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,stride=1)
        #-------ECA注意力层--------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (N,C,1,1),自适应平均池化
        self.conv1d = nn.Conv1d(1, 1,kernel_size=k,padding=(k - 1) // 2,bias=False)
        self.conv1x1 = nn.Conv2d(out_channel, out_channel,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        #-------GMP网络--------
        self.GAP_conv3x3=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,stride=1)
        self.max_pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    def forward(self, features):
        """
            输入 features: (N, C, H, W)
            返回: ECA+GMP 融合后的特征图
        """
        N, C, H, W = features.size()

        # ---- 两个空间卷积层 ----
        features = self.conv5x5(features) #-→(N, C, H , W )
        features = self.relu(features)
        features = self.conv3x3(features) #-→(N, C, H, W)

        # ---- 全局平均池化 ----
        avg_features = self.avg_pool(features)  # -> (N, C, 1, 1)
        avg_features = avg_features.view(N, C).unsqueeze(1)  # (N, 1, C)

        # ---- Conv1d：沿通道方向交互 ----
        avg_features = self.conv1d(avg_features)  # (N, 1, C)
        avg_features = avg_features.squeeze(1).view(N, C, 1, 1)  # (N, C, 1, 1)

        # ---- Conv1x1 映射 ----
        eca_weight = self.conv1x1(avg_features)  # (N, C, 1, 1)
        eca_weighted = features * eca_weight  # 加权

        # ---- Sigmoid 归一化 ----
        weight_ECA = self.sigmoid(eca_weighted) #(N,C,1,1)

        # ---- GMP 分支 ----
        GMP_conv = self.conv3x3(features) #(N,C,H,W)
        GMP_pooled = self.max_pool(GMP_conv)#(N,C,H/2,W/2)
        GMP_weighted = weight_ECA * GMP_pooled #(N,C,H/2,W/2)

        # ---- 融合输出 ----
        output = GMP_weighted + GMP_pooled#(N,C,H/2,W/2)
        return output

class Decoder_RAF_module(nn.Module):
    '''解码器+RAF模块'''
    def __init__(self,in_channel,out_channel,conv_stride):
        super().__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=conv_stride)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x, encoder_feature=None):
        # 卷积 + 上采样 + ReLU
        x_conv = F.relu(F.interpolate(self.conv(x), #(N,C,H,W)
                                      scale_factor=(2,2),#(N,C,H*2,W*2)
                                      mode='bilinear',
                                      align_corners=False))

        # 反注意力权重
        weight = 1 - torch.sigmoid(x_conv)  # 等价于 -sigmoid(x) + 1

        # 融合编码器特征
        if encoder_feature is not None:
            x_output = encoder_feature * weight + x_conv
        else:
            x_output = x_conv
        return x_output

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2,base_channels=64):
        super(UNet, self).__init__()
        self.conv_begin=nn.Conv2d(in_channels,base_channels,kernel_size=3,padding=1)                   #(3,64,128,128)

        # ECA+GMP 模块,编码器
        self.eca1 = ECA_GMP_module(base_channels, base_channels)
        self.eca2 = ECA_GMP_module(base_channels * 2, base_channels * 2)
        self.eca3 = ECA_GMP_module(base_channels * 4, base_channels * 4)
        self.eca4 = ECA_GMP_module(base_channels * 8, base_channels * 8)

        # 解码器
        self.dec1 = Decoder_RAF_module(base_channels * 8, base_channels * 8, conv_stride=2)
        self.dec2 = Decoder_RAF_module(base_channels * 8, base_channels * 4, conv_stride=1)
        self.dec3 = Decoder_RAF_module(base_channels * 4, base_channels * 2, conv_stride=1)
        self.dec4 = Decoder_RAF_module(base_channels * 2, base_channels, conv_stride=1)
        self.dec5 = Decoder_RAF_module(base_channels,int(base_channels*0.5), conv_stride=1)

        # 输出
        self.adjust = nn.Conv2d(int(base_channels*0.5), num_classes, kernel_size=1,stride=1)

    def forward(self, x):

        #encoder
        #input=3*3*128*128
        x=self.conv_begin(x)  #[3*64*128*128]

        # ECA+GMP
        f1 = self.eca1(x)     #[3*64*64*64]
        f2 = self.eca2(f1)    #[3*128*32*32]
        f3 = self.eca3(f2)    #[3*256*16*16]
        f4 = self.eca4(f3)    #[3*512* 8* 8]

        # decoder
        #input=3*512*8*8
        d1 = self.dec1(f4, encoder_feature=f4) #[3*512*8*8]
        d2 = self.dec2(d1, encoder_feature=f3) #[3*256*16*16]
        d3 = self.dec3(d2, encoder_feature=f2) #[3*128*32*32]
        d4 = self.dec4(d3, encoder_feature=f1) #[3*64*64*64]
        d5 = self.dec5(d4)                     #[3*32*128*128]

        # 输出
        out = self.adjust(d5)                  #[3*2*128*128]
        return out


# ====================
# 3. 损失函数 & 优化器
# ====================
def get_loss_optimizer(model, lr=1e-3):
    return nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr)


# ====================
# 4. 训练函数
# ====================
def train_model(model, train_loader, val_loader, device, criterion, optimizer, epochs=20, save_dir='./checkpoints', visualize=False):
    """
    UNet训练函数

    参数:
        model: 待训练的UNet模型
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        device: 'cuda' 或 'cpu'
        criterion: 损失函数
        optimizer: 优化器
        epochs: 总训练轮数
        save_dir: 模型和预测结果保存路径
        visualize: 是否可视化验证集预测结果
    """

    model.to(device)

    for epoch in range(epochs):
        model.train()  # 设置训练模式
        running_loss = 0.0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)

            # 计算损失
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / (batch_idx + 1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # ===== 验证 & 保存 =====
        model.eval()
        with torch.no_grad():
            for val_idx, (X_val, y_val, *rest) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_out = model(X_val)

                # 可视化
                if visualize and val_idx == 0:
                    visualize_prediction(X_val, y_val, y_out, idx=0)

                # 二值化预测
                pred = y_out.detach().cpu().numpy()
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0

                label = y_val.detach().cpu().numpy()
                label[label > 0] = 1
                label[label <= 0] = 0

                pred, label = pred.astype(np.uint8), label.astype(np.uint8)
                pred[pred == 1] = 255
                label[label == 1] = 255

                fulldir = os.path.join(save_dir, str(epoch))
                os.makedirs(fulldir, exist_ok=True)

                if rest and isinstance(rest[0][0], str):
                    filename = rest[0][0]
                else:
                    filename = f"{val_idx+1:03}.png"

                cv2.imwrite(os.path.join(fulldir, filename), pred[0,1,:,:])

    print("Training completed!")

# ====================
# 5. 可视化
# ====================
def visualize_prediction(X, y_true, y_pred, idx=0):
    """
    可视化输入图像、真实mask和预测mask

    参数:
        X: 输入图像 tensor, shape=[B, C, H, W]
        y_true: 真实mask tensor, shape=[B, 1 或 num_classes, H, W]
        y_pred: 预测mask tensor, shape=[B, num_classes, H, W]
        idx: 可视化第几个样本
    """
    # 转为 numpy
    img = X[idx].cpu().numpy().transpose(1, 2, 0)  # HWC
    img = (img - img.min()) / (img.max() - img.min())  # 归一化到0-1

    mask_true = y_true[idx].cpu().numpy()
    if mask_true.shape[0] > 1:  # 如果是one-hot
        mask_true = np.argmax(mask_true, axis=0)

    mask_pred = y_pred[idx].detach().cpu().numpy()
    if mask_pred.shape[0] > 1:
        mask_pred = np.argmax(mask_pred, axis=0)

    # 二值化显示（可根据需要调整）
    mask_true_display = mask_true * 255
    mask_pred_display = mask_pred * 255

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(mask_true_display, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(mask_pred_display, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.show()

# ====================
# 6. 主函数
# ====================
if __name__ == "__main__":
    # 1. 加载数据集
    train_loader, val_loader = prepare_dataset()

    # 2. 创建模型
    model = UNet(in_channels=3, num_classes=2)

    #3.损失函数和优化器
    criterion, optimizer = get_loss_optimizer(model)

    # 3. 设置训练参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20


    # 4. 开始训练
    train_model(model, train_loader=train_loader, val_loader=val_loader,
                device=device,criterion=criterion,optimizer=optimizer,
                epochs=20, save_dir='./checkpoints', visualize=True)
