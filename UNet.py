import time
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        # mask 文件名规则
        if "IMD" in img_name:
            mask_name = img_name.replace('.jpg', '_lesion.jpg')
        else:
            mask_name = img_name.replace('.jpg', '_segmentation.jpg')

        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 转换
        if self.transform_img:
            img = self.transform_img(img)
        else:
            img = transforms.ToTensor()(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            # 默认二值化
            mask = torch.from_numpy((mask > 128).astype(np.int64))

        return img, mask
#======================
# 1. 数据集模块
#======================
def prepare_dataset():
    # ========== 定义 transform ==========
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy((np.array(img) > 128).astype(np.int64)))
    ])


    # ========== 使用数据集 ==========
    train_dataset = ISICDataset(
        image_dir=r"E:\Project\UNet\ISIC2018\train\images\images",
        mask_dir=r"E:\Project\UNet\ISIC2018\train\masks\masks",
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    test_dataset = ISICDataset(
        image_dir=r"E:\Project\UNet\ISIC2018\test\images\images",
        mask_dir=r"E:\Project\UNet\ISIC2018\test\masks\masks",
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)
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
        self.GMP_conv3x3=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=1)
        self.max_pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    def forward(self, features):
        N, C, H, W = features.size()
        # print(f"Input features: {features.shape}")

        original_features = features.clone()
        features = self.conv5x5(features)
        #print("After conv5x5:", features.shape)
        features = self.relu(features)
        features = self.conv3x3(features)
        #print("After conv3x3:", features.shape)

        avg_features = self.avg_pool(features)
        #print("After avg_pool:", avg_features.shape)
        y = avg_features  # (N, C, 1, 1)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))  # (N, C, 1)
        # print("After bianxing:", y.shape)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (N, C, 1, 1)
        # print("After chongzu:", y.shape)
        y = torch.ones_like(avg_features)
        # print("After conv1D:", y.shape)

        eca_weight = self.conv1x1(y)
        #print("After conv1x1:", eca_weight.shape)
        eca_weighted = features * eca_weight
        #print("After ECA weighting:", eca_weighted.shape)
        weight_ECA = self.sigmoid(eca_weighted)
        #print("After sigmoid:", weight_ECA.shape)

        GMP_conv = self.GMP_conv3x3(original_features)
        #print("GMP conv3x3:", GMP_conv.shape)
        GMP_pooled = self.max_pool(GMP_conv)
        #print("After GMP maxpool:", GMP_pooled.shape)
        GMP_weighted = weight_ECA * GMP_pooled
        #print("After GMP weighting:", GMP_weighted.shape)

        output = GMP_weighted + GMP_pooled
        return output

class Decoder_RAF_module(nn.Module):
    '''解码器+RAF模块'''
    def __init__(self,in_channel,out_channel,conv_stride):
        super().__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=conv_stride)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x, encoder_feature=None):
        x_conv = self.conv(x)
        #print("After decoder conv:", x_conv.shape)
        x_conv = F.relu(F.interpolate(
            x_conv,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False))
        #print("After upsampling:", x_conv.shape)

        weight = 1 - torch.sigmoid(x_conv)
        #print("Decoder weight shape:", weight.shape)

        if encoder_feature is not None:
            x_output = encoder_feature * weight + x_conv
        else:
            x_output = x_conv
        #print("Decoder output shape:", x_output.shape)
        return x_output

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2,base_channels=64):
        super(UNet, self).__init__()
        self.conv_begin=nn.Conv2d(in_channels,base_channels,kernel_size=3,padding=1)                   #(3,64,128,128)

        # ECA+GMP 模块,编码器
        self.eca1 = ECA_GMP_module(base_channels, base_channels)
        self.eca2 = ECA_GMP_module(base_channels, base_channels * 2)
        self.eca3 = ECA_GMP_module(base_channels * 2, base_channels * 4)
        self.eca4 = ECA_GMP_module(base_channels * 4, base_channels * 8)

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
        #print(f"Input x: {x.shape}")
        x = self.conv_begin(x)
        #print(f"After conv_begin: {x.shape}")

        # ECA+GMP
        f1 = self.eca1(x)
        #print(f"After eca1: {f1.shape}")
        f2 = self.eca2(f1)
        #print(f"After eca2: {f2.shape}")
        f3 = self.eca3(f2)
        #print(f"After eca3: {f3.shape}")
        f4 = self.eca4(f3)
        #print(f"After eca4: {f4.shape}")

        # decoder
        #input=3*512*8*8
        d1 = self.dec1(f4, encoder_feature=f4)
        #print(f"After dec1: {d1.shape}")
        d2 = self.dec2(d1, encoder_feature=f3)
        #print(f"After dec2: {d2.shape}")
        d3 = self.dec3(d2, encoder_feature=f2)
        #print(f"After dec3: {d3.shape}")
        d4 = self.dec4(d3, encoder_feature=f1)
        #print(f"After dec4: {d4.shape}")
        d5 = self.dec5(d4)
        #print(f"After dec5: {d5.shape}")

        # 输出
        out = self.adjust(d5)
        #print(f"After adjust: {out.shape}")
        return out


# ====================
# 3. 损失函数 & 优化器
# ====================
def get_loss_optimizer(model, lr=1e-4, weight_decay=1e-6):
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    return criterion, optimizer

# ====================
# 4. 训练函数
# ====================
def train_model(model, train_loader, val_loader, device, criterion, optimizer,
                epochs=50, save_dir='./checkpoints', visualize=False):
    """
    UNet训练函数（带验证集准确率计算）

    参数:
        model: 待训练的UNet模型
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        device: 'cuda' 或 'cpu'
        criterion: 损失函数
        optimizer: 优化器
        epochs: 总训练轮数
        save_dir: 模型和预测结果保存路径
        visualize: 是否可视化验证集预测结果（暂不使用）
    """

    model.to(device)
    start_time = time.time()

    for epoch in range(epochs):
        # ----------------- 训练 -----------------
        model.train()
        running_loss = 0.0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if y_batch.ndim == 4:
                y_batch = y_batch.squeeze(1)  # [N,H,W] 单通道mask

            outputs = model(X_batch)  # [N, num_classes, H, W]
            loss = criterion(outputs, y_batch.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / (batch_idx + 1)

        # ----------------- 验证 -----------------
        model.eval()
        running_val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            for val_idx, (X_val, y_val, *rest) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)

                if y_val.ndim == 4:
                    y_val = y_val.squeeze(1)

                y_out = model(X_val)

                val_loss = criterion(y_out, y_val.long())
                running_val_loss += val_loss.item()

                # ---------- 像素级准确率 ----------
                pred = torch.argmax(y_out, dim=1)  # [N,H,W]
                correct_pixels += (pred == y_val).sum().item()
                total_pixels += y_val.numel()

        avg_val_loss = running_val_loss / (val_idx + 1)
        val_accuracy = correct_pixels / total_pixels

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.4f}")

    # ----------------- 训练完成后输出指标 -----------------
    # 1. 模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {num_params}")
    print("Training completed!")

# def visualize_prediction(images, masks_true, masks_pred, idx=0):
#     """
#     可视化图像分割结果
#     参数:
#         images: Tensor, shape [N, C, H, W], 输入图像
#         masks_true: Tensor, shape [N, 1, H, W], 单通道真实mask
#         masks_pred: Tensor, shape [N, 1, H, W] 或 [N, H, W], 模型预测mask（logits）
#         idx: int, 显示第 idx 张图
#     """
#     # 提取当前样本
#     image = images[idx]  # [C, H, W]
#     mask_true = masks_true[idx, 0]  # [H, W]
#
#     # 如果预测 mask 有通道维，取第 0 通道并 sigmoid
#     if masks_pred.ndim == 4:
#         mask_pred = torch.sigmoid(masks_pred[idx, 0])
#     else:
#         mask_pred = torch.sigmoid(masks_pred[idx])
#
#     # 转为 numpy，C,H,W -> H,W,C
#     if image.shape[0] == 3:
#         image_display = image.permute(1, 2, 0).cpu().numpy()
#     else:  # 单通道灰度图
#         image_display = image[0].cpu().numpy()
#
#     mask_true_display = mask_true.cpu().numpy()
#     mask_pred_display = mask_pred.cpu().detach().numpy()
#
#     # 绘图
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     axes[0].imshow(image_display)
#     axes[0].set_title("Input Image")
#     axes[0].axis('off')
#
#     axes[1].imshow(mask_true_display, cmap='gray')
#     axes[1].set_title("Ground Truth")
#     axes[1].axis('off')
#
#     axes[2].imshow(mask_pred_display, cmap='gray')
#     axes[2].set_title("Prediction")
#     axes[2].axis('off')
#
#     plt.tight_layout()
#     plt.show()


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

    # 4. 开始训练
    train_model(model, train_loader=train_loader, val_loader=val_loader,
                device=device,criterion=criterion,optimizer=optimizer,
                epochs=100, save_dir='./checkpoints', visualize=True)
