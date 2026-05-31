import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    '''
    ## Structure Loss 让模型更关注结构完整性，提升 S-measure
    '''
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    # 添加权重调整的MAE损失
    mae = torch.abs(pred - mask).mean()
    return (wbce + wiou).mean() + 0.3 * mae

def focal_loss(pred, mask, alpha=0.25, gamma=2):
    '''
    ## Focal Loss 让模型更关注细节、边界，提升 F-measure
    '''
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, mask, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = alpha * (1-pt)**gamma * bce
    return focal_loss.mean()

def dice_loss(pred, mask):
    '''
    ## 	Dice Loss 让模型更关注前景区域的一致性，提升 S-measure
    '''
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    inter = (pred * mask).sum(dim=(2,3))
    union = (pred + mask).sum(dim=(2,3))
    dice = (2 * inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def mae_loss(pred, mask):
    return torch.abs(pred - mask).mean()

def edge_aware_loss(pred, mask, edge_width=3):
    '''
    ## Edge Aware Loss 关注边缘区域，提升结构完整性和 S-measure
    '''
    # 使用Sobel算子提取边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    
    # 对真实mask提取边缘
    mask_padded = F.pad(mask, [1, 1, 1, 1], mode='replicate')
    edge_x = F.conv2d(mask_padded, sobel_x)
    edge_y = F.conv2d(mask_padded, sobel_y)
    mask_edge = torch.sqrt(edge_x**2 + edge_y**2)
    
    # 扩展边缘区域
    if edge_width > 1:
        mask_edge = F.max_pool2d(mask_edge, kernel_size=edge_width, stride=1, padding=edge_width//2)
    
    # 对预测结果进行sigmoid激活
    pred = torch.sigmoid(pred)
    
    # 计算边缘区域的损失（增加边缘权重）
    edge_weight = 1.0 + 2.0 * mask_edge
    edge_loss = F.binary_cross_entropy(pred, mask, reduction='none') * edge_weight
    
    return edge_loss.mean()

def total_loss(pred, mask, alpha=0.7, beta=0.8, gamma=0.5, delta=0.9):
    return (alpha * structure_loss(pred, mask)
            + beta * focal_loss(pred, mask)
            + gamma * dice_loss(pred, mask)
            + delta * mae_loss(pred, mask))