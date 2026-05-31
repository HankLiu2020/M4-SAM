import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Dict
from torch import Tensor
from memsam.modeling.mem import Mem
from memsam.modeling.mem_modules import HiddenReinforcer  # 导入原生的HiddenReinforcer
class MemModel(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any] ,
    ) -> None:
        super().__init__()
        self.image_encoder = PassEncoder()

        self.memory=Mem(config)
        #对key进行编码以压缩特征
        self.memory.key_encoder =KeyEncoder(self.memory.feat_dim, self.memory.key_dim)
        #用于从Feature获取imge
        
        use_lightweight_value_encoder = config.get('use_lightweight_value_encoder', False)
        use_sam_based_value_encoder = config.get('use_sam_based_value_encoder', False)
        use_attention_based_value_encoder = config.get('use_attention_based_value_encoder', False)
        
        if use_sam_based_value_encoder:
            # 使用基于SAM embedding的Value Encoder
            self.memory.value_encoder = SAMBasedValueEncoder(
                value_dim=self.memory.value_dim,
                hidden_dim=self.memory.hidden_dim,
                single_object=True
            )
        elif use_attention_based_value_encoder:
            # 使用注意力机制的SAM-based Value Encoder
            self.memory.value_encoder = AttentionBasedValueEncoder(
                value_dim=self.memory.value_dim,
                hidden_dim=self.memory.hidden_dim,
                single_object=True
            )
        elif use_lightweight_value_encoder:
            # 使用轻量化的Value Encoder
            self.memory.value_encoder = LightweightValueEncoder(
                value_dim=self.memory.value_dim,
                hidden_dim=self.memory.hidden_dim,
                single_object=True
            )

        self.mask_decoder = MaskDecoderNoSkip(self.memory.value_dim,64, self.memory.feat_dim)
    def forward(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        features: torch.Tensor, # [b,t,c,h,w]#直接输入SAM Encoder+下游decoder得到的特征
        alter_gt#用于替用Xmem中的首帧GT# b c h w
    ) -> torch.Tensor:
        b, t, c, h, w = features.shape  # b t c h w
        # encode imgs to imgs embedding
        # feature承接sam的输出，key承接编码后proj的特征
        key, shrinkage, selection, imge = self.memory.encode_key(features)
        # init memory,在xmem结构图中表示为ht隐特征
        hidden = torch.zeros((b, 1, self.memory.hidden_dim, *key.shape[-2:])).to(imge.device)
        frames_pred = []
        # first frame

        mask = alter_gt # b c h w
        # frames_pred.append(mask)
        values_0, hidden = self.memory.encode_value2(None, features[:,0], hidden, mask)
        values = values_0[:,:,:,:0]

        # process frames
        for ti in range(0, t):
            if ti == 0 :#首帧
                ref_keys = key[:,:,[0]] 
                ref_shrinkage = shrinkage[:,:,[0]]
                ref_values = values_0 
            else:
                ref_keys = key[:,:,:ti]
                ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                ref_values = values

            # get single frame
            # read memory
            memory_readout = self.memory.read_memory(
                key[:, :, ti],
                selection[:, :, ti] if selection is not None else None,
                ref_keys, ref_shrinkage, ref_values)
            #ref keys即为用于memory读取的key，和Value成对，ref_shrinkage和ref_values为memory的shrinkage和value
            # generate memory embedding
            # 在decode步骤中，使用fuser做cbma实现Hidden和readout的融合
            hidden, memory_e = self.memory.decode(imge[:,ti], hidden, memory_readout)
            #hidden 2x1x64x44x44 memory_readout 2x1x256x44x44

            # decode
            #在memSAM中用的是SAM Decoder，解码出memory embedding:本方法再构建mask_decoder,将memory得到的特征还原为mask
            mask = self.mask_decoder(memory_e, features[:,ti,:,:,:]) #b n h w->b 1 h w
            mask = torch.sigmoid(mask)

            frames_pred.append(mask) #t c h w

            # last frame no encode
            if ti < t-1:
                # update memory
                is_deep_update = np.random.rand() < 0.5 #0.2
                #如果深度更新，就会用到mem_modules中的HiddenReinforcer
                v16, hidden = self.memory.encode_value2(imgs[:,ti], features[:,ti], hidden, mask, is_deep_update=is_deep_update)
                values = torch.cat([values, v16], 3)

        pred = torch.stack(frames_pred, dim=1) # b t c h w

        return pred

class PassEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class KeyEncoder(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.encoder(x)

class LightweightValueEncoder(nn.Module):
    """轻量化的Value Encoder，替代完整的ResNet18"""
    def __init__(self, value_dim=128, hidden_dim=64, single_object=True):
        super().__init__()
        self.single_object = single_object
        self.value_dim = value_dim
        
        # 轻量化的编码器结构
        input_channels = 4 if single_object else 5  # RGB + mask (+ others)
        
        # 使用更小的网络结构
        self.encoder = nn.Sequential(
            # 第一层：4->32, 1/2
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第二层：32->64, 1/4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层：64->128, 1/8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第四层：128->value_dim, 1/16
            nn.Conv2d(128, value_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(value_dim),
            nn.ReLU(inplace=True),
        )
        
        # 特征融合层
        self.fuser = nn.Sequential(
            nn.Conv2d(value_dim + 64, value_dim, kernel_size=1),  # 64是imge的通道数
            nn.BatchNorm2d(value_dim),
            nn.ReLU(inplace=True),
        )
        
        # Hidden reinforcer - 使用原生实现
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None
            
        self._initialize_weights()
    
    def forward(self, image, image_feat, h, masks, others, is_deep_update=True):
        # 拼接输入
        if self.single_object:
            x = torch.cat([image, masks.unsqueeze(1)], dim=1)
        else:
            x = torch.cat([image, masks.unsqueeze(1), others.unsqueeze(1)], dim=1)
        
        # 编码
        g = self.encoder(x)
        
        # 特征融合
        g = self.fuser(torch.cat([g, image_feat], dim=1))
        
        # 转换为原生HiddenReinforcer期望的格式 [batch, num_objects, channels, height, width]
        g = g.unsqueeze(1)  # [batch, 1, channels, height, width]
        
        # Hidden更新
        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)
            
        # 转换回原始格式
        g = g.squeeze(1)  # [batch, channels, height, width]
            
        return g, h
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SAMBasedValueEncoder(nn.Module):
    """基于SAM embedding的Value Encoder，适配128通道"""
    def __init__(self, value_dim=128, hidden_dim=64, single_object=True):
        super().__init__()
        self.single_object = single_object
        self.value_dim = value_dim
        
        # 直接基于SAM embedding进行mask增强
        self.mask_enhancer = nn.Sequential(
            # 将mask信息融合到SAM embedding中
            nn.Conv2d(128 + 1, 128, kernel_size=3, padding=1),  # 128维SAM embedding + 1维mask
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
            # 进一步提取mask-aware特征
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 投影到value维度
        self.value_proj = nn.Sequential(
            nn.Conv2d(128, value_dim, kernel_size=1),
            nn.BatchNorm2d(value_dim),
            nn.ReLU(inplace=True),
        )
        # Hidden reinforcer原生实现
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None
            
        self._initialize_weights()
    
    def forward(self, image, image_feat, h, masks, others, is_deep_update=True):
        # image_feat是SAM embedding (128维) masks是bchw
        # 直接使用SAM embedding，不重新编码RGB图像
        
        # 将mask信息融合到SAM embedding中
        mask_resized = F.interpolate(masks, size=image_feat.shape[-2:], mode='bilinear', align_corners=False)
        enhanced_feat = torch.cat([image_feat, mask_resized], dim=1)
        # 给Embedding和mask混合回到128c
        g = self.mask_enhancer(enhanced_feat)
        # 投影到value维度
        g = self.value_proj(g)
        # 转换为原生HiddenReinforcer期望的格式 [batch, num_objects, channels, height, width]
        g = g.unsqueeze(1)  # [batch, 1, channels, height, width]
        # Hidden更新
        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)
        # 转换回原始格式
        return g, h
        #vaule: B * num_objects * ChannelValue * H * W  ,hidden
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AttentionBasedValueEncoder(nn.Module):
    """使用注意力机制的SAM-based Value Encoder，适配128通道"""
    def __init__(self, value_dim=128, hidden_dim=64, single_object=True):
        super().__init__()
        self.single_object = single_object
        self.value_dim = value_dim
        # 空间注意力：mask引导的注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 通道注意力：自适应特征选择
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1),
            nn.Sigmoid()
        )
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(128, value_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(value_dim),
            nn.ReLU(inplace=True),
        )
        # Hidden reinforcer原生实现
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None
            
        self._initialize_weights()
    
    def forward(self, image, image_feat, h, masks, others, is_deep_update=True):
        # 空间注意力：mask引导
        mask_resized = F.interpolate(masks, size=image_feat.shape[-2:], mode='bilinear', align_corners=False)
        spatial_attn = self.spatial_attention(mask_resized)
        # 通道注意力：自适应特征选择
        channel_attn = self.channel_attention(image_feat)
        # 应用注意力
        g = image_feat * spatial_attn * channel_attn
        # 特征融合
        g = self.fusion(g)
        # 转换为原生HiddenReinforcer期望的格式 [batch, num_objects, channels, height, width]
        g = g.unsqueeze(1)  # [batch, 1, channels, height, width]
        # Hidden更新
        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)            
        return g, h
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MaskDecoderNoSkip(nn.Module):
    """无跳接版本的MaskDecoder，依旧使用反卷积保持相同尺寸输出"""
    def __init__(self, input_dim=256, output_dim=64, features_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.features_dim = features_dim

        # H/8 44 -> H/4 88 256c->128c
        self.deconv1 = nn.ConvTranspose2d(input_dim, features_dim, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(features_dim)
        self.act1 = nn.GELU()

        # H/4 88 -> H/2 176 128c->64c
        self.deconv2 = nn.ConvTranspose2d(features_dim, features_dim // 2, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(features_dim // 2)
        self.act2 = nn.GELU()

        # H/2 176 -> H 352 64c->32c
        self.deconv3 = nn.ConvTranspose2d(features_dim // 2, features_dim // 4, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(features_dim // 4)
        self.act3 = nn.GELU()

        # 最终预测层 32c->1c
        self.final_conv = nn.Conv2d(features_dim // 4, 1, kernel_size=3, padding=1)
        self._initialize_weights()

    def forward(self, x, features):
        # x: [b, n, c, h, w]
        # features参数保留但不使用，保持接口一致
        batch_size, channel_num, _, feat_size, _ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)  # [b*n, c, h, w]

        # 反卷积1
        x1 = self.deconv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        # 反卷积2
        x2 = self.deconv2(x1)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)

        # 反卷积3
        x3 = self.deconv3(x2)
        x3 = self.bn3(x3)
        x3 = self.act3(x3)

        # 最终输出
        logits = self.final_conv(x3)
        logits = logits.view(batch_size, channel_num, *logits.shape[-2:])
        return logits   # [b, n, h, w]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
