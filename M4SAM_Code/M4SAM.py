import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import argparse


import numpy as np
from typing import Any, Dict, List, Tuple
import os,copy,math

from Network.xmem4sam import MemModel
from Network.finetune_utils import LoRA,Adapter,InjectLoRAmoeLayer,InjectLoRAmoeLayer_bypass,InjectModalitySpecificMoE,InjectModalitySpecificMoE_v2
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class UIM(nn.Module):
    '''from DCTNet'''
    def __init__(self, in_planes):
        super(UIM, self).__init__()
        self.catconv1 = BasicConv2d(in_planes=in_planes * 2, out_planes=in_planes, kernel_size=3,
                                    padding=1, stride=1)
        self.catconv2 = BasicConv2d(in_planes=in_planes * 4, out_planes=in_planes, kernel_size=3,
                                    padding=1, stride=1)

    def forward(self, input1, input2):
        cat_out = self.catconv1(torch.cat([input1, input2], dim=1))
        mul_out = input1 * input2
        sub_out = input1 - input2
        max_put = torch.maximum(input1, input2)
        interactive_out = self.catconv2(torch.cat([cat_out, mul_out, sub_out, max_put], dim=1))

        return interactive_out

class MultiLevelFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.catconv = nn.Sequential(
            nn.Conv2d(256, 64, 1,bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        #self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        Xc = torch.cat(feature_list, dim = 1).contiguous()
        X1=feature_list[0]
        Xc_64=self.catconv(Xc)
        Pg=self.pool(Xc_64)
        return X1+Pg*Xc_64

class Gated_MLF(nn.Module):
    '''
    ## 在原先的MultiLevelFeatureFusion基础上增加了通道注意力与门控残差
    '''
    def __init__(self):
        super().__init__()
        self.catconv = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 通道注意力模块
        self.channel_fc = nn.Sequential(
            nn.Conv2d(64, 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 64, 1, bias=False)
        )
        # 空间注意力模块
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1, bias=False)
        )
        # 门控残差模块
        self.gate_conv = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_list):
        # 融合多尺度特征
        Xc = torch.cat(feature_list, dim=1).contiguous()
        X1 = feature_list[0]
        Xc_64 = self.catconv(Xc)
        # 通道注意力
        channel_att = self.pool(Xc_64)
        channel_att = self.channel_fc(channel_att)
        channel_att = torch.sigmoid(channel_att)
        # 空间注意力
        spatial_att = torch.mean(Xc_64, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(spatial_att)
        spatial_att = torch.sigmoid(spatial_att)
        # 加权特征
        enhanced_feat = Xc_64 * channel_att * spatial_att
        # 门控残差连接
        gate = self.gate_conv(torch.cat([X1, enhanced_feat], dim=1))
        gate = torch.sigmoid(gate)
        # 最终输出：门控残差连接
        return gate * enhanced_feat + (1 - gate) * X1

class M4SAM(nn.Module):
    '''
    ## 输出：out1 out2 out mem_out moe_loss
    ## 其中out1是最底层输出
    '''
    def __init__(self, args,checkpoint_path=None,device="cuda") -> None:
        super(M4SAM, self).__init__()    
        self.args=args
        #判断SAM版本
        if args.sam==20:
            model_cfg = "sam2_hiera_l.yaml"
        elif args.sam==21:
            model_cfg = "sam2.1_hiera_l.yaml"
        
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path,device=device)
        else:#测试情况，不加载SAM预训练权重，整体载入
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if args.merge_branch==0:#不合并这两个模态的Encoder分支
            self.encoder_d = copy.deepcopy(self.encoder)
            for param in self.encoder_d.parameters():
                param.requires_grad = False
        
        if args.lora==-1:#纯SAM baseline，完全冻结encoder
            # encoder已经在第246行冻结
            # 不添加任何adapter/lora/moe模块
            pass  # 保持原始SAM encoder不变
        elif args.lora==0:#采用Adapter
            blocks = []
            for block in self.encoder.blocks:
                blocks.append(
                    Adapter(block)
                )
            self.encoder.blocks = nn.Sequential(
                *blocks
            )
            if args.merge_branch==0:#不合并
                blocks = []
                for block in self.encoder_d.blocks:
                    blocks.append(
                        Adapter(block)
                    )
                self.encoder_d.blocks = nn.Sequential(
                    *blocks
                )
        elif args.lora==1:#采用LoRA
            build_lora=LoRA(rank=args.rank)
            self.encoder.blocks = build_lora(self.encoder.blocks)
            if not args.merge_branch:#不合并
                self.encoder_d.blocks = build_lora(self.encoder_d.blocks)
        elif args.lora==2:#采用conv+lora+moe,默认合并分支  
            self.encoder=InjectLoRAmoeLayer_bypass(
                self.encoder,"conv_lora_DSC",rank=args.rank,lora_alpha=1,expert_num=args.expert)
        elif args.lora==21:
            # 消融实验，采用论文中提出的conLoRA
            # https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
            self.encoder=InjectLoRAmoeLayer_bypass(
                self.encoder,"conv_lora",rank=args.rank,lora_alpha=1,expert_num=args.expert)
        elif args.lora==3:#采用模态特定的MoE
            self.encoder=InjectModalitySpecificMoE(
                self.encoder, rank=args.rank, lora_alpha=1, expert_per_modality=args.expert)
        elif args.lora==4:#采用改进版模态特定的MoE
            self.encoder=InjectModalitySpecificMoE_v2(
                self.encoder, rank=args.rank, lora_alpha=1, expert_per_modality=args.expert)
        else:
            raise ValueError("Invalid lora parameter")

        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        self.uim1 = UIM(144)
        self.uim2 = UIM(288)
        self.uim3 = UIM(576)
        self.uim4 = UIM(1152)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        
        self.edge1 = nn.Conv2d(64, 1, kernel_size=1)
        self.edge2 = nn.Conv2d(64, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(64, 1, kernel_size=1)



        if self.args.nomem == 0:# 在nomem=1消融mem模块时，不对mem进行声明
            self.mem_model = MemModel(config={
                    "key_dim": 64, #128,
                    "value_dim": 256,
                    "hidden_dim": 64,
                    "feat_dim":128,
                    "reinforce":False,
                    "single_object":True,
                    "use_sam_based_value_encoder":True,
                    "use_lightweight_value_encoder":False,
                    "use_attention_based_value_encoder":False
                
                })
            self.mem_model.cuda()
        else:
            self.mem_model = None
        #self.mlf=args.mlf
        if args.mlf==0:
            pass
        elif args.mlf==1:
            self.MLF = MultiLevelFeatureFusion() 
        elif args.mlf==2:
            self.MLF = Gated_MLF()
    def forward(self,video,video_depth,reshape=False):
        #img: (B,3,H,W)#depth: (B,1,H,W)#video: (B,T,3,H,W)#video_depth: (B,T,1,H,W)
        #数据结构为算法服务：给Video里的batch和t拉平
        batch_size, num_frame = video.shape[:2]
        imgs=video.flatten(start_dim=0, end_dim=1)
        depth=video_depth.flatten(start_dim=0, end_dim=1)

        if self.args.lora==-1:#纯SAM baseline，完全冻结encoder
            # 直接使用冻结的encoder进行前向传播
            x1, x2, x3, x4,_ = self.encoder(imgs)
            d1, d2, d3, d4,_ = self.encoder(depth.repeat(1, 3, 1, 1))
            moe_loss = torch.tensor(0.0).cuda()  # 无MoE损失
        elif self.args.lora==2:#采用conv+lora+moe,默认合并分支,输出moe_loss
            x1, x2, x3, x4,moe_loss = self.encoder(imgs)            
            if depth.size(1) == 1:
                input_depth = depth.repeat(1, 3, 1, 1)  # Project到3通道
            else:
                input_depth = depth
            d1, d2, d3, d4, moe_loss_d = self.encoder(input_depth)

            moe_loss += moe_loss_d           
        elif self.args.lora==3 or self.args.lora==4:#采用模态特定的MoE
            # 处理RGB模态 - 先设置模态类型
            for layer in self.encoder.blocks:
                if hasattr(layer.attn.qkv, 'set_modality_type'):
                    layer.attn.qkv.set_modality_type('rgb')
            # 然后调用forward方法
            x1, x2, x3, x4, moe_loss_rgb = self.encoder(imgs)
            
            # 处理深度模态 - 先设置模态类型
            for layer in self.encoder.blocks:
                if hasattr(layer.attn.qkv, 'set_modality_type'):
                    layer.attn.qkv.set_modality_type('depth')
            # 然后调用forward方法
            if self.args.no_depth==2:#使用video替代depth，不再进行forward
                d1, d2, d3, d4, moe_loss_depth=x1, x2, x3, x4, moe_loss_rgb
            else:#正常模式
                d1, d2, d3, d4, moe_loss_depth = self.encoder(depth.repeat(1,3,1,1))
            
            # 计算总的MoE损失
            moe_loss = moe_loss_rgb + moe_loss_depth
        elif self.args.merge_branch:#使用同一个Encoder编码，通过内部moe判断 早期版本，这样的情况不用lora
            x1, x2, x3, x4,moe_loss = self.encoder(imgs)
            d1, d2, d3, d4,moe_loss_d = self.encoder(depth.repeat(1,3,1,1))
            moe_loss += moe_loss_d  
        else:#对两个模态使用不同的编码器，分别进行编码
            x1, x2, x3, x4,moe_loss = self.encoder(imgs)
            d1, d2, d3, d4,moe_loss_d = self.encoder_d(depth.repeat(1,3,1,1))
            moe_loss += moe_loss_d  
        
        x1= self.uim1(x1,d1)
        x2 = self.uim2(x2, d2)
        x3 = self.uim3(x3, d3)
        x4 = self.uim4(x4, d4)

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        edge1 = F.interpolate(self.edge1(x), scale_factor=16, mode='bilinear')
        x_up2 = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x_up2), scale_factor=8, mode='bilinear')
        edge2 = F.interpolate(self.edge2(x_up2), scale_factor=8, mode='bilinear')
        x = self.up3(x_up2, x1)#最顶层，up3进行双卷积
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        edge_out = F.interpolate(self.edge_head(x), scale_factor=4, mode='bilinear')
        
        out1= out1.view(batch_size, num_frame, *out1.shape[1:])
        out2= out2.view(batch_size, num_frame, *out2.shape[1:])
        out = out.view(batch_size, num_frame, *out.shape[1:])
        edge1 = edge1.view(batch_size, num_frame, *edge1.shape[1:])
        edge2 = edge2.view(batch_size, num_frame, *edge2.shape[1:])
        edge_out = edge_out.view(batch_size, num_frame, *edge_out.shape[1:])
        
        
        # 仅在nomem=0时执行Memory计算
        if self.mem_model is not None:
            '''为mem进行顶层特征融合：x1与下游特征
            为节省mem的显存，遵循下游特征大小44x44'''
            # 使用预定义的下采样层
            if self.args.mlf:
                x_fusion=self.MLF([x1, F.interpolate(x2, scale_factor=2, mode='bilinear'), 
                                 F.interpolate(x3, scale_factor=4, mode='bilinear'),
                                 F.interpolate(x4, scale_factor=8, mode='bilinear')])
                f1=x_fusion#（基于x1）
                f2=x_up2
            else:#args.mlf=0 
                f1=x1
                f2=x_up2

            #f1 f2分别是Memory模块的输入特征的两个操作数
            if self.args.resize44:#(B, 64, 44, 44)
                # f1 = self.conv_down(f1)  # 原卷积下采样：64x88x88 -> 64x44x44
                f1 = F.interpolate(f1, size=(44, 44), mode='bilinear', align_corners=True)  # 双线性插值下采样：64x88x88 -> 64x44x44
                f1=f1.view(batch_size, num_frame,*f1.shape[1:])
                #f1:64x44x44 f2:64x44x44
                f2=f2.view(batch_size, num_frame, *f2.shape[1:])#复用up2的特征
                features = torch.concat([f1,f2], dim=2)#3 4 128 44 44
            else: 
                f1 = f1.view(batch_size, num_frame, *f1.shape[1:])
                #f1:64x88x88 f2:64x44x44
                f2 = F.interpolate(f2, scale_factor=2, mode='bilinear')
                f2=f2.view(batch_size, num_frame, *f2.shape[1:])
                features = torch.concat([f1,f2], dim=2)#3 4 128 88 88

            '''Xmem中使用的mask是0-1的灰度'''
            alter_gt=out[:,0,:]
            alter_gt = torch.sigmoid(alter_gt)
            alter_gt=(alter_gt-alter_gt.min())/(alter_gt.max()-alter_gt.min())
            
            #仅需要首帧伪gt，用于Xmem的value Encoder
            #with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True if self.args.amp==1 else False):
            mem_out=self.mem_model(video,features,alter_gt)
            # 如果输入图像是704尺寸，缩小到352尺寸
            if not self.args.resize44:
                mem_out = F.interpolate(
                    mem_out.flatten(start_dim=0, end_dim=1),
                    size=(352, 352),
                    mode='bilinear'
                ).view(batch_size, num_frame,1, 352, 352)
        else:
            # nomem=1时，直接返回与out形状相同的虚拟tensor，跳过所有Memory相关计算
            mem_out = out.detach().clone()
        if not reshape:
            return  out1.flatten(start_dim=0, end_dim=1),out2.flatten(start_dim=0, end_dim=1),\
                    out.flatten(start_dim=0, end_dim=1),mem_out.flatten(start_dim=0, end_dim=1),moe_loss,\
                    edge1.flatten(start_dim=0, end_dim=1),edge2.flatten(start_dim=0, end_dim=1),\
                    edge_out.flatten(start_dim=0, end_dim=1)
        else:
            return  out1,out2,out,mem_out,moe_loss,edge1,edge2,edge_out
