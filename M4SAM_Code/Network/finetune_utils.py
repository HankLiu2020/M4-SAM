import torch
import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
import math
#from typing import Any, Dict

from .adaptation_layers import ConvLoRALinear,ConvLoRALinear_bypass,ConvLoRALinear_bypass_DSC,ModalitySpecificMoE,ModalitySpecificMoE_v2
from .adaptation_layers import IA3Linear, IA3LoRALinear, LoRALinear

def create_adaptation(peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.
    Parameters:
    peft
        Name of the adaptation module.
    layer
       The layer the adaptation module should be applied to.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules
    Returns
    -------
    Model with injected LoRA modules.
    """
    if "ia3_lora" in peft:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "conv_lora" in peft:
        #for example in 144 out 864
        return ConvLoRALinear(
            layer.in_features,
            layer.out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        )
    elif "ia3" in peft:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in peft:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
        )

def InjectLoRAmoeLayer(model:nn.modules,peft_type="conv_lora",rank=4,lora_alpha=1.0,expert_num=4):
    #并非所有Blocks都需要lora
    for layer_i, layer in enumerate(model.blocks):
        if layer_i not in [2,8,10,20,25,30,35,40,47]:
            continue
        adaptation_layer = create_adaptation(peft_type, layer.attn.qkv, rank, lora_alpha, conv_lora_expert_num=expert_num)
        adaptation_layer.weight = layer.attn.qkv.weight
        adaptation_layer.bias = layer.attn.qkv.bias
        layer.attn.qkv = adaptation_layer
        
    return model

def InjectLoRAmoeLayer_bypass(model:nn.modules,peft_type="conv_lora",rank=4,lora_alpha=1.0,expert_num=4):
    #由于Hiera中使用了打包式qkv，对直接在线性层上加lora带来了困扰
    #并非所有Blocks都需要lora
    for layer_i, layer in enumerate(model.blocks):
        if peft_type=="conv_lora":
            adaptation_layer = LoRA_moe(
                layer.attn.qkv, rank, lora_alpha, conv_lora_expert_num=expert_num)
        elif peft_type=="conv_lora_DSC":
            adaptation_layer = LoRA_moe_DSC(
                layer.attn.qkv, rank, lora_alpha, conv_lora_expert_num=expert_num)
        #给原先qkv包裹进lora，替换原来的attn.qkv
        layer.attn.qkv = adaptation_layer
        
    return model
class LoRA_moe(nn.Module):
    def __init__(self, attn_qkv, rank, lora_alpha, conv_lora_expert_num):
        super().__init__()
        #在init部分，对一个attn.qkv进行插入q和v两种lora
        self.attn_qkv = attn_qkv
        #由于引用关系，attn_qkv依旧冻结梯度
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.conv_lora_expert_num = conv_lora_expert_num
        '''
        在SAM里 qkv输入输出是1->3的
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)'''
        self.dim = attn_qkv.in_features
        self.lora_q = ConvLoRALinear_bypass(self.dim,
            self.dim,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=self.conv_lora_expert_num,
        )
        self.lora_v = ConvLoRALinear_bypass(self.dim,
            self.dim,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=self.conv_lora_expert_num,
        )
    
    def forward(self, x):
        #在forward部分，对一个attn.qkv进行插入q和v两种lora
        qkv = self.attn_qkv(x)
        new_q,moe_loss_q = self.lora_q(x)
        new_v,moe_loss_v = self.lora_v(x)
        #更新qv两个线性层，合并到qkv的结果里
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        moe_loss=   moe_loss_q + moe_loss_v
        #输出要传出moe_loss以保证专家均衡
        return qkv,moe_loss

class LoRA_moe_DSC(nn.Module):
    def __init__(self, attn_qkv, rank, lora_alpha, conv_lora_expert_num):
        super().__init__()
        #在init部分，对一个attn.qkv进行插入q和v两种lora
        self.attn_qkv = attn_qkv
        #由于引用关系，attn_qkv依旧冻结梯度
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.conv_lora_expert_num = conv_lora_expert_num
        '''
        在SAM里 qkv输入输出是1->3的
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)'''
        self.dim = attn_qkv.in_features
        self.lora_q = ConvLoRALinear_bypass_DSC(self.dim,
            self.dim,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=self.conv_lora_expert_num,
        )
        self.lora_v = ConvLoRALinear_bypass_DSC(self.dim,
            self.dim,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=self.conv_lora_expert_num,
        )
    
    def forward(self, x):
        #在forward部分，对一个attn.qkv进行插入q和v两种lora
        qkv = self.attn_qkv(x)
        new_q,moe_loss_q = self.lora_q(x)
        new_v,moe_loss_v = self.lora_v(x)
        #更新qv两个线性层，合并到qkv的结果里
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        moe_loss=   moe_loss_q + moe_loss_v
        #输出要传出moe_loss以保证专家均衡
        return qkv,moe_loss


class LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    对每个Block里的qkv部分进行处理，添加真正的bypass部分"""
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class LoRA(nn.Module):
    '''用于构建LoRA，替换atten中的qkv'''
    def __init__(self,rank=4):
        super().__init__()
        self.rank = rank
    def forward(self,blk: nn.Module):
        self.block = blk
        self.w_As = []  # LoRA产生的线性层，用于后续初始化权重
        self.w_Bs = []
        for t_layer_i, blk in enumerate(self.block):
            # If we only want few lora layer instead of all
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            #为每个block添加LoRA_qkv（替换原先的qkv）
            #列举的blk是Block里的一个MultiScaleBlock，一个Block由多个MSB构成
            #blk.atten里的一个MultiScaleAttention
            #blk.atten.qkv是一个qkv线性层
            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        return self.block
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt#跳连接
        net = self.block(promped)
        return net
    #参考issue添加了权重初始化
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                #print(m)
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)


def InjectModalitySpecificMoE(model:nn.modules, rank=4, lora_alpha=1.0, expert_per_modality=4):
    """
    向模型注入模态特定的MoE层
    
    Parameters
    ----------
    model: nn.modules
        要注入的模型
    rank: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    expert_per_modality: int
        每个模态的专家数量
    
    Returns
    -------
    model: nn.modules
        注入后的模型
    """
    # 为每个block注入ModalitySpecificMoE
    for layer_i, layer in enumerate(model.blocks):

        adaptation_layer = ModalitySpecificMoE(
            layer.attn.qkv, rank, lora_alpha, expert_per_modality=expert_per_modality)
        
        # 替换原始的qkv层
        layer.attn.qkv = adaptation_layer

    return model


def InjectModalitySpecificMoE_v2(model:nn.modules, rank=4, lora_alpha=1.0, expert_per_modality=4):
    """
    向模型注入改进版模态特定的MoE层
    
    Parameters
    ----------
    model: nn.modules
        要注入的模型
    rank: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    expert_per_modality: int
        每个模态的专家数量
    
    Returns
    -------
    model: nn.modules
        注入后的模型
    """
    # 为每个block注入ModalitySpecificMoE_v2
    for layer_i, layer in enumerate(model.blocks):
        adaptation_layer = ModalitySpecificMoE_v2(
            layer.attn.qkv, rank, lora_alpha, expert_per_modality=expert_per_modality)
        
        # 替换原始的qkv层
        layer.attn.qkv = adaptation_layer
    
    return model