# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def identity(x):
    return x


class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r=8,
        lora_alpha=8,
        lora_dropout: float = 0.0,
        fan_in_fan_out=False,
        merge_weights=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge_weights: False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights
        )  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.
    原先的线性层权重是冻住的，bypass掉，只更新LoRA的权重
    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        **kwargs,
    ):#init部分负责重构并插入lora层
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # MoE-Conv
            topk = 1
            
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            # 创建一个从1到专家数量的上采样比例列表，计划使用每个专家使用不同的上采样比例来处理特征图
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        #计算冻结权重的结果
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)

            # Distribute data samples to experts.
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert.
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            result += (lora_res @ self.lora_B.T) * self.scaling * 0.1#scale,防止出现nan

        return result, moe_loss

class ConvLoRALinear_bypass(nn.Linear, LoRALayer):
    """
    应对sam中qkv打包的写法，只在qv上使用lora
    本bypass类只做lora的支路，不对layer进行注入
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        **kwargs,
    ):#init部分负责重构并插入lora层
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # MoE-Conv
            topk = 1
            
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
                
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        #计算冻结权重的结果
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)

            # Distribute data samples to experts.
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert.
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            result = (lora_res @ self.lora_B.T) * self.scaling #* 0.1#scale,防止出现nan

        return result, moe_loss

class ConvLoRALinear_bypass_DSC(nn.Linear, LoRALayer):
    """
    ConvLoRA的bypass版本，使用深度可分离卷积(Depthwise Separable Convolution)替代标准卷积。
    这个类只实现LoRA的支路，不对原始层进行注入。
    相比标准卷积，深度可分离卷积可以显著减少参数量和计算量，同时保持类似的特征提取能力。
    Parameters
    ----------
    in_features: int
        输入特征维度
    out_features: int
        输出特征维度
    r: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    lora_dropout: float
        Dropout概率
    fan_in_fan_out: bool
        是否使用fan_in_fan_out模式
    merge_weights: bool
        是否在推理时合并权重
    conv_lora_expert_num: Optional[int]
        MoE中的专家数量
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # MoE with different convolution types
            topk = 2
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            
            # 创建第一组专家
            # 专家1：3x3等宽卷积
            expert1 = nn.Sequential(
                nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(r),
                nn.GELU()
            )
            self.lora_moe_experts.append(expert1)
            
            # 专家2：5x5等宽卷积
            expert2 = nn.Sequential(
                nn.Conv2d(r, r, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(r),
                nn.GELU()
            )
            self.lora_moe_experts.append(expert2)
            
            # 专家3：深度可分离卷积
            expert3 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, groups=r, bias=False),
                nn.BatchNorm2d(r),
                nn.GELU(),
                # Pointwise Convolution
                nn.Conv2d(r, r, kernel_size=1, stride=1, padding=0, bias=True),
                nn.GELU()
            )
            expert3[-2].bias.data.zero_()
            self.lora_moe_experts.append(expert3)
            
            # 创建第二组专家（结构相同）
            # 专家4：3x3等宽卷积
            expert4 = nn.Sequential(
                nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(r),
                nn.GELU()
            )
            self.lora_moe_experts.append(expert4)
            
            # 专家5：5x5等宽卷积
            expert5 = nn.Sequential(
                nn.Conv2d(r, r, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(r),
                nn.GELU()
            )
            self.lora_moe_experts.append(expert5)
            
            # 专家6：深度可分离卷积
            expert6 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, groups=r, bias=False),
                nn.BatchNorm2d(r),
                nn.GELU(),
                # Pointwise Convolution
                nn.Conv2d(r, r, kernel_size=1, stride=1, padding=0, bias=True),
                nn.GELU()
            )
            expert6[-2].bias.data.zero_()
            self.lora_moe_experts.append(expert6)
                
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)

            # Distribute data samples to experts
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                cur_res = expert_inputs[i]
                cur_res = self.lora_moe_experts[i](cur_res)
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            result = (lora_res @ self.lora_B.T) * self.scaling

            return result, moe_loss
        return 0, 0  # 当r=0时返回零贡献

class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        #self.w_noise是可学习参数，一开始是全0的，依靠反传进行更新。如果moe_loss没打通，就会导致一直全0
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        batch_size = feats.shape[0]
        #将输入特征图通过全局平均池化压缩成向量表示。
        feats_S = self.gap(feats).view(batch_size, -1)
        #特征向量与门控权重相乘，得到每个专家的原始得分。
        clean_logits = feats_S @ self.w_gate
        if self.noisy_gating and self.training:
            #添加高斯噪声，促进训练时的探索，避免模型过早收敛到固定的专家选择模式
            raw_noise_stddev = feats_S @ self.w_noise
            #由于训练中出现inf，避免/0问题，所以做限制            
            #raw_noise_stddev = torch.clamp(raw_noise_stddev, min=0.001, max=1)

            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        #选择得分最高的K个专家，对选中专家的得分进行softmax归一化，生成稀疏的门控值矩阵

        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        #鼓励专家之间的负载均衡，变异系数平方（cv_squared）
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #避免噪音导致inf
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / (noise_stddev))
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / (noise_stddev))
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [bs * num_of chosen experts, dim]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class ModalitySpecificMoE(nn.Module):
    """
    模态特定的混合专家模型，将专家池分为三组：RGB专家、深度专家和融合专家
    每组专家数量相等，总共3n个专家
    
    Parameters
    ----------
    attn_qkv: nn.Module
        原始的qkv线性层
    rank: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    expert_per_modality: int
        每个模态的专家数量，总专家数量为3*expert_per_modality
    """
    def __init__(self, attn_qkv, rank, lora_alpha, expert_per_modality=4):
        super().__init__()
        self.attn_qkv = attn_qkv
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.expert_per_modality = expert_per_modality
        self.total_experts = expert_per_modality * 3  # RGB, 深度, 融合
        self.dim = attn_qkv.in_features
        
        # 创建两个LoRA层，一个用于q，一个用于v
        self.lora_q = self._create_modality_specific_lora()
        self.lora_v = self._create_modality_specific_lora()
        
        # 模态类型标识，用于在前向传播中区分不同模态
        self.modality_type = None  # 'rgb', 'depth', 'fusion'
    
    def _create_modality_specific_lora(self):
        """创建模态特定的LoRA层，包含三组专家"""
        return ModalityLoRA(
            self.dim,
            self.dim,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            expert_per_modality=self.expert_per_modality
        )
    
    def set_modality_type(self, modality_type):
        """设置当前处理的模态类型"""
        assert modality_type in ['rgb', 'depth'], "模态类型必须是'rgb'或'depth'"
        self.modality_type = modality_type
        self.lora_q.set_modality_type(modality_type)
        self.lora_v.set_modality_type(modality_type)
    
    def forward(self, x):
        """
        前向传播，根据模态类型选择对应的专家组
        
        Parameters
        ----------
        x: torch.Tensor
            输入特征
        """
        # 如果传入了模态类型，则更新当前模态类型
        
        if self.modality_type is None:
            raise ValueError("必须在forward前通过set_modality_type设置模态类型或传入modality_type参数")
        
        qkv = self.attn_qkv(x)
        new_q, moe_loss_q = self.lora_q(x)
        new_v, moe_loss_v = self.lora_v(x)
        
        # 更新qv两个线性层，合并到qkv的结果里
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        moe_loss = moe_loss_q + moe_loss_v
        
        return qkv, moe_loss


class ModalityLoRA(nn.Linear):
    """
    模态特定的LoRA实现，包含三组专家：RGB专家、深度专家和融合专家
    
    Parameters
    ----------
    in_features: int
        输入特征维度
    out_features: int
        输出特征维度
    r: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    merge_weights: bool
        是否在推理时合并权重
    expert_per_modality: int
        每个模态的专家数量
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        expert_per_modality: int = 4,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化LoRA层的基础参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout >= 0.0 else None
        self.merge_weights = merge_weights
        
        self.fan_in_fan_out = fan_in_fan_out
        self.expert_per_modality = expert_per_modality
        self.total_experts = expert_per_modality * 3  # RGB, 深度, 融合
        self.modality_type = None  # 'rgb', 'depth', 'fusion'
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            
            # 为每个模态创建专家组
            self.lora_moe_experts = nn.ModuleList([])
            
            # 创建RGB专家组
            for i in range(expert_per_modality):
                if i % 3 == 0:
                    # 3x3等宽卷积
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU()
                    )
                elif i % 3 == 1:
                    # 5x5等宽卷积
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=5, stride=1, padding=2, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU()
                    )
                else:
                    # 深度可分离卷积
                    expert = nn.Sequential(
                        # Depthwise Convolution
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, groups=r, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU(),
                        # Pointwise Convolution
                        nn.Conv2d(r, r, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.GELU()
                    )
                    expert[-2].bias.data.zero_()
                self.lora_moe_experts.append(expert)
            
            # 创建深度专家组
            for i in range(expert_per_modality):
                if i % 3 == 0:
                    # 3x3等宽卷积，但使用深度特定的初始化
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU()
                    )
                elif i % 3 == 1:
                    # 5x5等宽卷积，适合深度边缘特征
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=5, stride=1, padding=2, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU()
                    )
                else:
                    # 深度可分离卷积，适合深度图的全局结构
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, groups=r, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU(),
                        nn.Conv2d(r, r, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.GELU()
                    )
                    expert[-2].bias.data.zero_()
                self.lora_moe_experts.append(expert)
            
            # 创建融合专家组
            for i in range(expert_per_modality):
                if i % 3 == 0:
                    # 使用1x1卷积进行特征融合
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU(),
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU()
                    )
                elif i % 3 == 1:
                    # 使用大核卷积+点卷积
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=5, stride=1, padding=2, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU(),
                        nn.Conv2d(r, r, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.GELU()
                    )
                    expert[-2].bias.data.zero_()
                else:
                    # 使用残差连接进行特征融合
                    expert = nn.Sequential(
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, groups=r, bias=False),
                        nn.BatchNorm2d(r),
                        nn.GELU(),
                        nn.Conv2d(r, r, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.GELU()
                    )
                    expert[-2].bias.data.zero_()
                self.lora_moe_experts.append(expert)
            
            # 创建模态特定的门控网络
            self.topk = self.expert_per_modality+1  # 每次选择的专家数量
            self.lora_moe_gating = MoEGate(M=self.total_experts, d=self.r, K=self.topk)
            self.multiply_by_gates = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def set_modality_type(self, modality_type):
        """设置当前处理的模态类型"""
        assert modality_type in ['rgb', 'depth'], "模态类型必须是'rgb'或'depth'"
        self.modality_type = modality_type
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def T(self, w):
        return w.T if self.fan_in_fan_out else w
    
    def forward(self, x: torch.Tensor):
        if self.modality_type is None:
            raise ValueError("必须在forward前通过set_modality_type设置模态类型或传入modality_type参数")
        
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]
            
            # 将特征转换为卷积格式
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            
            # 根据模态类型选择专家范围
            # 对于RGB输入：使用RGB专家和融合专家
            # 对于深度输入：使用深度专家和融合专家
            fusion_expert_start = 2 * self.expert_per_modality
            fusion_expert_end = 3 * self.expert_per_modality
            
            if self.modality_type == 'rgb':
                # 使用RGB专家和融合专家
                modality_expert_start = 0
                modality_expert_end = self.expert_per_modality
            elif self.modality_type == 'depth':
                # 使用深度专家和融合专家
                modality_expert_start = self.expert_per_modality
                modality_expert_end = 2 * self.expert_per_modality
            else:  # fusion
                # 只使用融合专家
                modality_expert_start = fusion_expert_start
                modality_expert_end = fusion_expert_end
            
            # 计算门控值
            gates, moe_loss = self.lora_moe_gating(lora_res)
            
            # 创建一个掩码，保留当前模态的专家和融合专家
            modality_mask = torch.zeros_like(gates)
            # 添加当前模态的专家
            modality_mask[:, modality_expert_start:modality_expert_end] = 1.0
            # 总是添加融合专家
            modality_mask[:, fusion_expert_start:fusion_expert_end] = 1.0
            
            # 应用掩码并重新归一化门控值
            gates = gates * modality_mask
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
            
            # 分发数据样本到专家
            dispatcher = SparseDispatcher(self.total_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            
            for i in range(self.total_experts):
                # 只处理有输入的专家，并且是当前模态的专家或融合专家
                if len(expert_inputs[i]) == 0 or modality_mask[0, i].item() == 0:
                    continue
                cur_res = expert_inputs[i]
                cur_res = self.lora_moe_experts[i](cur_res)
                expert_outputs.append(cur_res)
            
            # 合并专家处理后的数据样本
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res
            
            # 转换回原始格式
            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            
            result = (lora_res @ self.lora_B.T) * self.scaling
            
            return result, moe_loss
        
        return 0, 0  # 当r=0时返回零贡献


class ModalityLoRA_v2(nn.Linear):
    """
    改进版模态特定的LoRA实现，先根据模态选择专家组，再在组内进行门控分配
    
    Parameters
    ----------
    in_features: int
        输入特征维度
    out_features: int
        输出特征维度
    r: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    merge_weights: bool
        是否在推理时合并权重
    expert_per_modality: int
        每个模态的专家数量
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        expert_per_modality: int = 4,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化LoRA层的基础参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout >= 0.0 else None
        self.merge_weights = merge_weights
        
        self.fan_in_fan_out = fan_in_fan_out
        self.expert_per_modality = expert_per_modality
        self.modality_type = None  # 'rgb', 'depth'
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            
            # 创建各模态专家组
            self.rgb_experts = nn.ModuleList([])
            self.depth_experts = nn.ModuleList([])
            self.fusion_experts = nn.ModuleList([])
            
            # 私有函数，为每个模态组创建专家
            self._create_rgb_experts()
            self._create_depth_experts()
            self._create_fusion_experts()
            
            # 在初始化时为每个模态组创建独立的门控网络
            self.topk = 2  # 每个组内选择的专家数量 #2
            self.rgb_gate = MoEGate(M=expert_per_modality, d=self.r, K=self.topk)
            self.depth_gate = MoEGate(M=expert_per_modality, d=self.r, K=self.topk)
            self.fusion_gate = MoEGate(M=expert_per_modality, d=self.r, K=self.topk)
            
            self.multiply_by_gates = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def _create_rgb_experts(self):
        """创建RGB模态的专家组"""
        for i in range(self.expert_per_modality):
            if i % 3 == 0:
                # 3x3等宽卷积
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU()
                )
            elif i % 3 == 1:
                # 5x5等宽卷积
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU()
                )
            else:
                # 深度可分离卷积
                expert = nn.Sequential(
                    # Depthwise Convolution
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, groups=self.r, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU(),
                    # Pointwise Convolution
                    nn.Conv2d(self.r, self.r, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GELU()
                )
                expert[-2].bias.data.zero_()
            self.rgb_experts.append(expert)
    
    def _create_depth_experts(self):
        """创建深度模态的专家组"""
        for i in range(self.expert_per_modality):
            if i % 3 == 0:
                # 3x3等宽卷积，但使用深度特定的初始化
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU()
                )
            elif i % 3 == 1:
                # 5x5等宽卷积，适合深度边缘特征
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU()
                )
            else:
                # 深度可分离卷积，适合深度图的全局结构
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, groups=self.r, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU(),
                    nn.Conv2d(self.r, self.r, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GELU()
                )
                expert[-2].bias.data.zero_()
            self.depth_experts.append(expert)
    
    def _create_fusion_experts(self):
        """创建融合专家组"""
        for i in range(self.expert_per_modality):
            if i % 3 == 0:
                # 使用1x1卷积进行特征融合
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU(),
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU()
                )
            elif i % 3 == 1:
                # 大核卷积+点卷积
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU(),
                    nn.Conv2d(self.r, self.r, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GELU()
                )
                expert[-2].bias.data.zero_()
            else:
                # 使用残差连接进行特征融合
                expert = nn.Sequential(
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, groups=self.r, bias=False),
                    nn.BatchNorm2d(self.r),
                    nn.GELU(),
                    nn.Conv2d(self.r, self.r, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.GELU()
                )
                expert[-2].bias.data.zero_()
            self.fusion_experts.append(expert)
    
    def set_modality_type(self, modality_type):
        """设置当前处理的模态类型"""
        assert modality_type in ['rgb', 'depth'], "模态类型必须是'rgb'或'depth'"
        self.modality_type = modality_type
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def T(self, w):
        return w.T if self.fan_in_fan_out else w
    
    def _process_expert_group(self, lora_res, experts, gate):
        """处理一组专家的输入和输出
        
        Parameters
        ----------
        lora_res: torch.Tensor
            输入特征
        experts: nn.ModuleList
            专家列表
        gate: MoEGate
            门控网络
            
        Returns
        -------
        output: torch.Tensor
            专家组处理后的输出
        moe_loss: float
            门控损失
        """
        # 计算门控值
        gates, moe_loss = gate(lora_res)
        
        # 分发数据样本到专家
        dispatcher = SparseDispatcher(len(experts), gates)
        expert_inputs = dispatcher.dispatch(lora_res)
        expert_outputs = []
        
        for i, expert in enumerate(experts):
            if len(expert_inputs[i]) == 0:
                continue
            cur_res = expert_inputs[i]
            cur_res = expert(cur_res)
            expert_outputs.append(cur_res)
        
        # 合并专家处理后的数据样本
        output = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
        
        return output, moe_loss
    
    def forward(self, x: torch.Tensor):
        if self.modality_type is None:
            raise ValueError("必须在forward前通过set_modality_type设置模态类型")
        
        if self.r > 0:
            # LoRA降维
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            
            # 转换为卷积格式
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]
            
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            
            # 根据模态类型选择专家组
            total_output = lora_res.clone()  # 用于累加专家组输出
            total_moe_loss = 0.0
            
            # 处理当前模态的专家组
            if self.modality_type == 'rgb':
                rgb_output, rgb_loss = self._process_expert_group(lora_res, self.rgb_experts, self.rgb_gate)
                total_output = total_output + rgb_output
                total_moe_loss += rgb_loss
            else:  # depth
                depth_output, depth_loss = self._process_expert_group(lora_res, self.depth_experts, self.depth_gate)
                total_output = total_output + depth_output
                total_moe_loss += depth_loss
            
            # 融合专家组总是参与处理
            fusion_output, fusion_loss = self._process_expert_group(lora_res, self.fusion_experts, self.fusion_gate)
            total_output = total_output + fusion_output
            total_moe_loss += fusion_loss
            
            # 转换回原始格式
            lora_res = total_output.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            
            # 升维
            result = (lora_res @ self.lora_B.T) * self.scaling
            
            return result, total_moe_loss
        
        return 0, 0  # 当r=0时返回零贡献


class ModalitySpecificMoE_v2(nn.Module):
    """
    改进版模态特定的混合专家模型，先根据模态选择专家组，再在组内进行门控分配
    
    Parameters
    ----------
    attn_qkv: nn.Module
        原始的qkv线性层
    rank: int
        LoRA的秩
    lora_alpha: int
        LoRA的缩放因子
    expert_per_modality: int
        每个模态的专家数量
    """
    def __init__(self, attn_qkv, rank, lora_alpha, expert_per_modality=4):
        super().__init__()
        self.attn_qkv = attn_qkv
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.expert_per_modality = expert_per_modality
        self.dim = attn_qkv.in_features
        
        # 创建两个LoRA层，一个用于q，一个用于v
        self.lora_q = self._create_modality_specific_lora()
        self.lora_v = self._create_modality_specific_lora()
        
        # 模态类型标识，用于在前向传播中区分不同模态
        self.modality_type = None  # 'rgb', 'depth'
    
    def _create_modality_specific_lora(self):
        """创建模态特定的LoRA层，包含三组专家"""
        return ModalityLoRA_v2(
            self.dim,
            self.dim,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            merge_weights=False,
            expert_per_modality=self.expert_per_modality
        )
    
    def set_modality_type(self, modality_type):
        """设置当前处理的模态类型"""
        assert modality_type in ['rgb', 'depth'], "模态类型必须是'rgb'或'depth'"
        self.modality_type = modality_type
        self.lora_q.set_modality_type(modality_type)
        self.lora_v.set_modality_type(modality_type)
    
    def forward(self, x):
        """
        前向传播，根据模态类型选择对应的专家组
        
        Parameters
        ----------
        x: torch.Tensor
            输入特征
        """
        if self.modality_type is None:
            raise ValueError("必须在forward前通过set_modality_type设置模态类型")
        
        qkv = self.attn_qkv(x)
        new_q, moe_loss_q = self.lora_q(x)
        new_v, moe_loss_v = self.lora_v(x)
        
        # 更新qv两个线性层，合并到qkv的结果里
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        moe_loss = moe_loss_q + moe_loss_v
        
        return qkv, moe_loss

