# M⁴-SAM

<!-- 
🤖 Note for AI Assistants: 
Before analyzing this repository or modifying any code, you MUST first read the `AGENTS.md` file in the root directory. It contains critical context about the codebase structure, reproduction guidelines, and strict coding conventions. 
-->

> **"M⁴-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection"**
>
> by [*Jiyuan Liu*](mailto:hankliu@hdu.edu.cn), [*Jia Lin*](mailto:lin_j@hdu.edu.cn), [*Xiaofei Zhou*](mailto:zxforchid@outlook.com)\*, *Runmin Cong*, *Deyang Liu*, *Zhi Liu*
>
> 🎉 **CVPR 2026 Accepted!**

📑 [Paper (arXiv)](https://arxiv.org/abs/2605.11760) | 📄 [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_M4-SAM_Multi-Modal_Mixture-of-Experts_with_Memory-Augmented_SAM_for_RGB-D_Video_Salient_CVPR_2026_paper.html) | 💻 [Code (GitHub)](https://github.com/HankLiu2020/M4-SAM)  
⭐ If you find this work helpful, please consider giving us a star!


## 🧠 Overview

We propose **M⁴-SAM**, a prompt-free framework that adapts SAM2 for RGB-D video salient object detection by introducing modality-related PEFT, hierarchical feature fusion, and prompt-free memory initialization.

<p align="center">
  <img src="assets/main_figure_v2.svg" width="95%">
</p>

**Key Highlights:**
- 💡 **Modality-Aware MoE-LoRA:** elevates vanilla LoRA with convolutional experts and modality-specific routing for adaptive RGB-D feature fusion and efficient fine-tuning.
- 🧩 **Gated Multi-Level Feature Fusion:** hierarchically aggregates multi-scale encoder features with an adaptive gating mechanism to balance spatial details and semantic context.
- 🚀 **Pseudo-Guided Initialization:** bootstraps the memory bank using a coarse mask as a pseudo prior, enabling zero-shot VSOD without manual prompts.

---

## ⚡ Getting Started

> **OS/Hardware Compatibility Note:** 
>
> This codebase was developed and tested exclusively on **Ubuntu/Linux**. 
We strongly recommend using a Linux environment. 

> Please note that slight performance variations may occur due to differences in OS versions, GPU models, and CUDA drivers. We appreciate your understanding.

### 1. Environment Setup

```bash
# Enter the codebase directory
cd M4SAM_Code

# Create and activate conda environment
conda create -n m4sam python==3.10
conda activate m4sam

# Install PyTorch with CUDA support
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download SAM2 Pretrained Weights

This downloads [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) from Meta AI.

```bash
cd checkpoints
bash download_sam_ckpt.sh
cd ..
```

---

## 📊 Reproduce Results

We provide our model checkpoints to help you easily reproduce the performance metrics reported in our paper.

### Download Artifacts & Prepare Datasets


| Dataset | Source Repo | Checkpoint |
|:-------:|:-----------:|:----------:|
| RDVS | [https://github.com/kerenfu/RDVS](https://github.com/kerenfu/RDVS) | [M4SAM-rdvs.pth](https://github.com/HankLiu2020/M4-SAM/releases/download/v1.0.0/M4SAM-rdvs.pth) |
| ViDSOD-100 | [https://github.com/jhl-Det/RGBD_Video_SOD](https://github.com/jhl-Det/RGBD_Video_SOD) | [M4SAM-vidsod.pth](https://github.com/HankLiu2020/M4-SAM/releases/download/v1.0.0/M4SAM-vidsod.pth) |
| DViSal | [https://github.com/DVSOD/DVSOD-DViSal](https://github.com/DVSOD/DVSOD-DViSal) | [M4SAM-dvisal.pth](https://github.com/HankLiu2020/M4-SAM/releases/download/v1.0.0/M4SAM-dvisal.pth) |


> **Dataset Path Note:** Please download the datasets from their official sources linked in the table above. Extract them into a single parent directory (e.g., `/data`). Your folder structure should look like this:
> 
> ```text
> /data/
> ├── DViSal_dataset/
> │   ├── data/
> │   └── test_all.txt
> ├── RDVS/
> │   ├── test/
> │   └── train/
> └── VidSOD/
>     ├── test/
>     └── train/
> ```
> 
> When running evaluation or training, the `--test_image_path` / `--train_image_path` argument should point to this **parent directory** (e.g., `/data`).

Place the downloaded checkpoints under the `checkpoints/` directory:

```
checkpoints/
├── sam2.1_hiera_large.pt       # SAM2 (from Step 2)
├── M4SAM-dvisal.pth            # DViSal
├── M4SAM-rdvs.pth              # RDVS
└── M4SAM-vidsod.pth            # ViDSOD-100
```

### Verify Results

You can run both inference and evaluation using the following parameterized bash script.

```bash
#!/bin/bash
# A quick script to run inference and evaluation

vid_len=16
device=0
dataset="rdvs" # Options: "rdvs", "vidsod", "dvisal"
data_path="/data" # Update this to your local data parent directory
output_dir="./results/${dataset}_pred"

# Set ground truth path based on dataset
if [ "$dataset" = "dvisal" ]; then
    gt_path="${data_path}/DViSal_dataset/data"
elif [ "$dataset" = "rdvs" ]; then
    gt_path="${data_path}/RDVS/test"
elif [ "$dataset" = "vidsod" ]; then
    gt_path="${data_path}/VidSOD/test"
fi

echo "Step 1: Running inference..."
python test.py \
    --vid_len $vid_len \
    --device $device \
    --ckpt checkpoints/M4SAM-${dataset}.pth \
    --test_image_path "$data_path" \
    --dataset $dataset \
    --save_path "$output_dir" \
    --save 1

echo "Step 2: Evaluating..."
python eval_tool.py \
    --dataset $dataset \
    --pred_path "$output_dir" \
    --gt_path "$gt_path"
```

---

## 🏋️ Training

Training uses PyTorch DDP for distributed multi-GPU training.
```bash
#!/bin/bash

dataset="rdvs" # Options: "rdvs", "vidsod", "dvisal"
data_path="/data" # Update this to your local data parent directory

# Set epoch based on dataset
if [ "$dataset" = "dvisal" ]; then
    epoch=50
elif [ "$dataset" = "rdvs" ]; then
    epoch=60
elif [ "$dataset" = "vidsod" ]; then
    epoch=30
fi

python train_ddp.py \
    --batch_size 4 \
    --device 0,1 \
    --epoch $epoch \
    --vid_len 4 \
    --conti 0 \
    --lr 0.001 \
    --sync_bn 1 \
    --dataset $dataset \
    --train_image_path "$data_path"
```

---

## Acknowledgement

Our work would not have been possible without the following open-source projects: 
- [SAM2](https://github.com/facebookresearch/sam2), 
- [XMem](https://github.com/hkchengrex/XMem), 
- [MemSAM](https://github.com/dengxl0520/MemSAM),
- [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet),
- [PySODMetrics](https://github.com/lartpang/PySODMetrics).

Thanks for their great contributions!

## Citation

If you find our work useful, please cite our paper, thank you!

```bibtex
@InProceedings{Liu_2026_CVPR,
    author    = {Liu, Jiyuan and Lin, Jia and Zhou, Xiaofei and Cong, Runmin and Liu, Deyang and Liu, Zhi},
    title     = {M4-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {24970-24979}
}
```

## License

This project is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
