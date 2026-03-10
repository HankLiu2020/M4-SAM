# M⁴-SAM

> **"M⁴-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection"**
> by [*Jiyuan Liu*](mailto:hankliu@hdu.edu.cn), [*Jia Lin*](mailto:lin_j@hdu.edu.cn), [*Xiaofei Zhou*](mailto:zxforchid@outlook.com)\*, *Runmin Cong*, *Deyang Liu*, *Zhi Liu*
> 🎉 **CVPR 2026 Accepted!**

📑 [Paper (arXiv)]() *(to be added)* | 💻 [Code(Github)](https://github.com/HankLiu2020/M4-SAM)

## 🧠 Overview

We propose **M⁴-SAM**, a prompt-free framework that adapts SAM2 for RGB-D video salient object detection by introducing modality-related PEFT, hierarchical feature fusion, and prompt-free memory initialization.

<p align="center">
  <img src="assets/main_figure_v2.svg" width="95%">
</p>

**Key Highlights:**
- 💡 **Modality-Aware MoE-LoRA:** elevates vanilla LoRA with convolutional experts and modality-specific routing for adaptive RGB-D feature fusion and efficient fine-tuning.
- 🧩 **Gated Multi-Level Feature Fusion:** hierarchically aggregates multi-scale encoder features with an adaptive gating mechanism to balance spatial details and semantic context.
- 🚀 **Pseudo-Guided Initialization:** bootstraps the memory bank using a coarse mask as a pseudo prior, enabling zero-shot VSOD without manual prompts.

## ⚡ Start

> **Code is coming soon!** Stay tuned.

### Prepare Dataset

[RDVS](https://github.com/kerenfu/RDVS), [ViDSOD-100](https://github.com/jhl-Det/RGBD_Video_SOD) and [DViSal](https://github.com/DVSOD/DVSOD-DViSal)

### Pretrained Checkpoint

Dependent Models: [SAM2](https://github.com/facebookresearch/sam2) — download [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

## Acknowledgement

Our work would not have been possible without the following open-source projects: 
- [SAM2](https://github.com/facebookresearch/sam2), 
- [XMem](https://github.com/hkchengrex/XMem), 
- [MemSAM](https://github.com/dengxl0520/MemSAM),
- [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet).

Thanks for their great contributions!

## Citation

If you find our work useful, please cite our paper, thank you!

```bibtex
@inproceedings{liu2026m4sam,
  title={M$^4$-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection},
  author={Liu, Jiyuan and Lin, Jia and Zhou, Xiaofei and Cong, Runmin and Liu, Deyang and Liu, Zhi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
