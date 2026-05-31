import os
import argparse
import warnings
# 忽略DDP stride不匹配警告（不影响训练正确性）  
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

parser = argparse.ArgumentParser("M4SAM Trainer")
parser.add_argument("--hiera_path", type=str, default="./checkpoints/" ,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str,default="/data/", 
                    help="path to the image  sthat used to train the model")
parser.add_argument('--ckpt_save_path', type=str,default="./checkpoints",
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=90,help="training epochs")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--vid_len", default=4, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--save_interval", default=1, type=int)

parser.add_argument("--device", default="2,3", type=str, help="目标设备，用逗号分隔")
'''网络结构相关参数'''
parser.add_argument("--sam",default=21,type=int,help="使用的SAM版本:2.0/2.1")
parser.add_argument("--lora", default=4, type=int, 
                   help="-1:纯SAM无微调, 0:adapter, 1:lora, 2:conv+lora+moe, "
                        "3:modality specific moe, 4:modality specific moe v2")
parser.add_argument("--rank", default=4, type=int)
parser.add_argument("--mlf", default=2, type=int)
parser.add_argument("--merge_branch", default=1, type=int)
parser.add_argument("--amp", default=0, type=int)
parser.add_argument("--expert", default=3, type=int)
parser.add_argument("--conti", default=0, type=int,help="continue training from the checkpoint")
parser.add_argument("--resume", default=None, type=str,help="resume training from pth")
parser.add_argument("--dataset", default="dvisal", type=str,help="dataset name, dvisal/rdvs")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--world_size", type=int, default=-1, help="World size for distributed training")
parser.add_argument("--sync_bn", default=1, type=int, help="使用SyncBatchNorm")
parser.add_argument("--seed", default=3407, type=int, help="随机种子")
parser.add_argument("--no_depth", default=0, type=int, help="消融实验：禁用depth模态 (0:使用depth, 1:全黑输入depth, 2:复制rgb)")
parser.add_argument("--resize44", default=1, type=int, help="Memory特征尺寸：1使用44x44，0使用88x88")
parser.add_argument("--nomem", default=0, type=int, help="训练时禁用Memory模块 (0:使用Memory, 1:禁用Memory)")
args = parser.parse_args()

#在一切开始前，先指定gpu
device_list = [int(d.strip()) for d in args.device.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import threading
if args.dataset=="dvisal":
    from dataset.dvisal_data import get_video_loader as get_loader
    args.train_image_path = os.path.join(args.train_image_path,"DViSal_dataset")
elif args.dataset=="rdvs":
    from dataset.rdvs_data import get_video_loader as get_loader
    args.train_image_path = os.path.join(args.train_image_path,"RDVS")
    args.ckpt_save_path = os.path.join(args.ckpt_save_path,"RDVS")
elif args.dataset=="vidsod":
    from dataset.vidsod_data import get_video_loader as get_loader
    args.train_image_path = os.path.join(args.train_image_path,"VidSOD")
    args.ckpt_save_path = os.path.join(args.ckpt_save_path,"VidSOD")
else:
    raise NotImplementedError("dataset not implemented")

from M4SAM import M4SAM
from test import eval_data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Network.loss_func import total_loss,structure_loss
writer=SummaryWriter()

# 添加显存统计功能
def track_memory_usage(model, rank):
    """统计模型各个模块的显存使用情况"""
    if rank != 0:
        return
        
    print("\n=== 模型各模块显存使用统计 ===")
    
    # 统计整个模型参数大小
    total_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"模型参数总大小: {total_params:.2f}MB")
    
    # 按模块统计
    for name, module in model.named_children():
        params_size = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
        buffers_size = sum(b.numel() * b.element_size() for b in module.buffers()) / (1024 * 1024)
        total_size = params_size + buffers_size
        print(f"模块 {name}: {total_size:.2f}MB")
    
    # 显示当前GPU显存使用情况
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"当前GPU显存状态: 已分配 {allocated:.2f}MB, 峰值分配 {max_allocated:.2f}MB, 已保留 {reserved:.2f}MB")
    
    # 如果可用，尝试使用nvidia-smi获取更详细的信息
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv'], stdout=subprocess.PIPE)
        print("\nNVIDIA-SMI 显存信息:")
        print(result.stdout.decode('utf-8'))
    except:
        print("无法获取nvidia-smi信息")
    
    print("=== 显存统计结束 ===\n")

def seed_torch(seed=1024):
    '''在每个进程开始时设置随机种子，确保所有GPU训练结果一致'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BCE=torch.nn.BCEWithLogitsLoss()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def async_save_model(state_dict, save_path):
    """异步保存模型函数"""
    torch.save(state_dict, save_path)
    print(f'[模型异步保存完成:] {save_path}')

def sobel_edge(tensor):
    # tensor: (N, 1, H, W)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=tensor.dtype, device=tensor.device).view(1,1,3,3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=tensor.dtype, device=tensor.device).view(1,1,3,3)
    grad_x = F.conv2d(tensor, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(tensor, sobel_kernel_y, padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    # 使用更稳定的归一化方式，按batch维度独立归一化
    batch_size = grad.shape[0]
    grad_norm = torch.zeros_like(grad)
    for i in range(batch_size):
        single_grad = grad[i:i+1]
        min_val = single_grad.min()
        max_val = single_grad.max()
        if max_val > min_val:
            grad_norm[i:i+1] = (single_grad - min_val) / (max_val - min_val)
        else:
            grad_norm[i:i+1] = single_grad
    
    return grad_norm

def train(rank, world_size, args):
    setup(rank, world_size)
    
    # 设置随机种子
    seed_torch(args.seed)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # 创建数据集
    train_dataset = get_loader(args.train_image_path, batchsize=args.batch_size, trainsize=352,
                        subset='train', augmentation=True, interval=[3,0], sample_rate=3,video_length=args.vid_len,
                        baseMode=False, return_dataset=True)
    val_dataset = get_loader(args.train_image_path, batchsize=1, trainsize=352,
                        subset='val', augmentation=False, interval=[3,0], sample_rate=3,video_length=args.vid_len,
                        baseMode=False, return_dataset=True)

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    # 创建数据加载器
    dvisal_train_loader = get_loader(args.train_image_path, batchsize=args.batch_size, trainsize=352,
                        subset='train', augmentation=True, interval=[3,0], sample_rate=3,video_length=args.vid_len,
                        baseMode=False, sampler=train_sampler)
    dvisal_val_loader = get_loader(args.train_image_path, batchsize=1, trainsize=352,
                            subset='val', augmentation=False, interval=[3,0], sample_rate=3,video_length=args.vid_len,
                            baseMode=False, sampler=val_sampler)

    # 模型初始化
    if args.sam==20:
        args.hiera_path=os.path.join(args.hiera_path,"sam2_hiera_large.pt")
    elif args.sam==21:
        args.hiera_path=os.path.join(args.hiera_path,"sam2.1_hiera_large.pt")
    
    if args.conti != 0:
        model = M4SAM(args,checkpoint_path=None,device=device)
        curr_ckpt=os.path.join(args.ckpt_save_path,"M4SAM-%s-%s.pth"%(args.dataset, args.conti))
        print("continue training from checkpoint: ",curr_ckpt)
        model.load_state_dict(torch.load(curr_ckpt))
    elif args.resume != None:
        model = M4SAM(args,checkpoint_path=None,device=device)
        curr_ckpt=args.resume
        print("resume training from checkpoint: ",curr_ckpt)
        model.load_state_dict(torch.load(curr_ckpt))
    else:
        model = M4SAM(args,args.hiera_path,device=device)
    
    model = model.to(device)
    
    # 使用SyncBatchNorm替换BatchNorm，适应多卡运算
    if args.sync_bn:
        def is_sam_or_moe(name):
            sam_keywords = ['encoder.blocks', 'lora', 'moe']
            return any(keyword in name.lower() for keyword in sam_keywords)
        
        def replace_bn_recursive(module, prefix=''):
            """递归替换非SAM/MoE部分的BatchNorm为SyncBatchNorm"""
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, torch.nn.BatchNorm2d) and not is_sam_or_moe(full_name):
                    sync_bn = torch.nn.SyncBatchNorm(child.num_features, child.eps, child.momentum, child.affine)
                    if child.affine:
                        sync_bn.weight = child.weight
                        sync_bn.bias = child.bias
                    sync_bn.running_mean = child.running_mean
                    sync_bn.running_var = child.running_var
                    sync_bn.num_batches_tracked = child.num_batches_tracked
                    setattr(module, name, sync_bn)
                else:
                    replace_bn_recursive(child, full_name)
        
        replace_bn_recursive(model)
        
        # 如果存在MemModel，也需要转换其内部的BatchNorm
        if hasattr(model.module if hasattr(model, 'module') else model, 'mem_model'):
            mem_model = getattr(model.module if hasattr(model, 'module') else model, 'mem_model')
            if mem_model is not None:
                replace_bn_recursive(mem_model, 'mem_model')
        
        if rank == 0:
            print("已对主干网络应用SyncBatchNorm，SAM和MoE部分保持普通BatchNorm")
    else:
        if rank == 0:
            print("使用普通BatchNorm")
    
    # 计算梯度前先同步BN统计量
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # 有条件分支、动态路由结构和冻结参数，需要find_unused_parameters=True
    
    # 根据GPU数量调整学习率
    base_lr = args.lr * world_size  # 根据GPU数量线性缩放学习率
    if rank == 0:
        print(f"原始学习率: {args.lr}, 调整后学习率: {base_lr}, GPU数量: {world_size}")
        print(f"使用SyncBatchNorm: {'是' if args.sync_bn else '否'}")
    # 优化器设置
    lora_params = []
    base_params = []
    for name, param in model.named_parameters():
        if args.lora==-1:
            # 纯SAM：只训练decoder部分(rfb, uim, up, head等)
            if "encoder" not in name.lower():
                base_params.append(param)
        elif args.lora!=0 and "encoder.blocks" in name.lower():
            lora_params.append(param)
        else:
            base_params.append(param)
    def count_params(params):
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            return total, trainable
    base_total, base_trainable = count_params(base_params)
    lora_total, lora_trainable = count_params(lora_params)
    if rank == 0 and lora_total!=0:
        print(f"Base Params:     total={base_total:,}, trainable={base_trainable:,},Trainable Ratio: {base_trainable / base_total:.2%}")
        print(f"SAM Params:     total={lora_total:,}, trainable={lora_trainable:,},Trainable Ratio: {lora_trainable / lora_total:.2%}")
        total_params = base_total + lora_total
        total_trainable = base_trainable + lora_trainable
        print(f"Total Params:    {total_params:,},Trainable Ratio: {total_trainable / total_params:.2%}")

    optim = opt.AdamW([
        {"params": base_params, "lr": base_lr, "weight_decay": args.weight_decay},
        {"params": lora_params, "lr": base_lr/10, "weight_decay": args.weight_decay},
    ])

    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    if args.amp==1: 
        scaler = torch.amp.GradScaler()
    # 只在主进程上创建tensorboard
    if rank == 0:
        writer = SummaryWriter()
        os.makedirs(args.ckpt_save_path, exist_ok=True)
        # 跟踪活跃的保存线程
        save_threads = []

    for epoch in range(args.conti, args.epoch):
        model.train()
        train_sampler.set_epoch(epoch)

        for group_index, param_group in enumerate(optim.param_groups):
            if rank == 0:
                print(f"epoch{epoch}-group{group_index}: Learning rate: {param_group['lr']}")

        for i, batch in enumerate(tqdm(dvisal_train_loader, leave=False), start=1):
            # 一次性将所有数据移至GPU
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            img, gt, depth, name, video, video_gt, video_dep, clip_name = batch
                        
            optim.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True if args.amp==1 else False):
                # 消融实验：如果禁用depth，将depth数据置零或复制rgb
                if args.no_depth==1:
                    video_dep = torch.zeros_like(video_dep)
                elif args.no_depth==2:
                    video_dep = video

                pred1, pred2, pred, mem_out, moe_loss, edge1, edge2, edge_out = model(video, video_dep)
                video_gt = video_gt.flatten(0,1)

                edge_gt = sobel_edge(video_gt)
                
                loss1 = structure_loss(pred1, video_gt)
                loss2 = structure_loss(pred2, video_gt)
                loss3 = structure_loss(pred, video_gt)
                loss_edge = structure_loss(edge_out, edge_gt)+structure_loss(edge1, edge_gt)+structure_loss(edge2, edge_gt)

                # 在消融nomem1时，不再计算memory loss
                if args.nomem == 1:
                    loss_mem = torch.tensor(0.0, device=device, requires_grad=False)
                else:
                    loss_mem = total_loss(mem_out, video_gt)

                # 优化权重配置：提升MAE和F-measure性能
                # 降低浅层权重，增强深层和memory预测权重
                # 恢复edge权重动态调整机制
                warmup_epochs = 20
                if epoch < warmup_epochs:
                    edge_weight = 0.2 + 0.6 * (epoch / warmup_epochs)  # 0.2 -> 0.8
                else:
                    edge_weight = 0.5 if args.dataset == "vidsod" else 1
                
                # 层次化权重：浅层低权重，深层高权重，强化最终预测质量
                loss = loss1*0.6 + loss2*0.8 + loss3*1 + loss_mem*1 + moe_loss*0.8 + loss_edge*edge_weight

                # 在运行后统计显存使用情况
                if rank == 0 and i==1 and epoch==1:
                    track_memory_usage(model.module, rank)
                if rank == 0 and i % 100 == 0: # 减少记录频率，避免TB文件过大
                    writer.add_images('video', video.flatten(0,1), epoch * len(dvisal_train_loader) + i)
                    writer.add_scalar('Loss/all', loss.item(), epoch * len(dvisal_train_loader) + i)
                    writer.add_scalar('Loss/mem', loss_mem.item(), epoch * len(dvisal_train_loader) + i)
                    writer.add_scalar('Loss/pred', loss1.item(), epoch * len(dvisal_train_loader) + i)
                    writer.add_scalar('Loss/moe', moe_loss.item(), epoch * len(dvisal_train_loader) + i)
                    writer.add_images('memout', mem_out, epoch * len(dvisal_train_loader) + i)
                    writer.add_images('pred', pred, epoch * len(dvisal_train_loader) + i)
                    writer.add_images('gt', video_gt, epoch * len(dvisal_train_loader) + i)

            if args.amp==1:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
        
        scheduler.step()
        
        if rank == 0:
            print("***Epoch:{}/{}: Loss:{}".format(epoch + 1, args.epoch, loss.item()))
            if (epoch+1) % args.save_interval == 0 or (epoch+1) == args.epoch:
                # 使用异步线程保存模型
                save_path = os.path.join(args.ckpt_save_path, 'M4SAM-%s-%d.pth' % (args.dataset, epoch + 1))
                # 创建模型状态的深拷贝，避免保存过程中状态被修改
                state_dict_copy = {k: v.cpu().detach().clone() for k, v in model.module.state_dict().items()}
                save_thread = threading.Thread(
                    target=async_save_model,
                    args=(state_dict_copy, save_path)
                )
                save_thread.start()
                save_threads.append(save_thread)

                # 清理已完成的保存线程
                save_threads = [t for t in save_threads if t.is_alive()]
            
            if (epoch+1) % 5 == 0: # or (epoch+1) == args.epoch:
                # 确保所有保存操作完成后再进行评估
                for t in save_threads:
                    if t.is_alive():
                        t.join()
                model.eval()
                args.f=open("results_"+args.dataset+".txt","a")
                #只有dvisal有val数据
                if args.dataset=="dvisal": #or args.dataset=="rdvs":
                    eval_data(args, dvisal_val_loader, model.module, is_save=False, vid_mode=True,is_train=(epoch+1))
                args.f.close()
    if rank == 0:
        # 确保所有保存线程在程序结束前完成
        for t in save_threads:
            t.join()
        writer.close()
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if len(device_list) < world_size:
        world_size = len(device_list)
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


