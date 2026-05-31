import argparse

parser = argparse.ArgumentParser("M4SAM Tester")
parser.add_argument("--ckpt", type=str, default=None,
                help="path to the checkpoint of m4sam")
parser.add_argument("--test_image_path", type=str, default="/data/", 
                    help="path to the image files for testing")
parser.add_argument("--save_path", type=str, default="./results/pred",
                    help="path to save the predicted masks")
parser.add_argument("--vid_len", default=4, type=int)
parser.add_argument("--e", default=59, type=str,help="当前测试的ckpt轮次数")

parser.add_argument("--device", default=2, type=int,help="目标设备")
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
parser.add_argument("--nomem", default=0, type=int,help="bypass xmem memory ")
parser.add_argument("--dataset", default="rdvs", type=str,help="dataset name, dvisal/rdvs/vidsod")
parser.add_argument("--save", default=0, type=int,help="save the results")
parser.add_argument("--batchsize", default=1, type=int, help="测试时的batch size")
parser.add_argument("--no_depth", default=0, type=int, help="消融实验：禁用depth模态 (0:使用depth, 1:消融depth)")
parser.add_argument("--resize44", default=1, type=int, help="Memory特征尺寸：1使用44x44，0使用88x88")

import os,datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from M4SAM import M4SAM

from tqdm import tqdm

import py_sod_metrics
import cv2


WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()

sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
FM = py_sod_metrics.FmeasureV2(
    metric_handlers={
        "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
        "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
        "rec": py_sod_metrics.RecallHandler(**sample_gray),
        "fpr": py_sod_metrics.FPRHandler(**sample_gray),
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
        "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
        "ber": py_sod_metrics.BERHandler(**sample_gray),
        "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
        "kappa": py_sod_metrics.KappaHandler(**sample_gray),
        "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
        "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
        "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
        "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
        "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
        "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
        "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
        "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
        "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
        "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
        "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
        "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
        "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
        "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
        "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
        "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
        "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
        "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
        "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
        "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
        "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
        "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
    }
)








def eval_data(args,test_loader=None,model:M4SAM=None,is_save=True,vid_mode=True,is_train=0):
    if not is_train:#在测试阶段为False，在eval阶段为True
        model.eval()
        model.cuda()
        if is_save: os.makedirs(args.save_path, exist_ok=True)
    if test_loader==None:
        test_loader = get_loader(args.test_image_path, batchsize=args.batchsize, trainsize=352,
                            subset='test_all', augmentation=False, interval=[3,0], sample_rate=3,video_length=args.vid_len,
                            baseMode=False,dont_trans_gt=True)
                            #GT输出原始大小！

    for i, pack in enumerate(tqdm(test_loader), start=1):
        with torch.no_grad():
            if vid_mode:
                img, gt, depth, name_pack, video, video_gt, video_dep,clip_name  = pack
                
                # 消融实验：如果禁用depth，将depth数据置零或复制rgb
                if args.no_depth==1:
                    video_dep = torch.zeros_like(video_dep)
                elif args.no_depth==2:
                    video_dep = video

                
                _, _, res_unet_vid,mem_out,*extra = model(video.cuda(),video_dep.cuda())
                if hasattr(args, 'nomem'):#先判断存在：在train的eval里是不存在nomem的
                    if args.nomem==1:
                        #不使用XMem,需要对sam和unet的结果进行sigmoid才能正常显示
                        res_vid = res_unet_vid
                    else:#正常输出
                        res_vid = mem_out
                else:#正常输出
                    res_vid = mem_out
                res_vid = torch.sigmoid(res_vid)
                res_vid=res_vid.view(-1,args.vid_len,1,352,352)
                #video b t c h w
                B, T = video.shape[0], video.shape[1]
                for b in range(B):
                    # 用于跟踪当前clip中已处理的帧名，避免处理padding的重复帧
                    processed_frames_in_clip = set()
                    for t in range(T):
                        # 检查是否为padding的重复帧
                        frame_name = clip_name[t][b]
                        if frame_name in processed_frames_in_clip:
                            # 跳过padding的重复帧
                            continue
                        processed_frames_in_clip.add(frame_name)
                        
                        image, gt, depth = video[b:b+1, t, :, :, :], video_gt[b:b+1, t, :, :, :], video_dep[b:b+1, t, :, :, :]
                        res = res_vid[b, t, :, :, :]
                        gt = gt.squeeze()
                        name = name_pack[1][b] + "_" + clip_name[t][b]

                        #确保测试时尺寸在数据集尺寸
                        if res.shape != gt.shape:
                            #res = cv2.resize(res, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                            res = F.interpolate(res.unsqueeze(0), size=(gt.shape[0], gt.shape[1]), mode='bilinear', align_corners=False).squeeze()

                        gt = np.asarray(gt*255, np.uint8)
                        
                        # fix: duplicate sigmoid
                        # res = torch.sigmoid(res)
                        #res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                        res = res.data.cpu()
                        res = res.numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        
                        '''计算测评指标'''
                        FM.step(pred=res, gt=gt)
                        #WFM.step(pred=res, gt=gt)
                        res = (res * 255).astype(np.uint8)
                        SM.step(pred=res, gt=gt)
                        EM.step(pred=res, gt=gt)
                        MAE.step(pred=res, gt=gt)
                        
                        if is_save:#不保存以加快测试速度
                            #使用plt会存在一些图片的问题:图片有毛边，效果也不好
                            '''os.makedirs(os.path.join(args.save_path, name_pack[1][0]), exist_ok=True)
                            plt.imsave(os.path.join(args.save_path, name_pack[1][0],clip_name[t][0] + ".png"), res,cmap='gray')'''

                            os.makedirs(os.path.join(args.save_path, name_pack[1][0]), exist_ok=True)
                            cv2.imwrite(os.path.join(args.save_path, name_pack[1][0],clip_name[t][0] + ".png"), res)                

            else:#no vid mode
                image, gt, depth,name_pack = pack
                gt=gt.squeeze()
                name=name_pack[1][0] + "_" + name_pack[0][0] 

                gt = np.asarray(gt*255, np.uint8)
                image,depth = image.cuda(),depth.cuda()
                
                # 消融实验：如果禁用depth，将depth数据置零
                if args.no_depth:
                    depth = torch.zeros_like(depth)
                
                res, _, _ = model(image,depth)
                # fix: duplicate sigmoid
                # res = torch.sigmoid(res)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu()
                res = res.numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)
                #print("Saving " + name)
                
                '''计算测评指标'''
                FM.step(pred=res, gt=gt)
                #WFM.step(pred=res, gt=gt)
                SM.step(pred=res, gt=gt)
                EM.step(pred=res, gt=gt)
                MAE.step(pred=res, gt=gt)
                if is_save:
                    os.makedirs(os.path.join(args.save_path, name_pack[1][0]), exist_ok=True)
                    plt.imsave(os.path.join(args.save_path, name_pack[1][0],name_pack[0][0] + ".png"), res,cmap='gray')
    fm = FM.get_results()["fm"]["adaptive"]
    #wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]["curve"].mean()
    mae = MAE.get_results()["mae"]

    time_now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")#type:str
    print(f"{time_now} EM: {em:.4f},SM: {sm:.4f},F_beta: {fm:.4f},MAE: {mae:.4f}")

    if not is_train:#测试阶段保存结构
        wirite_buff=time_now +" "+args.dataset+" Epoch:"+str(args.e)+" nomem:"+str(args.nomem)+"\n"
    else:
        wirite_buff=time_now +" "+args.dataset+" Epoch:"+str(is_train)+"\n"
    args.f.write(f"{wirite_buff} EM: {em:.4f},SM: {sm:.4f},F_beta: {fm:.4f},MAE: {mae:.4f}\n")
        #单次写入结果，以防错开

if __name__ == '__main__':
    #仅在直接调用时解析参数！
    args = parser.parse_args()
    
    #在文件中记录测试结果，以便异步测试提升GPU效率
    args.f=open("results_"+args.dataset+".txt","a")

    #from dataset.dvisal_data import get_video_loader as get_loader
    if args.dataset=="dvisal":
        from dataset.dvisal_data import get_video_loader as get_loader
        args.test_image_path = os.path.join(args.test_image_path,"DViSal_dataset")
    elif args.dataset=="rdvs":
        from dataset.rdvs_data import get_video_loader as get_loader
        args.test_image_path = os.path.join(args.test_image_path,"RDVS")   
    elif args.dataset=="vidsod":
        from dataset.vidsod_data import get_video_loader as get_loader
        args.test_image_path = os.path.join(args.test_image_path,"VidSOD")
    else:
        raise NotImplementedError("dataset not implemented")

    '''当前使用的权重'''
    if args.ckpt is not None:
        ckpt = args.ckpt
    else:
        if args.dataset=="rdvs":
            ckpt = "./checkpoints/RDVS/M4SAM-rdvs-"+str(args.e)+".pth"
        elif args.dataset=="vidsod":
            ckpt = "./checkpoints/VidSOD/M4SAM-vidsod-"+str(args.e)+".pth"
        else:
            ckpt = "./checkpoints/M4SAM-dvisal-"+str(args.e)+".pth"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print("Using Checkpoint"+ckpt)
    if args.nomem==1:#不使用XMem
        print("Bypass XMem")
    with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True if args.amp==1 else False):
        model = M4SAM(args,checkpoint_path=None,device="cuda")
        model.load_state_dict(torch.load(ckpt), strict=True)
        #eval_data(dvisal_val_loader,model)
        eval_data(args,None,model,vid_mode=True,is_save=args.save,is_train=False)

    args.f.close()