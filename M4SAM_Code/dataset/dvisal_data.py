import os, sys
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch

from .dataset_utils import *

'''为了适应这个没有时间信息的网络，只使用已标注图像'''
class SalObjDataset(data.Dataset):
    def __init__(self, data_root, subset, augmentation, interval, sample_rate, trainsize, baseMode):

        with open(os.path.join(data_root, subset + '.txt')) as f:
            lines = f.readlines()
            videolists = sorted([line.strip() for line in lines])

        self.filenames_gt = []
        self.filenames = []
        self.filenames_dep = []

        for video in videolists:
            # Create List for only labeled GT
            label_path = os.path.join(data_root, 'data', video, 'GT')
            filenames_gt_i = [os.path.join(label_path, f) for f in os.listdir(label_path)
                              if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            self.filenames_gt += sorted(filenames_gt_i)

            # Create List for only labeled Image seqs
            image_path = os.path.join(data_root, 'data', video, 'RGB')
            file_postfix = os.listdir(image_path)[-1][-4:]
            filenames_i = [os.path.join(image_path, f[:-4] + file_postfix) for f in os.listdir(label_path)
                           if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            self.filenames += sorted(filenames_i)

            # Create List for only labeled depth seqs
            depth_path = os.path.join(data_root, 'data', video, 'Depth')
            file_dep_postfix = os.listdir(depth_path)[-1][-4:]
            filenames_dep_i = [os.path.join(depth_path, f[:-4] + file_dep_postfix) for f in os.listdir(label_path)
                              if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            self.filenames_dep += sorted(filenames_dep_i)

        self.filenames_gt.sort()
        self.filenames.sort()
        self.filenames_dep.sort()
        self.size = len(self.filenames)


        self.trainsize = trainsize
        self.baseline_mode = baseMode
        self.augment = augmentation
        self.interval = interval
        self.sample_rate = sample_rate


        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])



    def __getitem__(self, index):
        file_path = self.filenames[index]
        file_path_gt = self.filenames_gt[index]
        file_path_dep = self.filenames_dep[index]

        labels = []
        images = []
        depths = []

        # Extracting images
        All_mem_size = self.interval[0] + 1     # i.e., win_size OR stm_queue_size + 1
        for i in reversed(range(-self.interval[1], All_mem_size * self.sample_rate, self.sample_rate)): # i = 3,2,1,0
            # Labels
            if not self.baseline_mode:#默认false
                abs_file_path_gt, new_file_path_gt = self.filename_from_index(file_path_gt, i)
            else:
                abs_file_path_gt, new_file_path_gt = self.filename_from_base(file_path_gt, i)

            if not os.path.exists(abs_file_path_gt):
                label = None
                new_file_path_gt = ""
                if i == self.interval[1]:
                    print(abs_file_path_gt, "does not exist !")
                    exit(1)
            else:
                label = self.binary_loader(abs_file_path_gt)

            labels.append(label)  # [None, None, None, GT]


            # Images
            if not self.baseline_mode:
                abs_file_path, new_file_path = self.filename_from_index(file_path, i)
            else:
                abs_file_path, new_file_path = self.filename_from_base(file_path, i)
            if not os.path.exists(abs_file_path):
                print(abs_file_path, "does not exist !")
                exit(1)
            image = self.rgb_loader(abs_file_path)

            images.append(image)  # [img3, img2, img1, ori_img]


            # Depth Images
            if not self.baseline_mode:
                abs_file_path_dep, new_file_path_dep = self.filename_from_index(file_path_dep, i)
            else:
                abs_file_path_dep, new_file_path_dep = self.filename_from_base(file_path_dep, i)
            if not os.path.exists(abs_file_path_dep):
                print(abs_file_path_dep, "does not exist !")
                exit(1)
            depth = self.binary_loader(abs_file_path_dep)

            depths.append(depth)  # [depth3, depth2, depth1, ori_depth]
        
        seg= '/' if '/' in file_path_gt else '\\'
        file_name = file_path_gt.split(seg)[-1]
        video_name = file_path_gt.split(seg)[-3]

        if self.augment:
            images, labels, depths = cv_random_flip_img(images, labels, depths)
            images, labels, depths = randomCrop_img(images, labels, depths)
            images, labels, depths = randomRotation_img(images, labels, depths)
            images = colorEnhance_img(images)

        for i, img in enumerate(images):
            if labels[i] is not None:
                labels[i] = self.gt_transform(labels[i])
            images[i] = self.img_transform(images[i])
            depths[i] = self.depths_transform(depths[i])

        images = torch.stack(images)
        depths = torch.stack(depths)
        labels = labels[self.interval[0]]

        #没有时间信息：仅有一张Image和一张GT,GT对应的帧是images[interval[0]]
        #return {'image': images[self.interval[0]], 'label': labels, 'depth': depths[self.interval[0]]}
        return images[self.interval[0]], labels, depths[self.interval[0]], [file_name, video_name]


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w,h),Image.BILINEAR),gt.resize((w,h),Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

    def filename_from_index(self, base_file_path, index):
        # e.g., ../DViSal/data/CDTB_bottle_room_occ_1/GT/00000026.png'
        file_name = os.path.basename(base_file_path)    # '00000026.png'
        old_num = int(file_name[-9:-4])                 # '00026' -> int = 26
        new_num = old_num - index                       # 26 - index, e.g., index=2, obtaining 24
        name_elts = "{:05d}".format(new_num)            # '00024'
        new_file_name = file_name[:-9]+name_elts+ file_name[-4:]  # new file_name

        new_path = os.path.join(base_file_path[:-len(new_file_name)], new_file_name)

        if os.path.exists(new_path):
            return new_path, new_file_name
        else:
            return base_file_path, file_name

    def filename_from_base(self, base_file_path, index):
        file_name = os.path.basename(base_file_path)
        return base_file_path, file_name

class SalObjDataset_vid(data.Dataset):
    def __init__(self, data_root, subset, augmentation, interval, sample_rate, trainsize, baseMode, video_length=5, frame_rate=25,drop_last=False,dont_trans_gt=False):

        with open(os.path.join(data_root, subset + '.txt')) as f:
            lines = f.readlines()#从列表txt读出当前采样模式指定的视频集名称们
            videolists = sorted([line.strip() for line in lines])

        #self.filenames_gt = []#整个数据集的gt文件名绝对路径
        self.filenames = []
        #self.filenames_dep = []

        self.video_length = video_length

        self.video_clips_rgb=[]#按照clip切片后的绝对路径
        self.video_clips_dep=[]
        self.video_clips_gt=[]

        for video in videolists:
            #一律按照标注帧来取
            gt_path = os.path.join(data_root, 'data', video, 'GT')#GT文件夹
            #取文件夹内所有标注帧的绝对路径
            fileabs_gt_i = [os.path.join(gt_path, f) for f in os.listdir(gt_path)
                              if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            fileabs_gt_i=sorted(fileabs_gt_i)
            #文件名 无后缀 00000026
            filenames_gt_i = [ f[-12:-4] for f in fileabs_gt_i]
            gt_postfix = os.listdir(gt_path)[-1][-4:]

            #接下来根据gt的name 处理rgb和depth   直接替换后缀和目录！
            rgb_path = os.path.join(data_root, 'data', video, 'RGB')
            rgb_postfix = os.listdir(rgb_path)[-1][-4:]#.jpg单独取出拓展名
            fileabs_rgb_i = [os.path.join(rgb_path,name+rgb_postfix) for name in filenames_gt_i]

            depth_path = os.path.join(data_root, 'data', video, 'Depth')
            dep_postfix = os.listdir(depth_path)[-1][-4:]
            fileabs_dep_i = [os.path.join(depth_path,name+dep_postfix) for name in filenames_gt_i]

            #在每个视频里取clip,保证clip不跨视频
            curr_clip_gt=clipper(fileabs_gt_i,video_length=video_length)
            curr_clip_rgb=clipper(fileabs_rgb_i,video_length=video_length)
            curr_clip_dep=clipper(fileabs_dep_i,video_length=video_length)
            curr_filename=clipper(filenames_gt_i,video_length=video_length)

            self.video_clips_gt+=curr_clip_gt
            self.video_clips_rgb+=curr_clip_rgb
            self.video_clips_dep+=curr_clip_dep

            
            self.filenames += curr_filename

        #==视频文件夹遍历结束==

        #在训练时使用多batch，可以drop掉不满video_length的视频
        self.size = len(self.video_clips_gt)

        self.trainsize = trainsize
        self.baseline_mode = baseMode
        self.augment = augmentation
        self.interval = interval
        self.sample_rate = sample_rate
        self.video_length=video_length


        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])#,
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if dont_trans_gt:
            self.gt_transform = transforms.Compose(
                [transforms.ToTensor()])
        else:
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])



    def __getitem__(self, index):
        #按照video clip，逐个clip取出
        video_clip_rgb=self.video_clips_rgb[index]
        video_clip_gt=self.video_clips_gt[index]
        video_clip_dep=self.video_clips_dep[index]   

        clip_filename=self.filenames[index]#[path.split('/')[-1] for path in video_clip_rgb]

        #从当前clip取出首图用于DVSOD
        '''file_path = self.filenames[index]
        file_path_gt = self.filenames_gt[index]
        file_path_dep = self.filenames_dep[index]'''
        file_path=video_clip_rgb[0]
        file_path_gt=video_clip_gt[0]
        file_path_dep=video_clip_dep[0]


        labels = []
        images = []
        depths = []

        # Extracting images
        # 在batch内部的一个单元里 向前取4帧 每间隔3帧取一帧
        # sample rate 向前取mem时的间隔

        # interval[0] 在这个batch里的 数组中取到GT的索引值 和序列长度有关
        # interval[1] i=在此处取到GT（相位调整，9,6,3,0） 
        All_mem_size = self.interval[0] + 1     # i.e., win_size OR stm_queue_size + 1 3+1
        for i in reversed(range(-self.interval[1], All_mem_size * self.sample_rate, self.sample_rate)):
            # range(-0,4*3,3)
            # i = 9,6,3,0
            # Labels GT
            if not self.baseline_mode:
                abs_file_path_gt, new_file_path_gt = self.filename_from_index(file_path_gt, i)
            else:
                abs_file_path_gt, new_file_path_gt = self.filename_from_base(file_path_gt, i)

            #只有最后一帧是有GT标注的 1 26 31
            if not os.path.exists(abs_file_path_gt):
                label = None
                new_file_path_gt = ""
                if i == self.interval[1]:
                    print(abs_file_path_gt, "does not exist !")
                    exit(1)
            else:
                label = self.binary_loader(abs_file_path_gt)

            labels.append(label)  # [None, None, None, GT]


            # Images
            if not self.baseline_mode:
                abs_file_path, new_file_path = self.filename_from_index(file_path, i)
            else:
                abs_file_path, new_file_path = self.filename_from_base(file_path, i)
            if not os.path.exists(abs_file_path):
                print(abs_file_path, "does not exist !")
                exit(1)
            image = self.rgb_loader(abs_file_path)

            images.append(image)  # [img3, img2, img1, ori_img]


            # Depth Images
            if not self.baseline_mode:
                abs_file_path_dep, new_file_path_dep = self.filename_from_index(file_path_dep, i)
            else:
                abs_file_path_dep, new_file_path_dep = self.filename_from_base(file_path_dep, i)
            if not os.path.exists(abs_file_path_dep):
                print(abs_file_path_dep, "does not exist !")
                exit(1)
            depth = self.binary_loader(abs_file_path_dep)

            depths.append(depth)  # [depth3, depth2, depth1, ori_depth]
        #灵活判断一下win和Linux路径分隔符不一样的问题 注意反斜杠转义符
        seg= '/' if '/' in file_path_gt else '\\'
        file_name = file_path_gt.split(seg)[-1]
        video_name = file_path_gt.split(seg)[-3]
        '''开始视频部分提取'''



        #读取视频clip图像  
        video_rgb=[self.rgb_loader(path) for path in video_clip_rgb]
        video_gt=[self.binary_loader(path) for path in video_clip_gt]
        video_dep=[self.binary_loader(path) for path in video_clip_dep]

        #开始随机增强，先增强再tensor再stack
        if self.augment:
            images, labels, depths, video_rgb,video_gt,video_dep = cv_random_flip(images, labels, depths, video_rgb,video_gt,video_dep)
            images, labels, depths, video_rgb,video_gt,video_dep = randomCrop(images, labels, depths,video_rgb,video_gt,video_dep)
            images, labels, depths, video_rgb,video_gt,video_dep = randomRotation(images, labels, depths,video_rgb,video_gt,video_dep)
            images, video_rgb = colorEnhance(images,video_rgb)




        for i, img in enumerate(images):
            if labels[i] is not None:
                labels[i] = self.gt_transform(labels[i])
            images[i] = self.img_transform(images[i])
            depths[i] = self.depths_transform(depths[i])

        images = torch.stack(images)
        depths = torch.stack(depths)
        labels = labels[self.interval[0]]

        #对videoclip进行变换and格式toTensor转换
        for i in range(len(video_rgb)):
            video_rgb[i]=self.img_transform(video_rgb[i])
            video_rgb[i]=(video_rgb[i]-video_rgb[i].min())/(video_rgb[i].max()-video_rgb[i].min())
            video_gt[i]=self.gt_transform(video_gt[i])
            video_dep[i]=self.depths_transform(video_dep[i])
        
        video_rgb=torch.stack(video_rgb)
        video_gt=torch.stack(video_gt)
        video_dep=torch.stack(video_dep)

        #需要额外判断video的长度是否足长，否则在多batch时dataloader无法stack在一起，引发错误。
        if len(video_rgb) < self.video_length:
            #
            #print("video length is not enough !")
            video_rgb   =   torch.concat([video_rgb,torch.stack([video_rgb[-1]]*(self.video_length-len(video_rgb)))])
            video_gt    =   torch.concat([video_gt,torch.stack([video_gt[-1]]*(self.video_length-len(video_gt)))])
            video_dep   =   torch.concat([video_dep,torch.stack([video_dep[-1]]*(self.video_length-len(video_dep)))])
            #使用最后一帧填充到目标长度
            clip_filename=  clip_filename+[clip_filename[-1]]*(self.video_length-len(clip_filename))
            #print("after",images.shape,video_rgb.shape)
        return images[self.interval[0]], labels, depths[self.interval[0]], [file_name, video_name],\
              video_rgb, video_gt, video_dep,clip_filename


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w,h),Image.BILINEAR),gt.resize((w,h),Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

    def filename_from_index(self, base_file_path, index):
        #从base开始，倒回去取index帧 base是26 index是3 那就取23
        # e.g., ../DViSal/data/CDTB_bottle_room_occ_1/GT/00000026.png'
        file_name = os.path.basename(base_file_path)    # '00000026.png'
        old_num = int(file_name[-9:-4])                 # '00026' -> int = 26
        new_num = old_num - index                       # 26 - index, e.g., index=2, obtaining 24
        name_elts = "{:05d}".format(new_num)            # '00024'
        new_file_name = file_name[:-9]+name_elts+ file_name[-4:]  # new file_name

        new_path = os.path.join(base_file_path[:-len(new_file_name)], new_file_name)

        if os.path.exists(new_path):
            return new_path, new_file_name
        else:
            return base_file_path, file_name

    def filename_from_base(self, base_file_path, index):
        #只取当前帧
        file_name = os.path.basename(base_file_path)
        return base_file_path, file_name


def get_loader(data_root, batchsize, trainsize,
               subset, augmentation, interval, sample_rate,
               shuffle=True, num_workers=12, pin_memory=True, baseMode=False):
    '''针对逐帧的DVISAL数据集进行重载'''
    dataset = SalObjDataset(data_root, subset, augmentation, interval, sample_rate, trainsize, baseMode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_video_loader(data_root, batchsize, trainsize,
               subset, augmentation, interval, sample_rate,video_length,
               shuffle=True, num_workers=6, pin_memory=True, baseMode=False, sampler=None, return_dataset=False, dont_trans_gt=False):
    '''针对带视频信息的DVISAL数据集进行重载'''
    dataset = SalObjDataset_vid(data_root, subset, augmentation, interval, sample_rate, trainsize, baseMode,video_length=video_length,dont_trans_gt=dont_trans_gt)
    
    if return_dataset:
        return dataset
        
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle if sampler is None else False,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  sampler=sampler,
                                  prefetch_factor=2)
                                  #增加了异步预读取
    return data_loader
