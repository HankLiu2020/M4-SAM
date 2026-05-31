
class STCNModel(nn.modules):
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        '''self.STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)'''
        self.STCN = STCN(self.single_object).cuda()

        self.train()
        '''self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()'''

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb']#希望改为：输入特征图,bxtx64x88x88
        Ms = data['gt']#希望改为：输出特征图,bxtx64x88x88,不打算进行decode

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            # Fs的结构是[batch_size, frame_num, c, h, w],包含当前所有帧，所以仅需一次
            k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', Fs)

            if self.single_object:
                ref_v = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v)
                #此处使用Encode Value，输入了frame1的输出特征prev mask
                prev_v = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask)

                values = torch.cat([ref_v, prev_v], 2)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values)

                out['mask_1'] = prev_mask
                out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits
            else:
                sec_Ms = data['sec_gt']
                selector = data['selector']

                ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
                ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])
                ref_v = torch.stack([ref_v1, ref_v2], 1)

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v, selector)
                
                prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
                prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1)
                values = torch.cat([ref_v, prev_v], 3)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values, selector)

                out['mask_1'] = prev_mask[:,0:1]
                out['mask_2'] = this_mask[:,0:1]
                out['sec_mask_1'] = prev_mask[:,1:2]
                out['sec_mask_2'] = this_mask[:,1:2]

                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

class XMemModel(nn.modules):
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank

        '''modified content'''
        self.mask_decoder=
        '''self.XMem = nn.parallel.DistributedDataParallel(
            XMem(config).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        '''
        self.XMem = XMem(config).cuda()
        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size', str(sum([param.nelement() for param in self.XMem.parameters()])))
        '''self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)'''

        self.train()
        '''self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.XMem.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()'''

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

    def forward_mem(self, data, it=0):#from XMem
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        # 将数据中的所有非列表、非字典、非整数的元素移动到GPU上
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        # 使用自动混合精度
        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # image features never change, compute once
            # 初始化构建sensory memory,所有帧仅需一次
            key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)

            filler_one = torch.zeros(1, dtype=torch.int64)
            # 初始化memory
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            #对第一帧进行value编码
            v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
            values = v16.unsqueeze(3) # add the time dimension

            # 遍历每一帧
            for ti in range(1, self.num_frames):
                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would 
                    # need broadcasting in gather which we don't have
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None

                # Segment frame ti
                memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
                        hidden, selector, h_out=(ti < (self.num_frames-1)))

                # No need to encode the last frame
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
            '''
            # 计算损失
            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)
                        '''

        '''# Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            self.optimizer.step()

        self.scheduler.step()'''

    def forward_with_memory(#modified from memSAM
        self,
        imgs: torch.Tensor, # [b,t,c,h,w] 
        imge,#直接给embeded Feature当做img输入，不再构建新的Encoder
        GT_alter
    ) -> torch.Tensor:
        b, t, c, h, w = imgs.shape  # b t c h w
        # encode imgs to imgs embedding
        #保留上游SAM Encoder来的embed feature
        key, shrinkage, selection, imge = self.memory('encode_key_from_embed', imge)
        # init memory
        hidden = torch.zeros((b, 1, self.memory.hidden_dim, *key.shape[-2:])).to(imge.device)
        #decode首帧mask，以供后续使用 后续给decoder部分打包一下方便修改
        #这里我直接给decoder从memsam原生的改吧
        mask, _ = self.mask_decoder(
                image_embedding=imge[:,0],down_f
                ) # b c h w
        mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
        # frames_pred.append(mask)
        values_0, hidden = self.memory('encode_value', imgs[:,0], imge[:,0], hidden, mask)
        values = values_0[:,:,:,:0]

        # process frames
        for ti in range(0, t):
            if ti == 0 :
                ref_keys = key[:,:,[0]] 
                ref_shrinkage = shrinkage[:,:,[0]]
                ref_values = values_0 
            else:
                ref_keys = key[:,:,:ti]
                ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                ref_values = values

            # get single frame
            frame = imge[:,ti]
            # read memory
            memory_readout = self.memory(
                'read_memory',
                key[:, :, ti],
                selection[:, :, ti] if selection is not None else None,
                ref_keys, ref_shrinkage, ref_values)
            
            # # cross attention with depth
            # memory_readout = rearrange(memory_readout, "b t c h w -> (b t) c (h w)").transpose(1, 2)
            # memory_readout, _ = self.ca_mem(memory_readout, depth_feature, depth_feature)
            
            # memory_readout = rearrange(memory_readout.transpose(1, 2), "(b t) c (h w) -> b t c h w", b=1, h=32)
            # # print(memory_readout.shape)

            # generate memory embedding#这篇文章track的是memory embedding，尝试直接track Feature
            hidden, me = self.memory('decode', frame, hidden, memory_readout)
            # # featmap
            # from mmengine.visualization import Visualizer
            # visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
            #                         save_dir='temp_dir')
            # drawn_img = visualizer.draw_featmap(featmap=me[0,0]*-1,
            #                         overlaid_image=imgs[0,ti].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8),
            #                         channel_reduction='squeeze_mean',
            #                         alpha=0.3)
            # if self.memory.reinforce :
            #     visualizer.add_image(f'featmap_reinforce', drawn_img, step=ti)
            # else:
            #     visualizer.add_image(f'featmap_noreinforce', drawn_img, step=ti)

            '''mask, _ = self.mask_decoder( 
                        image_embeddings=frame,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=None,
                        dense_prompt_embeddings=me[:,0], # remove object num dim
                        multimask_output=False,
                    ) # b c h w'''
            mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
            frames_pred.append(mask)

            # last frame no encode
            if ti < t-1:
                # update memory
                is_deep_update = np.random.rand() < 0.2
                # v16, hidden = self.memory('encode_value', imgs[:,ti], me[:,0], hidden, mask, is_deep_update=is_deep_update)
                v16, hidden = self.memory('encode_value', imgs[:,ti], imge[:,ti], hidden, mask, is_deep_update=is_deep_update)
                values = torch.cat([values, v16], 3)

        pred = torch.stack(frames_pred, dim=1) # b t c h w

        return pred
    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.XMem.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.XMem.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.XMem.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.XMem.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.XMem.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.XMem.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.XMem.eval()
        return self

