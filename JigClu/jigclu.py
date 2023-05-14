# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.distributed as dist
import diffdist
import random
from .losses import SupCluLoss


class JigClu(nn.Module):
    def __init__(self, base_encoder, dim=128, T=0.07):
        """
        dim: feature dimension (default: 128)
        T: softmax temperature (default: 0.07)
        """
        super(JigClu, self).__init__()

        self.criterion_clu = SupCluLoss(temperature=T)


        self.criterion_loc = nn.CrossEntropyLoss()

        # num_classes is the output fc dimension
        self.encoder = base_encoder(num_classes=dim)

        #dim_mlp = self.encoder.fc_clu.weight.shape[1]
        self.encoder.fc_clu = nn.Sequential(
            nn.Linear(2048, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256), 
            nn.BatchNorm1d(256, affine=False),

            nn.Linear(256, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256))

    @torch.no_grad()
    def _batch_gather_ddp(self, images):
        """
        gather images from different gpus and shuffle between them
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        images_gather = []
        for i in range(2):
            batch_size_this = images[i].shape[0]
            images_gather.append(concat_all_gather(images[i]))
            batch_size_all = images_gather[i].shape[0]
        num_gpus = batch_size_all // batch_size_this


        n,c,h,w = images_gather[0].shape
        #print(n)
        permute = torch.randperm(n*2).cuda()
        torch.distributed.broadcast(permute, src=0)
        images_gather = torch.cat(images_gather, dim=0)
        images_gather = images_gather[permute,:,:,:]
        col1 = torch.cat([images_gather[0:int(0.5*n)], images_gather[int(0.5*n):int(1.0*n)]], dim=3)
        col2 = torch.cat([images_gather[int(1.0*n):int(1.5*n)], images_gather[int(1.5*n):]], dim=3)
        images_gather = torch.cat([col1, col2], dim=2)
 

        bs = images_gather.shape[0] // num_gpus
        gpu_idx = torch.distributed.get_rank()

        return images_gather[bs*gpu_idx:bs*(gpu_idx+1)], permute, n

    def forward(self, images, progress):
        images_gather, permute, bs_all = self._batch_gather_ddp(images)


        # compute features
        q = self.encoder(images_gather)

        q_gather = concat_all_gather(q)
        n,c,h,w = q_gather.shape
        c1,c2 = q_gather.split([1,1],dim=2)
        f1,f2 = c1.split([1,1],dim=3)
        f3,f4 = c2.split([1,1],dim=3)
        q_gather = torch.cat([f1,f2,f3,f4],dim=0)
        q_gather = q_gather.view(n*4,-1)



        # info branch
        # label_info = permute % bs_all  # label
        # idx_info = label_info.sort()[1]  # index
        # clu_info = q_gather[idx_info, :]
        # result_idx1 = []
        # result_idx2 = []
        # result_idx3 = []
        # result_idx4 = []
        # result_idx5 = []
        # for idx in range(0, 1280, 5):
        #     result_idx1.append(idx)
        #     result_idx2.append(idx + 1)
        #     result_idx3.append(idx + 2)
        #     result_idx4.append(idx + 3)
        #     result_idx5.append(idx + 4)
        # pooler = [clu_info[result_idx1, :] , clu_info[result_idx2, :] , clu_info[result_idx3, :] , clu_info[result_idx4, :]]
        # sele_idx1 = random.randint(0,3)
        # sele_idx2 = random.randint(0,3)
        # #sele_idx3 = random.randint(0,3)
        # l1_4 = (pooler[sele_idx1]+pooler[sele_idx2]) / 2
        # #l1_4 = clu_info[result_idx1, :]
        # l5 = clu_info[result_idx5, :]
        # info_gather = torch.cat([l1_4, l5], dim=0)
        # info_gather = info_gather.view(l1_4.shape[0] * 2, -1)  # q_gather.shape:([1024,512])
        # go_clu_info = self.encoder.fc_info(info_gather)
        # go_clu_info = nn.functional.normalize(go_clu_info, dim=1)  # q_clu.shape:([1024,128])
        # label_info = torch.LongTensor(range(l1_4.shape[0])).cuda()
        # go_label_info = torch.cat([label_info, label_info], dim=0)
        # loss_info = self.criterion_info(go_clu_info, go_label_info)



        # clustering branch
        label_clu = permute % bs_all
        q_clu = self.encoder.fc_clu(q_gather)
        q_clu = nn.functional.normalize(q_clu, dim=1)
        loss_clu = self.criterion_clu(q_clu, label_clu)

        # location branch
        label_loc = torch.LongTensor([0]*bs_all+[1]*bs_all).cuda()
        label_loc = label_loc[permute]
        q_loc = self.encoder.fc_loc(q_gather)
        loss_loc = self.criterion_loc(q_loc, label_loc)

        return loss_clu, loss_loc

def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    tensors_gather = diffdist.functional.all_gather(tensors_gather, tensor, next_backprop=None, inplace=True)

    output = torch.cat(tensors_gather, dim=0)
    return output
