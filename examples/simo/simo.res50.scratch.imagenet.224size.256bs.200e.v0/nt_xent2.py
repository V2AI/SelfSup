import torch
import torch.nn as nn

from cvpods.utils import comm

from torch import distributed as dist


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class NT_Xent(nn.Module):
    def __init__(self, device_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.device_size = device_size
        self.temperature = temperature
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, device_size, rank=0):
        neg_mask_i = torch.ones((device_size, batch_size), dtype=bool)
        # neg_mask_j = torch.ones((device_size, batch_size), dtype=bool)

        for idx in range(device_size):
            neg_mask_i[idx, device_size * rank + idx] = 0  # i
            # neg_mask_j[idx, device_size * rank + idx] = 0  # j

        pos_mask_i = neg_mask_i.clone()
        # pos_mask_j = neg_mask_j.clone()

        # neg_mask_i[:, device_size * rank:device_size * (rank + 1)].fill_diagonal_(0)
        # neg_mask_j[:, device_size * rank:device_size * (rank + 1)].fill_diagonal_(0)

        return ~pos_mask_i, neg_mask_i

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        group = comm._get_global_gloo_group()
        local_rank = comm.get_rank()

        zi_large = [torch.zeros_like(z_i) for _ in range(comm.get_world_size())]
        zj_large = [torch.zeros_like(z_j) for _ in range(comm.get_world_size())]

        dist.all_gather(zi_large, z_i, group=group)
        dist.all_gather(zj_large, z_j, group=group)

        z_large = []
        for idx in range(comm.get_world_size()):
            z_large.append(zj_large[idx])

        device_size = z_i.shape[0]
        batch_size = device_size * comm.get_world_size()

        z_large = torch.cat(z_large)

        print(z_large.shape)
        print(z_i.requires_grad, z_large.requires_grad)

        sim_i_large = self.similarity_f(z_i.unsqueeze(1), z_large.unsqueeze(0)) / self.temperature

        pos_mask_i, neg_mask_i = self.mask_correlated_samples(batch_size, device_size, local_rank)

        pos_mask_i = pos_mask_i.to(self.device)
        neg_mask_i = neg_mask_i.to(self.device)

        positive_samples_i = sim_i_large[pos_mask_i].reshape(device_size, 1)
        negative_samples_i = sim_i_large[neg_mask_i].reshape(device_size, -1)

        print(positive_samples_i.shape, negative_samples_i.shape)

        labels_i = torch.zeros(device_size).to(self.device).long()
        logits_i = torch.cat((positive_samples_i, negative_samples_i), dim=1)

        loss_i = self.criterion(logits_i, labels_i) / device_size
        acc1, acc5 = accuracy(logits_i, labels_i, topk=(1, 5))

        return loss_i, acc1, acc5
