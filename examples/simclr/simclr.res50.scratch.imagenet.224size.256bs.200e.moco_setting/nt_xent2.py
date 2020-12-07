import torch
import torch.nn as nn

from cvpods.utils import comm

from torch import distributed as dist


class NT_Xent(nn.Module):
    def __init__(self, device_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.device_size = device_size
        self.temperature = temperature
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        pos_mask_i, pos_mask_j, neg_mask_i, neg_mask_j = \
            self.mask_correlated_samples(comm.get_world_size() * self.device_size, self.device_size)

        self.pos_mask_i = pos_mask_i.to(self.device)
        self.neg_mask_i = neg_mask_i.to(self.device)

        self.pos_mask_j = pos_mask_j.to(self.device)
        self.neg_mask_j = neg_mask_j.to(self.device)

    def mask_correlated_samples(self, batch_size, device_size, _rank=0):
        neg_mask_i = torch.ones((batch_size, batch_size * 2), dtype=bool)
        neg_mask_j = torch.ones((batch_size, batch_size * 2), dtype=bool)

        for rank in range(int(batch_size / device_size)):
            for idx in range(device_size):
                neg_mask_i[device_size * rank + idx, device_size * (2 * rank + 1) + idx] = 0  # i
                neg_mask_j[device_size * rank + idx, device_size * (2 * rank) + idx] = 0  # j

        pos_mask_i = neg_mask_i.clone()
        pos_mask_j = neg_mask_j.clone()

        for rank in range(int(batch_size / device_size)):
            neg_mask_i[
                device_size * rank: device_size * (rank + 1),
                device_size * 2 * rank: device_size * (2 * rank + 1),
            ].fill_diagonal_(0)
            neg_mask_j[
                device_size * rank: device_size * (rank + 1),
                device_size * (2 * rank + 1): device_size * 2 * (rank + 1),
            ].fill_diagonal_(0)

        return ~pos_mask_i, ~pos_mask_j,  neg_mask_i, neg_mask_j

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        local_rank = comm.get_rank()

        if comm.get_world_size() > 1:
            group = comm._get_global_gloo_group()

            zi_large = [torch.zeros_like(z_i) for _ in range(comm.get_world_size())]
            zj_large = [torch.zeros_like(z_j) for _ in range(comm.get_world_size())]

            dist.all_gather(zi_large, z_i, group=group)
            dist.all_gather(zj_large, z_j, group=group)
        else:
            zi_large = [z_i]
            zj_large = [z_j]

        z_large = []
        for idx in range(comm.get_world_size()):
            if idx == local_rank:
                # current device
                z_large.append(z_i)
                z_large.append(z_j)
            else:
                z_large.append(zi_large[idx])
                z_large.append(zj_large[idx])

        zi_large[local_rank] = z_i
        zj_large[local_rank] = z_j

        zi_large = torch.cat(zi_large)
        zj_large = torch.cat(zj_large)

        device_size = z_i.shape[0]
        batch_size = device_size * comm.get_world_size()

        z_large = torch.cat(z_large)

        sim_i_large = self.similarity_f(zi_large.unsqueeze(1), z_large.unsqueeze(0)) / self.temperature
        sim_j_large = self.similarity_f(zj_large.unsqueeze(1), z_large.unsqueeze(0)) / self.temperature

        positive_samples_i = sim_i_large[self.pos_mask_i].reshape(batch_size, 1)
        negative_samples_i = sim_i_large[self.neg_mask_i].reshape(batch_size, -1)

        positive_samples_j = sim_j_large[self.pos_mask_j].reshape(batch_size, 1)
        negative_samples_j = sim_j_large[self.neg_mask_j].reshape(batch_size, -1)

        # partial gradient of l ref to s_ij
        # l_ij / s_ij
        # l_ij_s_ij = (negative_samples_i / (positive_samples_i + negative_samples_i.sum(dim=1, keepdim=True))).sum(dim=1, keepdim=True).mean()
        # print(f"l_ij / s_ij : {l_ij_s_ij}")

        # r = (positive_samples_i.exp() / negative_samples_i.exp().sum(dim=1, keepdim=True)).mean()
        # if local_rank == 0:
        #     print("SimQK to SimQN: ", r)

        labels_i = torch.zeros(batch_size).to(self.device).long()
        logits_i = torch.cat((positive_samples_i, negative_samples_i), dim=1)

        labels_j = torch.zeros(batch_size).to(self.device).long()
        logits_j = torch.cat((positive_samples_j, negative_samples_j), dim=1)

        loss_i = self.criterion(logits_i, labels_i)
        loss_j = self.criterion(logits_j, labels_j)

        loss_i /= device_size
        loss_j /= device_size

        return loss_i, loss_j
