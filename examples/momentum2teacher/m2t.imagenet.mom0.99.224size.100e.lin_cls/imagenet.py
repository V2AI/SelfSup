import torch

from torch import nn

from cvpods.layers import ShapeSpec


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Classification(nn.Module):
    def __init__(self, cfg):
        super(Classification, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.network = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        self.freeze()
        self.network.eval()

        # init the fc layer
        self.network.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.network.linear.bias.data.zero_()

        self.norm = nn.BatchNorm1d(1000)

        self.loss_evaluator = nn.CrossEntropyLoss()

        self.to(self.device)

    def freeze(self):
        for name, param in self.network.named_parameters():
            if name not in ['linear.weight', 'linear.bias']:
                param.requires_grad = False

    def forward(self, batched_inputs):
        self.network.eval()
        images = torch.stack([x["image"] for x in batched_inputs]).to(self.device)
        outputs = self.network(images)
        preds = self.norm(outputs["linear"])

        if self.training:
            labels = torch.tensor([gi["category_id"] for gi in batched_inputs]).cuda()
            losses = self.loss_evaluator(preds, labels)
            acc1, acc5 = accuracy(preds, labels, topk=(1, 5))

            return {
                "loss_cls": losses,
                "top1_acc": acc1,
                "top5_acc": acc5,
            }
        else:
            return preds
