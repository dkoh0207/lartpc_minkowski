import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.backbone.acnn import ACNN
from collections import defaultdict

class ACNN_Chain(nn.Module):

    def __init__(self, cfg, name='acnn'):
        super(ACNN_Chain, self).__init__()
        self.model_cfg = cfg['modules'][name]
        self.net = ACNN(cfg, name='acnn')
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.segmentation = ME.MinkowskiLinear(self.net.outputFeatures,
                                               self.num_classes)
        print('Total Number of Trainable Parameters = {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        out = defaultdict(list)
        num_gpus = len(input)
        for igpu, x in enumerate(input):
            res = self.net(x)
            res = self.segmentation(res)
            out['segmentation'].append(res.F)
        return out

class SegmentationLoss(nn.Module):

    def __init__(self, cfg, name='segmentation_loss'):
        super(SegmentationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, label, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        segmentation = outputs['segmentation']
        assert len(segmentation) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        batch_ids = [d[:, -2] for d in label]
        total_loss = 0
        total_acc = 0
        count = 0
        # Loop over GPUS
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = label[i][:, -1][batch_index]
                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.cross_entropy(event_segmentation, event_label)
                if weight is not None:
                    event_weight = weight[i][batch_index]
                    event_weight = torch.squeeze(event_weight, dim=-1).float()
                    total_loss += torch.mean(loss_seg * event_weight)
                else:
                    total_loss += torch.mean(loss_seg)
                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() / float(predicted_labels.nelement())
                total_acc += acc
                count += 1

        return {
            'accuracy': total_acc/count,
            'loss': total_loss/count
        }
