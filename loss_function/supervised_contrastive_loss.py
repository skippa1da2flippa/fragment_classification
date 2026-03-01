"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, input, target=None, mask=None):
      device = input.device

      if len(input.shape) < 3:
          raise ValueError('`features` needs to be [bsz, n_views, ...]')
      if len(input.shape) > 3:
          input = input.view(input.shape[0], input.shape[1], -1)

      batch_size = input.shape[0]
      contrast_count = input.shape[1]
      contrast_feature = torch.cat(torch.unbind(input, dim=1), dim=0)

      if target is not None and mask is not None:
          raise ValueError('Cannot define both `labels` and `mask`')
      elif target is not None:
          target = target.contiguous().view(-1, 1)
          mask = torch.eq(target, target.T).float().to(device)
      elif mask is None:
          mask = torch.eye(batch_size, dtype=torch.float32).to(device)
      else:
          mask = mask.float().to(device)

      if self.contrast_mode == 'one':
          anchor_feature = input[:, 0]
          anchor_count = 1
      elif self.contrast_mode == 'all':
          anchor_feature = contrast_feature
          anchor_count = contrast_count
      else:
          raise ValueError(f'Unknown contrast_mode: {self.contrast_mode}')

      anchor_dot_contrast = torch.div(
          torch.matmul(anchor_feature, contrast_feature.T),
          self.temperature
      )

      # Stability fix
      logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
      logits = anchor_dot_contrast - logits_max.detach()

      mask = mask.repeat(anchor_count, contrast_count)
      logits_mask = torch.ones_like(mask).scatter_(
          1,
          torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
          0
      )
      mask = mask * logits_mask

      exp_logits = torch.exp(logits) * logits_mask
      log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

      mask_pos_pairs = mask.sum(1)
      valid = mask_pos_pairs > 0

      if valid.sum() == 0:
          return torch.tensor(0.0, device=device, requires_grad=True)

      mean_log_prob_pos = (mask * log_prob)[valid].sum(1) / mask_pos_pairs[valid]
      loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
      return loss.mean()
