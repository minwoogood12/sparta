# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from skimage import color
import os
import matplotlib.pyplot as plt

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def unfold_w_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    return unfolded_x

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def compute_pairwise_term_neighbor(mask_logits, mask_logits_neighbor, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob_neigh = F.logsigmoid(mask_logits_neighbor)
    log_bg_prob_neigh = F.logsigmoid(-mask_logits_neighbor)

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)
    
    log_fg_prob_unfold = unfold_w_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_w_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob_neigh[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob_neigh[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    return -log_same_prob[:, 0]

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()
def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self._warmup_iters = 600
        self.register_buffer("_iter", torch.zeros([1]))
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores


    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor,
                images_lab_sim,
                images_lab_sim_nei,
                images_lab_sim_nei1,
                images_lab_sim_nei2,
                images_lab_sim_nei3,
                images_lab_sim_nei4,
                images_lab_sim_nei5,
                images_lab_sim_nei6,
                images_lab_sim_nei7
               ):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        mfvis_masks,mfvis_gt = self.convert_predict(outs_batch, targets_batch)

            
        loss_proj, loss_pairwise, loss_tk = self.loss_masks_proj(mfvis_masks, mfvis_gt,
                                                       images_lab_sim,
                                                       images_lab_sim_nei,
                                                       images_lab_sim_nei1,
                                                       images_lab_sim_nei2,
                                                       images_lab_sim_nei3,
                                                       images_lab_sim_nei4,
                                                       images_lab_sim_nei5,
                                                       images_lab_sim_nei6,
                                                       images_lab_sim_nei7)
      
        

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        losses["loss_tk"] += loss_tk
        losses["loss_proj"] += loss_proj
        losses["loss_pairwise"] += loss_pairwise
        loss_sum = loss_tk + loss_proj + loss_pairwise

        losses[CORE_LOSS_KEY] += (
            self.weight_dict["loss_tk"] * loss_tk +
            self.weight_dict["loss_proj"] * loss_proj +
            self.weight_dict["loss_pairwise"] * loss_pairwise
        )

        return losses

    def convert_predict(self, outs_batch, gt_masks ):
      #mask
      T, N, H, W = gt_masks.shape
      gt_masks_all = gt_masks.permute(1, 0, 2, 3).contiguous()
      
      pred_masks_per_frame = []
      for idx, frame_out in enumerate(outs_batch):
        # Step-wise outputs: List[(N, M, H, W)], choose the last step
        multistep_masks = frame_out["multistep_pred_multimasks_high_res"]
        last_step_masks = multistep_masks[-1]  # Tensor of shape (N, M, H, W)

        gt_mask_frame = gt_masks_all[:,idx:idx+1].float()
        gt_mask_frame = gt_mask_frame.expand_as(last_step_masks)

        
        if last_step_masks.shape[1] == 1:
            # Choose the last mask candidate (M-1) for each object
            final_masks = last_step_masks[:, -1]  # Shape: (N, H, W)
            pred_masks_per_frame.append(final_masks)
        else :
            num_objects = last_step_masks.shape[1]

            loss_multimask = sigmoid_focal_loss(
            last_step_masks,
            gt_mask_frame,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
            )
            loss_multidice = dice_loss(
                last_step_masks, gt_mask_frame, num_objects, loss_on_multimask=True
            )
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            final_masks = last_step_masks[batch_inds, best_loss_inds] 
               
            pred_masks_per_frame.append(final_masks)

      pred_masks_all = torch.stack(pred_masks_per_frame, dim=0)
      pred_masks_all = pred_masks_all.permute(1, 0, 2, 3)

      return pred_masks_all, gt_masks_all

 

    def topk_mask(self, images_lab_sim, k):
        images_lab_sim_mask = torch.zeros_like(images_lab_sim)
        topk, indices = torch.topk(images_lab_sim, k, dim =1) # 1, 3, 5, 7
        #n배치 개수만큼 반복해서 모든 배치에서 top-k를 선정해서 나머지는 0으로 만듬
        images_lab_sim_mask = images_lab_sim_mask.scatter(1, indices, topk)
        return images_lab_sim_mask

    def loss_masks_proj(self, outputs, targets,
                    images_lab_sim,
                    images_lab_sim_nei,
                    images_lab_sim_nei1,
                    images_lab_sim_nei2,
                    images_lab_sim_nei3,
                    images_lab_sim_nei4,
                   images_lab_sim_nei5,
                   images_lab_sim_nei6,
                   images_lab_sim_nei7):
        
        self._iter += 1




        src_masks = outputs
        target_masks = targets.float()
        '''
        import os
        from torchvision.utils import save_image

        save_dir = "./masks_vis"
        os.makedirs(save_dir, exist_ok=True)

        N, T, H, W = outputs.shape

        for n in range(N):
            for t in range(T):
                # === outputs (logits → sigmoid) ===
                out_mask = torch.sigmoid(outputs[n, t]).unsqueeze(0)  # [1,H,W]
                out_path = os.path.join(save_dir, f"output_n{n}_t{t:03d}.png")
                save_image(out_mask, out_path)

                # === targets (binary) ===
                tgt_mask = targets[n, t].unsqueeze(0).float()  # [1,H,W]
                tgt_path = os.path.join(save_dir, f"target_n{n}_t{t:03d}.png")
                save_image(tgt_mask, tgt_path)

        '''

        N, T, H, W = src_masks.shape
        src_masks_4d = src_masks.reshape(N * T, 1, H, W)
      
    
        src_masks_down = F.interpolate(
            src_masks_4d.float(),
            (images_lab_sim[0].shape[-2], images_lab_sim[0].shape[-1]),
            mode="bilinear",
        )
     
        # 결과 복원 [N, T, H', W']
        H_, W_ = src_masks_down.shape[-2:]
        src_masks = src_masks_down.view(N, T, H_, W_)
       

        images_lab_sim = torch.cat(images_lab_sim, dim =0)
        #8 8 256 256

        images_lab_sim_nei = torch.cat(images_lab_sim_nei, dim=0)
        #1 9 256 256
        images_lab_sim_nei1 = torch.cat(images_lab_sim_nei1, dim=0)
        images_lab_sim_nei2 = torch.cat(images_lab_sim_nei2, dim=0)
        images_lab_sim_nei3 = torch.cat(images_lab_sim_nei3, dim=0)
        images_lab_sim_nei4 = torch.cat(images_lab_sim_nei4, dim=0)
        images_lab_sim_nei5 = torch.cat(images_lab_sim_nei5, dim=0)
        images_lab_sim_nei6 = torch.cat(images_lab_sim_nei6, dim=0)
        images_lab_sim_nei7 = torch.cat(images_lab_sim_nei7, dim=0)
        


        images_lab_sim = images_lab_sim.view(-1, target_masks.shape[1], images_lab_sim.shape[-3], images_lab_sim.shape[-2], images_lab_sim.shape[-1])
        #1 8 8 256 256 

        images_lab_sim_nei = images_lab_sim_nei.unsqueeze(1)
        #1 1 9 256 256

        images_lab_sim_nei1 = images_lab_sim_nei1.unsqueeze(1)
        images_lab_sim_nei2 = images_lab_sim_nei2.unsqueeze(1)
        images_lab_sim_nei3 = images_lab_sim_nei3.unsqueeze(1)
        images_lab_sim_nei4 = images_lab_sim_nei4.unsqueeze(1)
        images_lab_sim_nei5 = images_lab_sim_nei5.unsqueeze(1)
        images_lab_sim_nei6 = images_lab_sim_nei6.unsqueeze(1)
        images_lab_sim_nei7 = images_lab_sim_nei7.unsqueeze(1)

    
        #[N*T,K^2,H,W]
        images_lab_sim = torch.cat([images_lab_sim for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1)
        #1 8 8 256 256 -> 첫 객체수에 따라 복제 # 1*n복제 8 8 256 256 -> 앞두개곱 n*8 8 256 256
        
        #images_lab_sim_nei = self.topk_mask(images_lab_sim_nei.flatten(0, 1), 5)
        images_lab_sim_nei = self.topk_mask(torch.cat([images_lab_sim_nei for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)
        images_lab_sim_nei1 = self.topk_mask(torch.cat([images_lab_sim_nei1 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)
       
        images_lab_sim_nei2 = self.topk_mask(torch.cat([images_lab_sim_nei2 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)
     
        images_lab_sim_nei3 = self.topk_mask(torch.cat([images_lab_sim_nei3 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)

        images_lab_sim_nei4 = self.topk_mask(torch.cat([images_lab_sim_nei4 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)
    
        images_lab_sim_nei5 = self.topk_mask(torch.cat([images_lab_sim_nei5 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)

        images_lab_sim_nei6 = self.topk_mask(torch.cat([images_lab_sim_nei6 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)

        images_lab_sim_nei7 = self.topk_mask(torch.cat([images_lab_sim_nei7 for _ in range(src_masks.shape[0])], dim=0).flatten(0, 1), 5)
  
    
        k_size = 3

        if src_masks.shape[0] > 0:
            pairwise_losses_neighbor = compute_pairwise_term_neighbor(
                src_masks[:,:1], src_masks[:,1:2], k_size, 3
            )
            pairwise_losses_neighbor1 = compute_pairwise_term_neighbor(
                src_masks[:,1:2], src_masks[:,2:3], k_size, 3
            )
            pairwise_losses_neighbor2 = compute_pairwise_term_neighbor(
                src_masks[:,2:3], src_masks[:,3:4], k_size, 3
            )
            pairwise_losses_neighbor3 = compute_pairwise_term_neighbor(
                src_masks[:,3:4], src_masks[:,4:5], k_size, 3
            )
            pairwise_losses_neighbor4 = compute_pairwise_term_neighbor(
                src_masks[:,4:5], src_masks[:,5:6], k_size, 3
            )
            pairwise_losses_neighbor5 = compute_pairwise_term_neighbor(
                src_masks[:,5:6], src_masks[:,6:7], k_size, 3
            )
            pairwise_losses_neighbor6 = compute_pairwise_term_neighbor(
                src_masks[:,6:7], src_masks[:,7:8], k_size, 3
            )
            pairwise_losses_neighbor7 = compute_pairwise_term_neighbor(
                src_masks[:,7:8], src_masks[:,0:1], k_size, 3
            )
            
    
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]
        target_masks = F.interpolate(target_masks, (src_masks.shape[-2], src_masks.shape[-1]), mode='bilinear')
      
        
        if src_masks.shape[0] > 0:
            loss_prj_term = compute_project_term(src_masks.sigmoid(), target_masks)
   
            pairwise_losses = compute_pairwise_term( #24 8 128 128
                src_masks, k_size, 2
            )
            weights = (images_lab_sim >= 0.3).float() * target_masks.float()
            

            #24 8 128 128
            target_masks_sum = target_masks.reshape(pairwise_losses_neighbor.shape[0], 8, target_masks.shape[-2], target_masks.shape[-1]).sum(dim=1, keepdim=True)
            #3 1 128 128
            target_masks_sum = (target_masks_sum >= 1.0).float() # ori is 1.0
            #3 1 128 128
            weights_neighbor = (images_lab_sim_nei >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1 , dy 0.5
            weights_neighbor1 = (images_lab_sim_nei1 >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1, dy 0.5
            weights_neighbor2 = (images_lab_sim_nei2 >= 0.05).float() * target_masks_sum # ori is 0.5, 0.01, 0.001, 0.005, 0.0001, 0.02, 0.05, 0.075, 0.1, dy 0.5
            weights_neighbor3 = (images_lab_sim_nei3 >= 0.05).float() * target_masks_sum
            weights_neighbor4 = (images_lab_sim_nei4 >= 0.05).float() * target_masks_sum
            weights_neighbor5 = (images_lab_sim_nei5 >= 0.05).float() * target_masks_sum
            weights_neighbor6 = (images_lab_sim_nei6 >= 0.05).float() * target_masks_sum
            weights_neighbor7 = (images_lab_sim_nei7 >= 0.05).float() * target_masks_sum
            
            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0) #1.0
            #학습 수 설정 나중 갈수록 더 가중치있게

            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            #한 프레임내의 주변픽셀과의 유사도[N*T, H, W] * [N*T, K², H, W] / 전체에서 반영된 픽셀의 총합
            loss_pairwise_neighbor = (pairwise_losses_neighbor * weights_neighbor).sum() / weights_neighbor.sum().clamp(min=1.0) * warmup_factor
            #(예측마스크 0번과 1번마스크 비교 * 0번과 1번마스크사이의 sim_nei 비교 가중치값)의 합
            loss_pairwise_neighbor1 = (pairwise_losses_neighbor1 * weights_neighbor1).sum() / weights_neighbor1.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor2 = (pairwise_losses_neighbor2 * weights_neighbor2).sum() / weights_neighbor2.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor3 = (pairwise_losses_neighbor3 * weights_neighbor3).sum() / weights_neighbor3.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor4 = (pairwise_losses_neighbor4 * weights_neighbor4).sum() / weights_neighbor4.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor5 = (pairwise_losses_neighbor5 * weights_neighbor5).sum() / weights_neighbor5.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor6 = (pairwise_losses_neighbor6 * weights_neighbor6).sum() / weights_neighbor6.sum().clamp(min=1.0) * warmup_factor
            loss_pairwise_neighbor7 = (pairwise_losses_neighbor7 * weights_neighbor7).sum() / weights_neighbor7.sum().clamp(min=1.0) * warmup_factor

                           
        else:
            loss_prj_term = src_masks.sum() * 0.
            loss_pairwise = src_masks.sum() * 0.
            loss_pairwise_neighbor = src_masks.sum() * 0.
            loss_pairwise_neighbor1 = src_masks.sum() * 0.
            loss_pairwise_neighbor2 = src_masks.sum() * 0.
            loss_pairwise_neighbor3 = src_masks.sum() * 0.
            loss_pairwise_neighbor4 = src_masks.sum() * 0.
            loss_pairwise_neighbor5 = src_masks.sum() * 0.
            loss_pairwise_neighbor6 = src_masks.sum() * 0.
            loss_pairwise_neighbor7 = src_masks.sum() * 0.
                

        # print('loss_proj term:', loss_prj_term)
        losses = {
            "loss_proj": loss_prj_term,
            "loss_pairwise": loss_pairwise,
            "loss_tk": (loss_pairwise_neighbor
                                    + loss_pairwise_neighbor1
                                    + loss_pairwise_neighbor2
                                    + loss_pairwise_neighbor3
                                    + loss_pairwise_neighbor4
                                    + loss_pairwise_neighbor5
                                    + loss_pairwise_neighbor6
                                    + loss_pairwise_neighbor7
                                   ) * 0.1, # * 0.33
        }

        del src_masks
        del target_masks
        return losses["loss_proj"], losses["loss_pairwise"], losses["loss_tk"]

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_tk" : 0, "loss_proj" : 0, "loss_pairwise" : 0 }
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
