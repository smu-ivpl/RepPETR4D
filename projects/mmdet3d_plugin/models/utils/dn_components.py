# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch


def prepare_for_dn(img_metas, dn_args, temp_args, training, batch_size, reference_points):
    if training:
        scalar, bbox_noise_scale, box_noise_train, num_classes, split, pc_range, scale_range, yaw_range, vel_range = dn_args
        num_propagated, memory_len = temp_args
        scalar = scalar * 2 # for cdn
        num_query = reference_points.shape[0]
        targets = [
            torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]), dim=1) for img_meta in img_metas]
        labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
        known = [(torch.ones_like(t)).cuda() for t in labels]
        know_idx = known
        unmask_bbox = unmask_label = torch.cat(known)
        # gt_num
        known_num = [t.size(0) for t in targets]
        if int(max(known_num)) == 0:
            scalar = 1
        else:
            if scalar >= 100:
                scalar = scalar // (int(max(known_num)))
            elif scalar < 1:
                scalar = 1
        if scalar == 0:
            scalar = 1

        labels = torch.cat([t for t in labels])
        boxes = torch.cat([t for t in targets])
        batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        # add noise
        known_indice = known_indice.repeat(2 * scalar, 1).view(-1)
        known_labels = labels.repeat(2 * scalar, 1).view(-1).long().to(reference_points.device)
        known_bid = batch_idx.repeat(2 * scalar, 1).view(-1)
        known_bboxs = boxes.repeat(2 * scalar, 1).to(reference_points.device)
        known_bboxs_expand_tmp = known_bboxs.clone()
        known_bbox_expand_center = known_bboxs[..., :3].clone()
        known_bbox_expand_scale = known_bboxs[..., 3:6].clone()
        known_bbox_expand_rot_x = known_bboxs[..., 6:7].clone()
        known_bbox_expand_vel = known_bboxs[..., 7:9].clone()

        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(scalar, 1)
        positive_idx += (torch.tensor(range(scalar)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if bbox_noise_scale > 0:
            diff = torch.zeros_like(known_bboxs_expand_tmp).cuda()
            diff[:, :3] = known_bbox_expand_scale / 2  # l/2, w/2, h/2
            diff[:, 3:6] = known_bbox_expand_scale  # l/2, w/2, h/2
            diff[:, 6:7] = known_bbox_expand_rot_x  # rot
            diff[:, 7:9] = known_bbox_expand_vel  #vel_x, vel_y

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0

            rand_part *= rand_sign
            known_bbox_expand_center += torch.mul(rand_part[..., :3], diff[..., :3])  # noise_scale
            known_bbox_expand_scale += torch.mul(rand_part[..., 3:6], diff[..., 3:6]) * 0.5  # noise_scale
            known_bbox_expand_rot_x += torch.mul(rand_part[..., 6:7], torch.ones_like(known_bbox_expand_rot_x)) * 1.5
            known_bbox_expand_vel += torch.mul(rand_part[..., 7:9], torch.ones_like(known_bbox_expand_vel)) * 12
            known_bbox_expand = torch.cat([known_bbox_expand_center, known_bbox_expand_scale, known_bbox_expand_rot_x.sin(),known_bbox_expand_rot_x.cos(), known_bbox_expand_vel], dim=-1)

            # img2lidar normalization
            known_bbox_expand[:, 0:3] = (known_bbox_expand[:, 0:3] - pc_range[0:3]) / (pc_range[3:6] - pc_range[0:3])

            known_bbox_expand[:, 3:6] = (known_bbox_expand[:, 3:6].log() - scale_range[0:3].log()) / (scale_range[3:6].log() - scale_range[0:3].log())

            known_bbox_expand[:, 6:8] = (known_bbox_expand[:, 6:8] - yaw_range[0:2]) / (yaw_range[2:4] - yaw_range[0:2])
            known_bbox_expand[:, 8:10] = (known_bbox_expand[:, 8:10] - vel_range[0:2]) / (vel_range[2:4] - vel_range[0:2])

            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * scalar)
        padding_bbox = torch.zeros(pad_size, 10).to(reference_points.device)
        input_query_bbox = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        # padding_bbox : noised query / reference_points : normal query

        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * scalar)]).long()
        if len(known_bid):
            input_query_bbox[(known_bid.long(), map_known_indice)] = known_bbox_expand.to(reference_points.device)

        tgt_size = pad_size + num_query
        attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        # update dn mask for temporal modeling
        query_size = pad_size + num_query + num_propagated  # pad_size:330, num_query: 644, num_propagated: 256 -> 1230
        tgt_size = pad_size + num_query + memory_len
        temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
        temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask
        temporal_attn_mask[pad_size:, :pad_size] = True
        attn_mask = temporal_attn_mask

        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size,
            'num_dn_group': scalar,
            'known_num': known_num,
        }

    else:
        input_query_bbox = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    return input_query_bbox, attn_mask, mask_dict


def dn_post_process(all_cls_scores, all_bbox_preds, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:  # dn_post_process
        output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
        output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
        outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
        outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
        outs = {
            'all_cls_scores': outputs_class,
            'all_bbox_preds': outputs_coord,
            'dn_mask_dict': mask_dict,

        }
    else:
        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'dn_mask_dict': None,
        }
    return outs


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice'].long()

    known_indice = mask_dict['known_indice'].long().cpu()

    batch_idx = mask_dict['batch_idx'].long()
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    known_num = mask_dict['known_num']
    scalar = mask_dict['num_dn_group']
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt, known_num, scalar


