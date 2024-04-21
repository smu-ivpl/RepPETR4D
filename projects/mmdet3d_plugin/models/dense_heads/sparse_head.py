import torch
import torch.nn as nn 
from mmcv.cnn import Linear, bias_init_with_prob, Scale

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from projects.mmdet3d_plugin.models.utils.dn_components import prepare_for_dn, dn_post_process, prepare_for_loss

import copy
from mmdet.models.utils import NormedLinear
from pyquaternion import Quaternion
@HEADS.register_module()
class SparseHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stride=[16],
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 num_cls_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 LID=True,
                 depth_step=0.8,
                 depth_num=64,
                 depth_start=1,
                 with_position=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights
            
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is SparseHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.num_cls_fcs = num_cls_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.with_dn = with_dn
        self.stride=stride
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        self.LID = LID
        self.with_position = with_position
        self.position_range = position_range
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.depth_start = depth_start

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(SparseHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(self.bbox_coder.pc_range), requires_grad=False)

        self.scale_range = nn.Parameter(torch.tensor([0.1, 0.1, 0.1, 15.0, 5.0, 6.0]), requires_grad=False)
        self.yaw_range = nn.Parameter(torch.tensor([-1.0, -1.0, 1.0, 1.0]), requires_grad=False)
        self.vel_range = nn.Parameter(torch.tensor([-7.0, -13.0, 7.0, 13.0]), requires_grad=False)

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)
        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        cls_branch = []
        for _ in range(self.num_cls_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.reference_points = nn.Embedding(self.num_query, 10)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 10)


        self.center_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.scale_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.rot_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.vel_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        )

        self.spatial_alignment = MLN(14, use_ln=False)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        self.context_embed = nn.Sequential(
            nn.Linear(self.in_channels+1, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        #The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        self.transformer.init_weights()


        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)

        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 10)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose  # ego pose (t-1 -> t)

            if self.memory_metas is not None:
                inter_time = (data['timestamp'] - self.memory_metas['timestamp']).expand(self.memory_reference_point.shape[1], 2, B).permute(2, 0, 1)
                memory_velo = self.memory_reference_point[..., -2:].clone()
                self.memory_reference_point[..., :2] = self.memory_reference_point[...,:2] - inter_time * memory_velo  # explicit way

            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], data['l2e_rotation_inv'], reverse=True)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)

        if self.num_propagated > 0:
            pseudo_reference_points_center = (self.pseudo_reference_points.weight[:, 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
            pseudo_reference_points_scale = (self.pseudo_reference_points.weight[..., 3:6] * (self.scale_range[3:6].log() - self.scale_range[0:3].log()) + self.scale_range[0:3].log())
            pseudo_reference_points_rot = (self.pseudo_reference_points.weight[..., 6:8] * (self.yaw_range[2:4] - self.yaw_range[0:2]) + self.yaw_range[0:2])
            pseudo_reference_points_vel = (self.pseudo_reference_points.weight[..., 8:10] * (self.vel_range[2:4] - self.vel_range[0:2]) + self.vel_range[0:2])

            pseudo_reference_points = torch.cat([pseudo_reference_points_center, pseudo_reference_points_scale, pseudo_reference_points_rot, pseudo_reference_points_vel], dim=1)

            self.memory_reference_point[:, :self.num_propagated] = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated] = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        self.memory_metas = data

        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :10][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :10][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()  # (1, 256, 256)
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], data['l2e_rotation'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point_center = (self.memory_reference_point[..., 0:3] - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_reference_point_scale = (self.memory_reference_point[..., 3:6] - self.scale_range[0:3].log()) / (self.scale_range[3:6].log() - self.scale_range[0:3].log())
        temp_reference_point_rot = (self.memory_reference_point[..., 6:8] - self.yaw_range[0:2]) / (self.yaw_range[2:4] - self.yaw_range[0:2])
        temp_reference_point_vel = (self.memory_reference_point[..., 8:] - self.vel_range[0:2]) / (self.vel_range[2:4] - self.vel_range[0:2])

        temp_reference_point = torch.cat([temp_reference_point_center, temp_reference_point_scale, temp_reference_point_rot, temp_reference_point_vel], dim=2)

        temp_sine_embed = pos2posemb3d(self.memory_reference_point)
        temp_center_embed = self.center_embedding(temp_sine_embed[..., :self.embed_dims * 3 // 2])
        temp_scale_embed = self.scale_embedding(temp_sine_embed[..., self.embed_dims * 3 // 2:self.embed_dims * 3])
        temp_rot_embed = self.rot_embedding(temp_sine_embed[..., self.embed_dims * 3:self.embed_dims * 4])
        temp_vel_embed = self.vel_embedding(temp_sine_embed[..., self.embed_dims * 4:self.embed_dims * 5])

        temp_reference_embed = temp_center_embed+temp_scale_embed+temp_rot_embed+temp_vel_embed
        temp_pos = self.query_embedding(temp_reference_embed)

        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[..., :3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[
                1] + self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    @torch.no_grad()
    def build_query2d_proposal(self, pred_bbox_list, data, bn, padHW, bbox2d_scores=None, context2d_feat=None):
        '''
        pred_centers2d: ~~(B*N H*W 2)~~, now is a list, BN*(Mi, 4)
        pred_depth: (B*N, H, W, 1) if not use topk depth proposals else (BN, H, W, D)
        pred_depth_var: (B*N, 1, H, W)
        '''
        B, N = bn
        eps = 1e-5
        n_d = 2 # the number of depths

        # bbox list to (sum(Mi), 2)
        bbox_nums = [len(bbox) for bbox in pred_bbox_list]  # BN values
        bboxes = [torch.cat(pred_bbox_list[N * i:N * (i + 1)]) for i in range(B)]  # gather boxes together

        if sum(bbox_nums) == 0:  # no effective 2d proposal
            return None, None

        # convert bin to float depth /
        depths = torch.linspace(0, 51.2, n_d + 2)[1:-1].unsqueeze(-1).to(bboxes[0].device)

        if bbox2d_scores is not None:
            thr = torch.tensor([0.1]).to(bbox2d_scores.device)  # score threshold
            log_odds = torch.log(bbox2d_scores / (1 - bbox2d_scores)) - torch.log(thr / (1 - thr))  # (M, 1)
            if context2d_feat is not None:
                context2d_feat = torch.cat([context2d_feat, log_odds], dim=-1)  # check dim cat
                context2d_feat = context2d_feat.unsqueeze(1).expand(-1, depths.shape[0], -1).reshape(-1, context2d_feat.shape[-1])
            context2d_feat = context2d_feat.reshape(B, -1, context2d_feat.shape[-1])


        bboxes = [bboxes_.unsqueeze(1).expand(-1, depths.shape[0], -1).reshape(-1, bboxes_.shape[-1]) for bboxes_ in bboxes]
        # (u,v), d -> (ud,vd,d,1)
        new_reference_points_lst = []  # (B, M_i, 4, 1)

        img2lidars = data['lidar2img'].inverse()  # (B, N, 4, 4)
        img2lidars = [torch.cat(
            [img2lidars[i][kth].repeat(num * depths.shape[0], 1, 1) for kth, num in enumerate(bbox_nums[N * i:N * (i + 1)])]) for i in
                      range(B)]

        for bboxes_, img2lidars_ in zip(bboxes, img2lidars):
            if bboxes_.shape[0] == 0:
                new_reference_points_lst.append(torch.tensor([]).reshape(1, 0, 2))
                continue

            coords = torch.cat([bboxes_[:, :2], depths.repeat(bboxes_.shape[0] // depths.shape[0], 1)],
                               dim=1)  # (M * 3, 4), order is (w, h, d)
            coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)  # (M * 3, 4)
            coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)
            coords = coords.unsqueeze(-1)  # (M, 4, 1)

            # matmul and normalize 3d coords
            coords3d = torch.matmul(img2lidars_, coords).squeeze(-1)[..., :3]  # (M, 3)

            coords3d[:, 0:3] = (coords3d[:, 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

            coords3d = coords3d.clamp(min=0.0, max=1.0)

            new_reference_points = coords3d.unsqueeze(0)  # (B, M, 3)
            new_reference_points_lst.append(new_reference_points)
        ref_2d_points = torch.cat(new_reference_points_lst, dim=0)
        return ref_2d_points, context2d_feat

    def forward(self, img_metas, outs_roi, **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        self.pre_update_memory(data)
        mlvl_feats = data['img_feats']
        B = mlvl_feats[0].size(0)

        reference_points = self.reference_points.weight
        dtype = reference_points.dtype
        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1)
        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten.append(mlvl_feat.to(dtype))
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1, )), spatial_flatten.prod(1).cumsum(0)[:-1]))

        # prepare for denoising strategy
        reference_points, attn_mask, mask_dict = prepare_for_dn(img_metas,
                                                                dn_args=(self.scalar, self.bbox_noise_scale, self.bbox_noise_trans, self.num_classes, self.split, self.pc_range,self.scale_range, self.yaw_range, self.vel_range),
                                                                temp_args=(self.num_propagated, self.memory_len),
                                                                training=self.training, batch_size=B,
                                                                reference_points=reference_points)

        # generate extra queries using 2d proposal
        if outs_roi['bbox2d_scores'].shape[0] > 0:
            _dim = feat_flatten.shape[-1]
            bbox2d_scores = outs_roi['bbox2d_scores'].detach()
            valid_indices = outs_roi['valid_indices']
            context_feat = feat_flatten[valid_indices.repeat(1, 1, _dim)].reshape(-1, _dim)
            context2d_feat = context_feat.detach()
            pred_bbox_list = [it.detach() for it in outs_roi['bbox_list']]
            ref_2d_proposal, context_feat = self.build_query2d_proposal(pred_bbox_list, data, (B, N),
                                                                            img_metas[0]['pad_shape'][0][:2],
                                                                            bbox2d_scores=bbox2d_scores,
                                                                            context2d_feat=context2d_feat)

            if ref_2d_proposal.shape[1] > 644:
                cloned_ref_points = self.reference_points.weight.clone().unsqueeze(0).repeat(B, 1, 1)[:, :, 3:]
                for i in range(1, (ref_2d_proposal.shape[1] // 644)):
                    cloned_ref_points = torch.cat([cloned_ref_points, self.reference_points.weight.clone().unsqueeze(0).repeat(B, 1, 1)[:, :, 3:]], dim=1)
                if ref_2d_proposal.shape[1] %644 != 0:
                    cloned_ref_points = torch.cat(
                        [cloned_ref_points, self.reference_points.weight.clone().unsqueeze(0).repeat(B, 1, 1)[:, :int(ref_2d_proposal.shape[1] % 644), 3:]],
                        dim=1)
            else:
                cloned_ref_points = self.reference_points.weight.clone().unsqueeze(0).repeat(B, 1, 1)[:, :ref_2d_proposal.shape[1], 3:]

            ref_2d_proposal = torch.cat([ref_2d_proposal, cloned_ref_points], dim=-1)
            reference_points = torch.cat([reference_points, ref_2d_proposal], dim=1)

            if self.training:
                pad_size = mask_dict['pad_size']
                origin_query_size = pad_size + self.num_query + self.num_propagated
                origin_tgt_size = pad_size + self.num_query + self.memory_len
                query_size = origin_query_size + ref_2d_proposal.shape[1]
                tgt_size = origin_tgt_size + ref_2d_proposal.shape[1]
                attn_mask_ = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
                attn_mask_[:origin_query_size, :origin_tgt_size] = attn_mask
                attn_mask_[pad_size:, :pad_size] = True
                attn_mask = attn_mask_
            else:
                attn_mask = None


        query_sine_embed = pos2posemb3d(reference_points)
        query_center_embed = self.center_embedding(query_sine_embed[..., :self.embed_dims * 3 // 2])
        query_scale_embed = self.scale_embedding(query_sine_embed[..., self.embed_dims * 3 // 2:self.embed_dims * 3])
        query_rot_embed = self.rot_embedding(query_sine_embed[..., self.embed_dims * 3:self.embed_dims * 4])
        query_vel_embed = self.vel_embedding(query_sine_embed[..., self.embed_dims * 4:self.embed_dims * 5])
        query_pos = self.query_embedding(query_center_embed+query_scale_embed+query_rot_embed+query_vel_embed)

        tgt = torch.zeros_like(query_pos)

        if outs_roi['bbox2d_scores'].shape[0] > 0:
            cont_f = self.context_embed(context_feat)
            tgt[:, -ref_2d_proposal.shape[1]:, :] = cont_f

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)


        self.transformer.decoder.reg_branches = self.reg_branches  # for layer-by-layer
        self.transformer.decoder.query_embedding = self.query_embedding
        self.transformer.decoder.center_embedding = self.center_embedding
        self.transformer.decoder.scale_embedding = self.scale_embedding
        self.transformer.decoder.rot_embedding = self.rot_embedding
        self.transformer.decoder.vel_embedding = self.vel_embedding

        outs_dec, references = self.transformer(tgt, query_pos, feat_flatten, spatial_flatten, level_start_index, temp_memory,
                                    temp_pos, attn_mask, reference_points, (self.pc_range, self.scale_range, self.yaw_range, self.vel_range), data, img_metas)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []

        reference_before_sigmoid = inverse_sigmoid(references.clone())

        for lvl in range(self.num_pred):
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])

            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:10] += reference_before_sigmoid[lvl]
            tmp[..., 0:10] = tmp[..., 0:10].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        all_bbox_preds[..., 3:6] = (all_bbox_preds[..., 3:6] * (self.scale_range[3:6].log() - self.scale_range[0:3].log()) + self.scale_range[0:3].log())
        all_bbox_preds[..., 6:8] = (all_bbox_preds[..., 6:8] * (self.yaw_range[2:4] - self.yaw_range[0:2]) + self.yaw_range[0:2])
        all_bbox_preds[..., 8:10] = (all_bbox_preds[..., 8:10] * (self.vel_range[2:4] - self.vel_range[0:2]) + self.vel_range[0:2])

        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        outs = dn_post_process(all_cls_scores, all_bbox_preds, mask_dict)

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)

        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    known_labels_weight,
                    known_bboxs_weight,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        #num_imgs = cls_scores.size(0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_pos * self.bg_cls_weight # num_total_pos == num_total_neg with CDN
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), known_labels_weight, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        known_bboxs_weight = known_bboxs_weight * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], known_bboxs_weight[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        if hasattr(self, 'context_embed'):
            loss_dict['loss_cls'] += 0.0 * (
                        self.context_embed[0].weight.sum() + self.context_embed[0].bias.sum() + self.context_embed[
                    2].weight.sum() + self.context_embed[2].bias.sum())



        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt, known_num, scalar = prepare_for_loss(preds_dicts['dn_mask_dict'])


            known_bbox_targets = torch.zeros_like(known_bboxs)
            known_bbox_labels = known_bboxs.new_full((num_tgt, ), self.num_classes, dtype=torch.long)
            known_labels_weights = torch.ones_like(known_labels)
            known_bbox_weights = known_bboxs.new_full((num_tgt, self.code_size), 0.0)

            num_bboxes = sum(known_num)
            if num_tgt > 0:
                t = torch.range(0, num_bboxes - 1).long().cuda()
                t = t.unsqueeze(0).repeat(scalar, 1)
                dn_pos_ind = (torch.tensor(range(scalar)) * num_bboxes * 2).long().cuda().unsqueeze(1) + t
                dn_pos_ind = dn_pos_ind.flatten()
                known_bbox_targets[dn_pos_ind] = known_bboxs[dn_pos_ind]
                known_bbox_labels[dn_pos_ind] = known_labels[dn_pos_ind]
                known_bbox_weights[dn_pos_ind] = 1.0

            all_known_bboxs_list = [known_bbox_targets for _ in range(num_dec_layers)]
            all_known_labels_list = [known_bbox_labels for _ in range(num_dec_layers)]
            all_known_labels_weights_list = [known_labels_weights for _ in range(num_dec_layers)]
            all_known_bbox_weights_list = [known_bbox_weights for _ in range(num_dec_layers)]
            all_num_tgts_list = [num_tgt//2 for _ in range(num_dec_layers)]

            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, all_known_labels_weights_list, all_known_bbox_weights_list,
                all_num_tgts_list) # decoder layer 수 만큼 실행
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
