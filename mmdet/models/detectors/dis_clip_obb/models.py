import torch
import torch.nn as nn
import open_clip
from clip import clip
from mmdet import ops
import numpy
from torch.nn.modules.utils import _pair
from mmdet.models.builder import BACKBONES
from mmdet.core import arb2result, arb2roi, build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head, build_roi_extractor, build_loss
from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_arb_nms, get_bbox_dim, bbox2type)
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.core import obb2hbb
from mmdet.ops import arb_batched_nms
from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.dense_heads.rpn_test_mixin import RPNTestMixin
from mmdet.models.dense_heads.obb.obb_anchor_head import OBBAnchorHead
from mmdet.models.roi_heads.roi_extractors.obb.obb_base_roi_extractor import OBBBaseRoIExtractor
import random
# from tensorboardX import SummaryWriter

# import torchvision.utils as vutils
# torch.autograd.set_detect_anomaly(True)

# writer = SummaryWriter("logs")
dota1_0_classes_names = ['large vehicle', 'swimming pool', 'helicopter', 'bridge',
                 'plane', 'ship', 'soccer ball field', 'basketball court',
                 'ground track field', 'small vehicle', 'baseball diamond',
                 'tennis court', 'roundabout', 'storage-tank', 'harbor']

dota1_5_classes_names = ['large vehicle', 'swimming pool', 'helicopter', 'bridge',
                 'plane', 'ship', 'soccer ball field', 'basketball court',
                 'ground track field', 'small vehicle', 'baseball diamond',
                 'tennis court', 'roundabout', 'storage-tank', 'harbor', 'container crane']


@BACKBONES.register_module()
class DistillationCLIP(nn.Module):
    def __init__(self,
                 clip_encoder_name,
                 obj_weight,
                 con_weight,
                 sem_weight,
                 unfreeze=None):
        super().__init__()
        # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(clip_encoder_name)
        self.clip_model, self.preprocess = clip.load(clip_encoder_name, 'cuda')
        self.clip_model.float()
        self.enc = self.clip_model.visual
        # self.tokenizer = open_clip.get_tokenizer(clip_encoder_name)
        self.obj_weight = obj_weight
        self.con_weight = con_weight
        self.sem_weight = sem_weight
        self.unfreeze = unfreeze
        self.classes_names = dota1_5_classes_names
        # self.classes_names = dota1_0_classes_names


        if unfreeze is not None:
            for name, val in self.enc.named_parameters():
                head = name.split('.')[0]
                if head not in self.unfreeze:
                    val.requires_grad = False
                else:
                    val.requires_grad = True

        # fg_prompt = 'a remote sensing image of a {}'
        self.fg_prompt = [
            "An aerial photograph of a {} captured by a satellite",
            "A high-resolution satellite image showing a {}",
            "A top-down view from a satellite, focusing on a {}",
            "A bird's-eye view of a {} in a remote sensing image",
            "A detailed satellite image showing the structure of a {}",

        ]
        # bg_prompt = 'a remote sensing image of an unknown class'
        self.bg_prompt = [
            "A remote sensing image of an unidentified object",
            "An aerial view of an unknown structure or object in a satellite image",
            "A satellite image showing an unclassified object",
            "An aerial photograph of an object of uncertain classification",
            "A remote sensing image with an unknown element in the scene"
        ]

        # self.clip_model = self.clip_model.cuda()

        # with torch.no_grad():
        #     text_inputs = torch.cat(
        #         [self.tokenizer(random.choice(self.fg_prompt).format(cls)) for cls in self.classes_names]).cuda()
        #     self.text_features = self.clip_model.encode_text(text_inputs)
        #     self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        #
        # with torch.no_grad():
        #     bg_text_inputs = self.tokenizer(random.choice(self.bg_prompt)).cuda()
        #     self.bg_text_features = self.clip_model.encode_text(bg_text_inputs)
        #     self.bg_text_features = self.bg_text_features / self.bg_text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            text_inputs = torch.cat(
                [clip.tokenize(random.choice(self.fg_prompt).format(cls)) for cls in self.classes_names]).cuda()
            self.text_features = self.clip_model.encode_text(text_inputs).float()
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            bg_text_inputs = clip.tokenize(random.choice(self.bg_prompt)).cuda()
            self.bg_text_features = self.clip_model.encode_text(bg_text_inputs).float()
        self.bg_text_features = self.bg_text_features / self.bg_text_features.norm(dim=-1, keepdim=True)

        # bg_prompts, tokenized_bg_prompt = self.prompt_learner.forward()
        # with torch.no_grad():
        #     self.bg_features = self.text_encoder(bg_prompts, tokenized_bg_prompt)
        # self.bg_features = self.bg_features / self.bg_features.norm(dim=-1, keepdim=True)


    def object_distillation(self,
                            bbox_feats_list,
                            bbox_imgs):
        """
        Args:
            bbox_feats_list (list[Tensor]): bbox features from scale_head, everyone with same number of rois.
            bbox_feats_list: [[1024,256,56,56], [1024, 512, 28, 28], [1024, 1024, 14,14], [1024, 2048, 7, 7]]
            bbox_imgs (Tensor [roi_number, 3, 224, 224]): img form scale_head with same number of rois.

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        bbox_imgs_feats = self.forward_layers(bbox_imgs)
        # optimization
        level_weights = [0., 1., 1., 1.]
        total_obj_loss = 0

        for i in range(len(bbox_imgs_feats)):
            obj_loss = torch.nn.functional.l1_loss(bbox_imgs_feats[i], bbox_feats_list[i])
            obj_loss = obj_loss * level_weights[i]
            total_obj_loss = total_obj_loss + obj_loss

        obj_weight = self.obj_weight

        return total_obj_loss * obj_weight

    def constant_distillation(self,
                              bbox_feats_list,
                              bbox_targets):
        """
        Args:
            bbox_feats_list (list[Tensor]): bbox features from scale_head, everyone with same number of rois.
            bbox_feats_list: [[1024,256,56,56], [1024, 512, 28, 28], [1024, 1024, 14,14], [1024, 2048, 7, 7]]
            bbox_targets (Tensor) : label of bbox, label, label weight, bbox_label, box weight

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        # with torch.no_grad():
        #     text_inputs = torch.cat([self.tokenizer(random.choice(self.fg_prompt).format(cls)) for cls in self.classes_names]).cuda()
        #     self.text_features = self.clip_model.encode_text(text_inputs.cuda())
        # self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        #
        # with torch.no_grad():
        #     bg_text_inputs = self.tokenizer(random.choice(self.bg_prompt)).cuda()
        #     self.bg_text_features = self.clip_model.encode_text(bg_text_inputs.cuda())
        # self.bg_text_features = self.bg_text_features / self.bg_text_features.norm(dim=-1, keepdim=True)

        # optimization
        labels = bbox_targets[0]
        total_lin_loss = 0

        rot_bbox_feats0 = torch.rot90(bbox_feats_list[0], k=1, dims=[2, 3])
        bbox_feats0 = self.forward_layer234(rot_bbox_feats0)
        att_bbox_feat0 = self.attention_global_pool(bbox_feats0)
        att_bbox_feat0 = att_bbox_feat0 / att_bbox_feat0.norm(dim=-1, keepdim=True)

        rot_bbox_feats1 = torch.rot90(bbox_feats_list[1], k=2, dims=[2, 3])
        bbox_feats1 = self.forward_layer34(rot_bbox_feats1)
        att_bbox_feat1 = self.attention_global_pool(bbox_feats1)
        att_bbox_feat1 = att_bbox_feat1 / att_bbox_feat1.norm(dim=-1, keepdim=True)

        rot_bbox_feats2 = torch.rot90(bbox_feats_list[2], k=3, dims=[2, 3])
        bbox_feats2 = self.forward_layer4(rot_bbox_feats2)
        att_bbox_feat2 = self.attention_global_pool(bbox_feats2)
        att_bbox_feat2 = att_bbox_feat2 / att_bbox_feat2.norm(dim=-1, keepdim=True)

        # rot_bbox_feats3 = torch.rot90(bbox_feats_list[3], k=0, dims=[2, 3])
        bbox_feats3 = bbox_feats_list[3]
        att_bbox_feat3 = self.attention_global_pool(bbox_feats3)
        att_bbox_feat3 = att_bbox_feat3 / att_bbox_feat3.norm(dim=-1, keepdim=True)

        att_bbox_feats_list = [att_bbox_feat0, att_bbox_feat1, att_bbox_feat2, att_bbox_feat3]
        # temperature = 1
        num_class = len(self.classes_names)
        #multi-modal constant
        for i in range(len(labels)):
            label = labels[i]
            if label != num_class:
                text_feats = self.text_features[label].unsqueeze(0).repeat(4, 1)
                # text_feats = self.text_features[label].unsqueeze(0).repeat(2, 1)
            else:
                text_feats = self.bg_text_features.repeat(4, 1)
                # text_feats = self.bg_text_features.repeat(2, 1)
            att_bbox_feat = torch.stack([att_bbox_feats[i] for att_bbox_feats in att_bbox_feats_list])
            # att_bbox_feat = att_bbox_feat[:2]

            lin_loss = (1 - torch.cosine_similarity(att_bbox_feat, text_feats, dim=1)).mean()

            # lin_loss = torch.sum(
            #     F.smooth_l1_loss(att_bbox_feat, text_feats, reduction='none'), dim=1).mean()

            # lin_loss = torch.sum(
            #     F.mse_loss(att_bbox_feat, text_feats, reduction='none'), dim=1).mean()

            total_lin_loss = total_lin_loss + lin_loss

        # visual constant
        # for i in range(len(att_bbox_feats_list)):
        #     for j in range(i + 1, len(att_bbox_feats_list)):
        #         vis_loss = (1 - torch.cosine_similarity(att_bbox_feats_list[i], att_bbox_feats_list[j], dim=1)).mean()
        #         total_vis_loss = total_vis_loss + vis_loss

        total_con_loss = total_lin_loss
        con_weight = self.con_weight

        return total_con_loss * con_weight, att_bbox_feats_list

    def semantic_distillation(self, bbox_targets, att_bbox_feats_list):
        """
        Args:
            bbox_targets (tuple) : label of bbox, label, label weight, bbox_label, box weight

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        # optimization
        total_cls_loss = 0
        labels = bbox_targets[0]

        for i in range(len(att_bbox_feats_list)):
            temperature = 0.01
            # if i == 0 or i == 1:
            scores = (torch.matmul(att_bbox_feats_list[i], self.text_features.detach().T) / temperature)
            bg_scores = (torch.matmul(att_bbox_feats_list[i], self.bg_text_features.detach().T) / temperature)
            scores = torch.cat([scores, bg_scores], 1)

            cls_loss = F.cross_entropy(scores, labels.cuda())

            # # 将 labels 转换为 one-hot：目标分布 P
            # target_prob = F.one_hot(labels, num_classes=17).float()
            # # 预测分布 Q：使用 log_softmax 便于 KLDivLoss
            # log_pred_prob = F.log_softmax(scores, dim=1)
            # # KL 散度：P 是目标分布，log(Q) 是 log_pred_prob
            # cls_loss = F.kl_div(log_pred_prob, target_prob, reduction='batchmean')

            total_cls_loss = total_cls_loss + cls_loss
        # feat [roi_number,512] text [15, 512] scores [roi_number, 15]
        # print(scores.min(),scores.max())

        # add for bkg class a score 0
        sem_weight = self.sem_weight
        return total_cls_loss * sem_weight

    def forward(self, bbox_feats_list, bbox_targets, bbox_imgs):
        con_losses, att_bbox_feats_list = self.constant_distillation(bbox_feats_list, bbox_targets)
        # obj_losses = self.object_distillation(bbox_feats_list, bbox_imgs)
        sem_losses = self.semantic_distillation(bbox_targets, att_bbox_feats_list)

        # return obj_losses, con_losses, sem_losses

        return con_losses, sem_losses

    def forward_layers(self, x):
        x = self.enc.relu(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        x1 = self.enc.layer1(x)
        x2 = self.enc.layer2(x1)
        x3 = self.enc.layer3(x2)
        x4 = self.enc.layer4(x3)
        return [x1, x2, x3, x4]

    def forward_layer234(self, x):
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        x = self.enc.layer4(x)
        return x

    def forward_layer34(self, x):
        x = self.enc.layer3(x)
        x = self.enc.layer4(x)
        return x

    def forward_layer4(self, x):
        x = self.enc.layer4(x)
        return x

    def attention_global_pool(self, x):
        x = self.enc.attnpool(x)
        return x

@HEADS.register_module()
class ScaleHead(RPNTestMixin, OBBAnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 img_roi_scale,
                 num_classes,
                 assigner,
                 sampler,
                 bbox_roi_extractor,
                 **kwargs):
        super(ScaleHead, self).__init__(
            1,
            in_channels,
            bbox_type='obb',
            reg_dim=6,
            background_label=0,
            **kwargs)

        self.img_roi_scale = img_roi_scale
        self.num_classes = num_classes

        self.train_cfg = kwargs.get('train_cfg', None)
        # self.proposal_cfg
        """Initialize bbox_roi_extractor"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

        """Initialize assigner and sampler"""
        self.bbox_assigner = build_assigner(assigner)
        self.bbox_sampler = build_sampler(sampler, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.bbox_roi_extractor, std=0.01)
        # self.bbox_roi_extractor.init_weights()

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(ScaleHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_sca_cls=losses['loss_cls'], loss_sca_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_dim)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w, h = proposals[:, 2], proposals[:, 3]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        hproposals = obb2hbb(proposals)
        nms_cfg = dict(type='nms', iou_thr=cfg.nms_thr)
        _, keep = arb_batched_nms(hproposals, scores, ids, nms_cfg)

        dets = torch.cat([proposals, scores[:, None]], dim=1)
        dets = dets[keep]
        return dets[:cfg.nms_post]

    def _get_target_single(self, pos_bboxes: object, neg_bboxes: object, pos_gt_bboxes: object,
                           pos_gt_labels: object, cfg: object) -> object:
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        target_dim = self.reg_dim if not self.reg_decoded_bbox \
            else get_bbox_dim(self.end_bbox_type)
        bbox_targets = pos_bboxes.new_zeros(num_samples, target_dim)
        bbox_weights = pos_bboxes.new_zeros(num_samples, target_dim)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets_roi(self,
                        sampling_results,
                        gt_bboxes,
                        gt_labels,
                        rcnn_train_cfg,
                        concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def scale_img_roi(self, scale, img, roi):
        img_scale = nn.Upsample(size=(int(img.shape[-2] * scale), int(img.shape[-1] * scale)), mode='bilinear')(img)
        rois_scale = roi.clone()
        rois_scale[:, 1:5] = roi[:, 1:5] * scale
        return img_scale, rois_scale

    def forward_train(self,
                      img,
                      x_backbone,
                      x,
                      img_metas,
                      gt_obboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)

        loss_inputs = outs + (gt_obboxes, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)

        #ROI
        target_bboxes = gt_obboxes
        target_bboxes_ignore = gt_bboxes_ignore
        num_imgs = len(img_metas)
        if target_bboxes_ignore is None:
            target_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], target_bboxes[i],
                target_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                target_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        #ROI_extractor
        rois = arb2roi([res.bboxes for res in sampling_results], bbox_type='obb')
        bbox_feats_list = self.bbox_roi_extractor(x_backbone, rois)

        #ROI_extractor for imgs rois=[_,6]
        img_scale, rois_scale = self.scale_img_roi(self.img_roi_scale, img, rois)
        bbox_imgs = self.bbox_roi_extractor.forward_img(img_scale, rois_scale)

        #Visualization
        # img1 = img_scale[0]
        # img1 = vutils.make_grid(img1, normalize=True)
        # writer.add_image('img', img1, dataformats='CHW')
        # bbox_imgs1 = bbox_imgs
        # for i in range(len(bbox_imgs)):
        #     bbox_imgs1[i] = vutils.make_grid(bbox_imgs1[i], normalize=True)
        #     writer.add_image(f'bbox_img{i}', bbox_imgs1[i], dataformats='CHW')

        #Bbox_head
        bbox_targets = self.get_targets_roi(sampling_results, gt_obboxes,
                                            gt_labels, self.train_cfg)

        return losses, bbox_feats_list, bbox_targets, bbox_imgs


@ROI_EXTRACTORS.register_module()
class OBBScaleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 roi_layer,
                 out_size,
                 clip_input_size,
                 featmap_strides,
                 extend_factor=(1., 1.),
                 finest_scale=56):
        super(OBBScaleRoIExtractor, self).__init__()
        self.roi_layer = roi_layer
        self.featmap_strides = featmap_strides
        self.img_strides = [1]
        self.out_size = out_size
        self.roi_layers = nn.ModuleList()
        for i in range(len(self.out_size)):
            self.roi_layer.out_size = out_size[i]
            roi_layer = self.build_roi_layers(self.roi_layer, [self.featmap_strides[i]])
            self.roi_layers.append(roi_layer)
        self.roi_layer.out_size = clip_input_size
        self.img_roi_layers = self.build_roi_layers(self.roi_layer, self.img_strides)
        self.fp16_enabled = False
        self.extend_factor = extend_factor
        self.finest_scale = finest_scale

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmdet/ops`` such as
                ``RoIAlign``.
            featmap_strides (int): The stride of input feature map w.r.t to the
                original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        # roi_layers = nn.ModuleList(
        #     [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        roi_layers = layer_cls(spatial_scale=1 / featmap_strides[0], **cfg)
        return roi_layers

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 6)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """
        if scale_factor is None:
            return rois
        h_scale_factor, w_scale_factor = _pair(scale_factor)
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        return new_rois

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function"""
        num_levels = len(feats)
        roi_feats_list = []
        rois = self.roi_rescale(rois, self.extend_factor)
        rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            roi_layer = self.roi_layers[i]
            roi_feats_t = roi_layer(feats[i], rois)
            roi_feats = roi_feats_t
            roi_feats_list.append(roi_feats)

        return roi_feats_list

    def forward_img(self, imgs, rois, roi_scale_factor=None):
        """Forward img function for instance level img"""
        rois = self.roi_rescale(rois, self.extend_factor)
        rois = self.roi_rescale(rois, roi_scale_factor)
        roi_imgs = self.img_roi_layers(imgs, rois)

        return roi_imgs
