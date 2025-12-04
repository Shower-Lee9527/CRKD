from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.obb.obb_single_stage import OBBSingleStageDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.core import arb2result

import torch.nn as nn


@DETECTORS.register_module()
class DisCLIPS2ANet(OBBSingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 distillation,
                 scale_head,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DisCLIPS2ANet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
        self.backbone = build_backbone(backbone)
        self.distillation = build_backbone(distillation)
        if scale_head is not None:
            self.scale_head = build_head(scale_head)

        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_scale_head(self):
        return hasattr(self, 'scale_head') and self.scale_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_scale_head:
            self.scale_head.init_weights()
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_backbone(self, img):
        """Directly extract features from the backbone
        """
        x = self.backbone(img)
        return x

    def extract_feat_neck(self, x):
        """Directly extract features from the neck
        """
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x_backbone = self.extract_feat_backbone(img)
        x = self.extract_feat_neck(x_backbone)

        losses = dict()

        # # RPN forward and loss
        # if self.with_rpn:
        #     proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
        #     target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
        #     target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
        #         else gt_obboxes_ignore
        #
        #     proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                       self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         target_bboxes,
        #         gt_labels=None,
        #         gt_bboxes_ignore=target_bboxes_ignore,
        #         proposal_cfg=proposal_cfg)
        #     losses.update(rpn_losses)
        # else:
        #     proposal_list = proposals

        # ScaleHead forward and loss
        if self.with_scale_head:
            proposal_type = 'obb'
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                else gt_obboxes_ignore

            proposal_cfg = self.train_cfg[0].get('rpn_proposal')
            sca_losses, bbox_feats_list, bbox_targets, bbox_imgs = self.scale_head.forward_train(
                img,
                x_backbone,
                x,
                img_metas,
                gt_obboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_obboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(sca_losses)

        # Distillation forward and loss
        # obj_losses, con_losses, sem_losses = self.distillation.forward(bbox_feats_list, bbox_imgs, bbox_targets)
        con_losses, sem_losses = self.distillation.forward(bbox_feats_list, bbox_targets, bbox_imgs)

        # losses.update(dict(obj_losses=obj_losses))
        losses.update(dict(con_losses=con_losses))
        losses.update(dict(sem_losses=sem_losses))

        # BBoxHead forward and loss
        bbox_losses = self.bbox_head.forward_train(x, img_metas, gt_obboxes,
                                              gt_labels, gt_obboxes_ignore)
        losses.update(bbox_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_type = getattr(self.bbox_head, 'bbox_type', 'hbb')
        bbox_results = [
            arb2result(det_bboxes, det_labels, self.bbox_head.num_classes, bbox_type)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation"""
        raise NotImplementedError