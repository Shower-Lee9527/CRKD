from mmdet.models.builder import DETECTORS
import torch.nn as nn
import torch
from mmdet.models.detectors.obb.obb_two_stage import OBBTwoStageDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmengine.visualization import Visualizer

@DETECTORS.register_module()
class DisCLIPFasterRCNNOBB(OBBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 distillation,
                 scale_head,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(DisCLIPFasterRCNNOBB, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.backbone = build_backbone(backbone)
        self.distillation = build_backbone(distillation)

        if scale_head is not None:
            self.scale_head = build_head(scale_head)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_scale_head(self):
        return hasattr(self, 'scale_head') and self.scale_head is not None

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBTwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_scale_head:
            self.scale_head.init_weights()
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
        drawn_img0 = visualizer.draw_featmap(x[0][:,163,:,:], channel_reduction='select_max')
        visualizer.add_image('x', drawn_img0)
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
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        proposal_type = 'hbb'
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )

        if proposal_type == 'hbb':
            proposals = torch.randn(1000, 4).to(img.device)
        elif proposal_type == 'obb':
            proposals = torch.randn(1000, 5).to(img.device)
        else:
            # poly proposals need to be generated in roi_head
            proposals = None
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x_backbone = self.extract_feat_backbone(img)
        x = self.extract_feat_neck(x_backbone)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                    else gt_obboxes_ignore

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                target_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ScaleHead forward and loss
        if self.with_scale_head:
            proposal_type = 'obb'
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                    else gt_obboxes_ignore

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            sca_losses, bbox_feats_list, bbox_targets, bbox_imgs = self.scale_head.forward_train(
                img,
                x_backbone,
                x,
                img_metas,
                target_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(sca_losses)

        # Distillation forward and loss
        # obj_losses, con_losses, sem_losses = self.distillation.forward(bbox_feats_list, bbox_imgs, bbox_targets)
        con_losses, sem_losses = self.distillation.forward(bbox_feats_list, bbox_targets, bbox_imgs)

        # losses.update(dict(obj_losses=obj_losses))
        losses.update(dict(con_losses=con_losses))
        losses.update(dict(sem_losses=sem_losses))

        # ROIHead forward and loss
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_obboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_obboxes_ignore,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rotate_aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
