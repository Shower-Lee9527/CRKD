from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def main():
    parser = ArgumentParser()
    parser.add_argument('--img',
                        # default='./experiment-img/distillation examples/P0865_0009.png',
                        default='/mnt/disk/Detection-TGARS/OBBDetection-master/data/split_ss_dota1_5/trainval/images/P0251_0004.png',
                        help='Image file')
    parser.add_argument('--config',
                        # default='../work_dirs/dota15/dis_clip_orcnn_r50_fpn_1x_dota15_6742/dis_clip_orcnn_r50_fpn_1x_dota15.py',
                        default='../work_dirs/dota15/roi_transformer/faster_rcnn_roitrans_r50_fpn_1x_dota15.py',
                        # default='../work_dirs/dota15/dis_clip_gliding_vertex/dis_clip_gliding_vertex_r50_fpn_1x_dota15.py',
                        # default='../work_dirs/dota15/orcnn_r50_fpn_1x_dota15/faster_rcnn_orpn_r50_fpn_1x_dota15.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        # default='../work_dirs/dota15/dis_clip_orcnn_r50_fpn_1x_dota15_6742/epoch_12.pth',
                        default='../work_dirs/dota15/roi_transformer/epoch_12.pth',
                        # default='../work_dirs/dota15/dis_clip_gliding_vertex/epoch_12.pth',
                        # default='../work_dirs/dota15/orcnn_r50_fpn_1x_dota15/epoch_12.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
