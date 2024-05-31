from mmdet.apis import DetInferencer
inferencer = DetInferencer(
    weights='work_dirs/faster-rcnn_r50_fpn_1x_voc/epoch_10.pth',
    device='cuda:0'
)
inferencer('demo/in', out_dir='outputs/in', no_save_pred=False)