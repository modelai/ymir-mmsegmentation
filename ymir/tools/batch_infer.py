from pprint import pprint

from mmcv.runner import init_dist

from mmseg.datasets import build_dataloader, build_dataset


def get_dataloader(mmcv_cfg, ymir_cfg):
    if ymir_cfg.param.get('aug_test', False):
        mmcv_cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        mmcv_cfg.data.test.pipeline[1].flip = True

    mmcv_cfg.model.pretrained = None
    mmcv_cfg.data.test.test_mode = True
    mmcv_cfg.data.test.split = ymir_cfg.ymir.input.candidate_index_file

    gpu_id: str = str(ymir_cfg.param.get('gpu_id', '0'))
    gpu_count: int = len(gpu_id.split(','))

    if gpu_count > 1:
        dist = True
        mmcv_cfg.gpu_ids = [int(x) for x in gpu_id.split(',')]
        init_dist('pytorch', **mmcv_cfg.dist_params)
    else:
        dist = False
        mmcv_cfg.gpu_ids = [int(x) for x in gpu_id.split(',')]

    dataset = build_dataset(mmcv_cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(mmcv_cfg.gpu_ids),
        dist=dist,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in mmcv_cfg.data.items()
        if k not in ['train', 'val', 'test', 'train_dataloader', 'val_dataloader', 'test_dataloader']
    })

    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **mmcv_cfg.data.get('test_dataloader', {})
    }

    samples_per_gpu = ymir_cfg.param.samples_per_gpu
    workers_per_gpu = ymir_cfg.param.workers_per_gpu
    test_loader_cfg['samples_per_gpu'] = samples_per_gpu
    test_loader_cfg['workers_per_gpu'] = workers_per_gpu
    # build the dataloader
    print('test_loader_cfg: ')
    pprint(test_loader_cfg)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    return data_loader
