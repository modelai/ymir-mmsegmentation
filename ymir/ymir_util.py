import glob
import logging
import os
import os.path as osp
import warnings
from typing import Any, Iterable, List

from easydict import EasyDict as edict
from mmcv.utils import Config
from ymir_exc.util import get_weight_files


def _find_any(str1: str, sub_strs: List[str]) -> bool:
    for s in sub_strs:
        if str1.find(s) > -1:
            return True
    return False


def get_best_weight_file(ymir_cfg: edict):
    """
    find the best weight file for ymir-executor
    1. find best_* in /in/models
    2. find epoch_* or iter_* in /in/models
    3. find xxx.pth or xxx.pt in /weights
    """
    weight_files = get_weight_files(ymir_cfg)

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    best_pth_files = [f for f in weight_files if osp.basename(f).startswith('best_')]
    if len(best_pth_files) > 0:
        return max(best_pth_files, key=os.path.getctime)

    epoch_pth_files = [f for f in weight_files if osp.basename(f).startswith(('epoch_', 'iter_'))]
    if len(epoch_pth_files) > 0:
        return max(epoch_pth_files, key=os.path.getctime)

    if ymir_cfg.ymir.run_training:
        model_name_splits = osp.basename(ymir_cfg.param.config_file).split('_')
        model_name = model_name_splits[0]
        weight_files = [
            f for f in glob.glob(f'/weights/**/{model_name}*', recursive=True) if f.endswith(('.pth', '.pt'))
        ]

        # eg: config_file = configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py
        # eg: best_model_file = fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth

        if len(weight_files) > 0:
            best_model_file = weight_files[0]
            for idx in range(2, len(model_name_splits)):
                prefix = '_'.join(model_name_splits[0:idx + 1])
                weight_files = [f for f in weight_files if osp.basename(f).startswith(prefix)]
                if len(weight_files) > 1:
                    best_model_file = weight_files[0]
                elif len(weight_files) == 1:
                    return weight_files[0]
                else:
                    return best_model_file

            return best_model_file

    return ""


def modify_mmcv_config(ymir_cfg: edict, mmcv_cfg: Config) -> None:
    """
    useful for training process
    - modify dataset config
    - modify model output channel
    - modify epochs, checkpoint, tensorboard config
    """
    def recursive_modify(mmcv_cfg: Config, attribute_key: str, attribute_value: Any):
        for key in mmcv_cfg:
            if key == attribute_key:
                mmcv_cfg[key] = attribute_value
            elif isinstance(mmcv_cfg[key], Config):
                recursive_modify(mmcv_cfg[key], attribute_key, attribute_value)
            elif isinstance(mmcv_cfg[key], Iterable):
                for cfg in mmcv_cfg[key]:
                    if isinstance(cfg, Config):
                        recursive_modify(cfg, attribute_key, attribute_value)

    # modify dataset config
    ymir_ann_files = dict(train=ymir_cfg.ymir.input.training_index_file,
                          val=ymir_cfg.ymir.input.val_index_file,
                          test=ymir_cfg.ymir.input.candidate_index_file)

    # validation may augment the image and use more gpu
    # so set smaller samples_per_gpu for validation
    samples_per_gpu = ymir_cfg.param.samples_per_gpu
    workers_per_gpu = ymir_cfg.param.workers_per_gpu
    mmcv_cfg.data.samples_per_gpu = samples_per_gpu
    mmcv_cfg.data.workers_per_gpu = workers_per_gpu

    num_classes = len(ymir_cfg.param.class_names)
    recursive_modify(mmcv_cfg.model, 'num_classes', num_classes)

    for split in ['train', 'val', 'test']:
        ymir_dataset_cfg = dict(type='YmirDataset',
                                split=ymir_ann_files[split],
                                img_suffix='.png',
                                seg_map_suffix='.png',
                                img_dir=ymir_cfg.ymir.input.assets_dir,
                                ann_dir=ymir_cfg.ymir.input.annotations_dir,
                                classes=ymir_cfg.param.class_names,
                                palette=ymir_cfg.param.get('palette', None),
                                data_root=ymir_cfg.ymir.input.root_dir)
        # modify dataset config for `split`
        if split not in mmcv_cfg.data:
            continue

        mmcv_dataset_cfg = mmcv_cfg.data.get(split)

        if isinstance(mmcv_dataset_cfg, (list, tuple)):
            for x in mmcv_dataset_cfg:
                x.update(ymir_dataset_cfg)
        else:
            src_dataset_type = mmcv_dataset_cfg.type
            if src_dataset_type in ['MultiImageMixDataset', 'RepeatDataset']:
                mmcv_dataset_cfg.dataset.update(ymir_dataset_cfg)
            elif src_dataset_type in ['ConcatDataset']:
                for d in mmcv_dataset_cfg.datasets:
                    d.update(ymir_dataset_cfg)
            else:
                mmcv_dataset_cfg.update(ymir_dataset_cfg)

    # modify epochs/iters, checkpoint, tensorboard config
    # if 'max_epochs' in ymir_cfg.param:
    #     max_epochs = ymir_cfg.param.max_epochs
    #     if max_epochs <= 0:
    #         pass
    #     # modify EpochBasedRunner-like runner
    #     elif 'max_epochs' in mmcv_cfg.runner:
    #         mmcv_cfg.runner.max_epochs = max_epochs
    #     else:
    #         # convert other type to EpochBasedRunner
    #         epoch_runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
    #         warnings.warn(f'modify {mmcv_cfg.runner} to {epoch_runner}')
    #         mmcv_cfg.runner = epoch_runner

    if 'max_iters' in ymir_cfg.param:
        max_iters = ymir_cfg.param.max_iters
        if max_iters <= 0:
            pass
        elif 'max_iters' in mmcv_cfg.runner:
            mmcv_cfg.runner.max_iters = max_iters
        else:
            iter_runner = dict(type='IterBasedRunner', max_iters=max_iters)
            warnings.warn(f'modify {mmcv_cfg.runner} to {iter_runner}')
            mmcv_cfg.runner = iter_runner

    mmcv_cfg.checkpoint_config['out_dir'] = ymir_cfg.ymir.output.models_dir
    tensorboard_logger = dict(type='TensorboardLoggerHook', log_dir=ymir_cfg.ymir.output.tensorboard_dir)
    if len(mmcv_cfg.log_config['hooks']) <= 1:
        mmcv_cfg.log_config['hooks'].append(tensorboard_logger)
    else:
        mmcv_cfg.log_config['hooks'][1].update(tensorboard_logger)

    if 'interval' in ymir_cfg.param:
        interval = int(ymir_cfg.param.interval)
        if interval > 0:
            mmcv_cfg.evaluation.interval = min(interval, mmcv_cfg.runner.max_iters // 10)
    else:
        if 'max_iters' in mmcv_cfg.runner:
            interval = max(1, mmcv_cfg.runner.max_iters // 10)
        elif 'max_epoch' in mmcv_cfg.runner:
            interval = max(1, mmcv_cfg.runner.max_epochs // 10)
        else:
            assert False, f'unknown runner {mmcv_cfg.runner}'
        # modify evaluation and interval

        mmcv_cfg.evaluation.interval = interval

    mmcv_cfg.checkpoint_config.interval = mmcv_cfg.evaluation.interval

    # fix DDP error
    mmcv_cfg.find_unused_parameters = True

    # set work dir
    mmcv_cfg.work_dir = ymir_cfg.ymir.output.models_dir

    # auto load offered weight file if not set by user!
    # maybe overwrite the default `load_from` from config file
    args_options = ymir_cfg.param.get("args_options", '')
    cfg_options = ymir_cfg.param.get("cfg_options", '')

    # if mmcv_cfg.load_from is None and mmcv_cfg.resume_from is None:
    if not (_find_any(args_options, ['--load-from', '--resume-from'])
            or _find_any(cfg_options, ['load_from', 'resume_from'])):  # noqa: W503
        weight_file = get_best_weight_file(ymir_cfg)
        if weight_file:
            mmcv_cfg.load_from = weight_file
        else:
            logging.warning('no weight file used for training!')
