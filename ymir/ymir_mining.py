import logging
import os
import os.path as osp
import sys
from functools import partial
from typing import Dict, List, Tuple

import cv2
import mmcv
import numpy as np
from easydict import EasyDict as edict
from mmcv.runner import wrap_fp16_model
from tqdm import tqdm
from ymir_exc.util import (YmirStage, get_merged_config, get_weight_files, write_ymir_monitor_process, get_bool)

from mmseg.apis import inference_segmentor, init_segmentor
from ymir.tools.result_to_coco import convert
from ymir.tools.superpixel import get_superpixel
from ymir.ymir_dist import run_dist
from ymir.ymir_util import get_best_weight_file, get_palette
from ymir.tools.batch_infer import get_dataloader

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_uncertainty(cfg: edict, result: np.ndarray):
    uncertainty = cfg.param.uncertainty_method

    if uncertainty == 'BvSB':
        pass
    else:
        raise Exception(f'unknown uncertainty method {uncertainty}')


def iter_fun(cfg, model, idx, image_filename, N, monitor_gap):
    result = inference_segmentor(model, image_filename)[0]

    if idx % monitor_gap == 0 and RANK in [0, -1]:
        write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    image = cv2.imread(image_filename)
    max_superpixel_per_image = cfg.param.max_superpixel_per_image
    superpixel = get_superpixel(cfg, image, max_superpixel=max_superpixel_per_image)
    uncertainty = get_uncertainty(cfg, result)
    # view mmseg.models.segmentors.base show_result()
    return dict(image=image, result=result[0])


def main() -> int:
    ymir_cfg: edict = get_merged_config()
    config_files = get_weight_files(ymir_cfg, suffix=('.py'))
    if len(config_files) == 0:
        raise Exception('not found config file (xxx.py) in pretrained weight files')
    elif len(config_files) > 1:
        raise Exception(f'found multiple config files {config_files} in pretrained weight files')

    checkpoint_file = get_best_weight_file(ymir_cfg)
    if not checkpoint_file:
        raise Exception('not found pretrain weight file (*.pt or *.pth)')

    mmcv_cfg = mmcv.Config.fromfile(config_files[0])
    mmcv_cfg.model.train_cfg = None
    model = init_segmentor(config=mmcv_cfg, checkpoint=checkpoint_file, device='cuda:0')

    if get_bool(ymir_cfg, 'fp16', False):
        wrap_fp16_model(model)

    dataloader = get_dataloader(mmcv_cfg, ymir_cfg)
    return 0
