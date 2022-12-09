import logging
import os
import os.path as osp
import sys
from typing import Dict, List

import cv2
import mmcv
import numpy as np
import torch
from easydict import EasyDict as edict
from mmcv.engine import collect_results_cpu
from mmcv.runner import wrap_fp16_model
from PIL import Image
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, get_bool, get_merged_config,
                           get_weight_files, write_ymir_monitor_process)

from mmseg.apis import init_segmentor
from ymir.tools.batch_infer import get_dataloader
from ymir.tools.superpixel import get_superpixel
from ymir.ymir_util import get_best_weight_file

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_uncertainty_info(cfg: edict, result: torch.Tensor, superpixel: np.ndarray) -> List[Dict]:
    """
    input:
        result: prediction result with softmax propability for each class
        superpixel: superpixel labels

    return:
        unc_list: uncertainty information for each superpixel region
            superpixel_idx: index
            score: uncertainty score
            class_id: class id for each superpixel region if need class balance
    """
    uncertainty_method: str = cfg.param.uncertainty_method
    need_class_balance: bool = get_bool(cfg, 'class_balance', False)

    c, h, w = result.shape
    sph, spw = superpixel.shape

    assert h == sph and w == spw, f'result shape {h, w} != superpixel shape {sph, spw}'

    if uncertainty_method == 'BvSB':
        # the bigger the better
        if c == 1:
            uncertainty = 1 - torch.abs(result - 0.5)
        else:
            # torch.sort is faster than torch.topk when c is small (<100)
            sorted_result, _ = torch.sort(result, dim=0, descending=True)
            uncertainty = sorted_result[1, :, :] / sorted_result[0, :, :]
    else:
        raise Exception(f'unknown uncertainty method {uncertainty_method}')

    np_unc = uncertainty.data.cpu().numpy()
    Max = superpixel.max()
    if need_class_balance:
        pred = torch.argmax(result, dim=0).data.cpu().numpy()

    unc_list: List[Dict] = []
    for idx in range(Max):
        y, x = np.where(superpixel == idx)
        score = float(np.mean(np_unc[y, x]))

        d = dict(superpixel_idx=idx, score=score)
        if need_class_balance:
            unique, unique_counts = np.unique(pred[y, x], return_counts=True)
            class_id = unique[np.argmax(unique_counts)]
            d['class_id'] = class_id

        unc_list.append(d)
    return unc_list


def iter_fun(cfg, model, idx, batch, N, monitor_gap):
    """
    batch: Dict(img=[tensor,], img_metas=[xxx,])

    return: image filename list and correspond topk scores
    """
    # result = inference_segmentor(model, image_filename)[0]
    batch_img = batch['img'][0]
    batch_meta = batch['img_metas'][0].data[0]
    device = torch.device('cuda:0' if RANK == -1 else f'cuda:{RANK}')
    batch_result = model.inference(batch_img.to(device), batch_meta, rescale=True)

    if idx % monitor_gap == 0 and RANK in [0, -1]:
        write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    results = []
    save_dir = cfg.ymir.output.root_dir
    os.makedirs(osp.join(save_dir, 'superpixel'), exist_ok=True)
    max_superpixel_per_image = int(cfg.param.max_superpixel_per_image)
    for idx, meta in enumerate(batch_meta):
        superpixel = get_superpixel(cfg, img=meta['filename'], max_superpixels=max_superpixel_per_image)
        unc_list = get_uncertainty_info(cfg, batch_result[idx], superpixel)

        # cannot use cv2.imwrite to save superpixel
        pil_img = Image.fromarray(superpixel, mode='I')
        superpixel_filepath = osp.join(save_dir, 'superpixel', osp.basename(meta['filename']))
        pil_img.save(superpixel_filepath)

        results.append(dict(image_filepath=meta['filename'], superpixel_filepath=superpixel_filepath,
                            unc_list=unc_list))
    return results


def update_image_scores(region_scores: List[Dict], threshold: float, topk: int):
    for dict_per_img in tqdm(region_scores):
        unc_list = sorted(dict_per_img['unc_list'], key=lambda a: a['score'], reverse=True)
        topk_scores = []
        image_score = 0
        for idx, d in enumerate(unc_list):
            if idx < topk:
                # image scores = sum of topk region scores
                image_score += d['score']
                topk_scores.append(d)
            elif d['score'] >= threshold:
                # allow label more than topk region better than threshold
                image_score += d['score']
                topk_scores.append(d)
            else:
                break

        dict_per_img['image_score'] = image_score
        dict_per_img['topk_scores'] = topk_scores


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
    N = len(dataloader)
    if N == 0:
        raise Exception('find empty dataloader')

    if RANK in [0, -1]:
        tbar = tqdm(dataloader)
    else:
        tbar = dataloader

    monitor_gap = max(1, N // 1000)

    rank_region_result = []
    for idx, batch in enumerate(tbar):
        batch_region_result = iter_fun(ymir_cfg, model, idx, batch, N, monitor_gap)
        rank_region_result.extend(batch_region_result)

    index_file_path = ymir_cfg.ymir.input.candidate_index_file
    with open(index_file_path, 'r') as f:
        num_image_all_rank = len(f.readlines())

    if WORLD_SIZE == 1:
        all_region_result = rank_region_result
    else:
        tmp_dir = osp.join(ymir_cfg.ymir.output.root_dir, 'tmp_dir')
        all_region_result = collect_results_cpu(rank_region_result, num_image_all_rank, tmp_dir)

    if RANK in [0, -1]:
        # note we remove normalization here
        need_class_balance: bool = get_bool(ymir_cfg, 'class_balance', False)
        if need_class_balance:
            class_num = len(ymir_cfg.param.class_names)
            region_num_per_class = np.zeros(shape=(class_num), dtype=np.float32)
            for unc_info in tqdm(all_region_result):
                for d in unc_info['unc_list']:
                    region_num_per_class[d['class_id']] += 1

            total_region_num = np.sum(region_num_per_class)
            class_weights = np.exp(-region_num_per_class / total_region_num)

        all_region_scores = []
        for img_idx, unc_info in enumerate(tqdm(all_region_result, desc='gather all region scores')):
            for d in unc_info['unc_list']:
                if need_class_balance:
                    d['score'] = d['score'] * class_weights[d['class_id']]
                all_region_scores.append(d['score'])

        # 80% scores <= threshold
        max_kept_mining_image = int(ymir_cfg.param.max_kept_mining_image)
        percent = round(100 * max(0.8, 1 - max_kept_mining_image / num_image_all_rank))
        threshold = np.percentile(all_region_scores, percent)
        logging.info(f'region scores: len={len(all_region_scores)}, percentile-{percent}={threshold}')
        logging.info(f'region scores: max={np.max(all_region_scores)}, min={np.min(all_region_scores)}')

        # add image_score
        topk = int(ymir_cfg.param.topk_superpixel_score)
        update_image_scores(all_region_result, threshold, topk)

        all_image_result = sorted(all_region_result, key=lambda x: x['image_score'], reverse=True)

        # create mask to label and remove superpixel file
        os.makedirs(osp.join(ymir_cfg.ymir.output.root_dir, 'masks'), exist_ok=True)
        ymir_mining_result = []
        for img_idx in range(num_image_all_rank):
            if img_idx < max_kept_mining_image:
                superpixel = np.array(Image.open(all_image_result[img_idx]['superpixel_filepath']))
                topk_index = [d['superpixel_idx'] for d in all_image_result[img_idx]['topk_scores']]
                mask_to_label = np.zeros(superpixel.shape, dtype=np.bool8)
                for superpixel_idx in topk_index:
                    mask_to_label = np.logical_or(mask_to_label, superpixel == superpixel_idx)

                mask_name = osp.join(ymir_cfg.ymir.output.root_dir, 'masks',
                                     osp.basename(all_image_result[img_idx]['image_filepath']))
                cv2.imwrite(mask_name, 255 * mask_to_label.astype(np.uint8))

            ymir_mining_result.append(
                (all_image_result[img_idx]['image_filepath'], all_image_result[img_idx]['image_score']))
            os.remove(all_image_result[img_idx]['superpixel_filepath'])

        rw.write_mining_result(mining_result=ymir_mining_result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
