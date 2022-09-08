from mmcv.utils import Config
from ymir_exc.util import get_merged_config

from mmseg.datasets import build_dataset
from ymir.ymir_util import modify_mmcv_config

if __name__ == '__main__':
    ymir_cfg = get_merged_config()

    mmcv_cfg = Config.fromfile(ymir_cfg.param.config_file)
    modify_mmcv_config(ymir_cfg, mmcv_cfg)

    train_dataset = build_dataset(mmcv_cfg.data.train)
    for d in train_dataset:
        class_ids = d['gt_semantic_seg'].data.unique()
        break

    num_classes = len(ymir_cfg.param.class_names)
    for idx in class_ids:
        if idx < 255:
            assert idx < num_classes
