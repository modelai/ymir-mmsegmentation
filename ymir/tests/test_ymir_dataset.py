from mmcv.utils import Config
from ymir_exc.util import get_merged_config
import numpy as np

from mmseg.datasets import build_dataset
from ymir.ymir_util import modify_mmcv_config
from tqdm import tqdm

if __name__ == '__main__':
    ymir_cfg = get_merged_config()

    mmcv_cfg = Config.fromfile(ymir_cfg.param.config_file)
    modify_mmcv_config(ymir_cfg, mmcv_cfg)

    class_names = ymir_cfg.param.class_names
    num_classes = len(ymir_cfg.param.class_names)

    print('class_name: ', class_names)
    train_dataset = build_dataset(mmcv_cfg.data.train)
    # val_dataset = build_dataset(mmcv_cfg.data.val)

    for dataset in [train_dataset]:
        count = {cls: 0 for cls in class_names}
        for img_id, d in enumerate(tqdm(dataset)):
            class_ids = d['gt_semantic_seg'].data.unique()
            # print(d['img_metas'])
            # print(f'class_ids is {class_ids}')
            # print(f'image {img_id}: ' + '*' * 50)
            for idx in class_ids:
                if idx < 255:
                    assert idx <= num_classes, f'{idx} should <= {num_classes}'
                    # print(idx, class_names[idx - 1], np.count_nonzero(d['gt_semantic_seg'].data == idx))

                    count[class_names[idx]] += 1

            # if img_id >= 10:
            #     break
        print(count)
