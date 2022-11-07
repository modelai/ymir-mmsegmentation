import numpy as np
from PIL import Image
from typing import Dict, Tuple

from ymir_exc.util import get_merged_config


def count_ann_image(np_rgb_ann: np.ndarray, color_maps: Dict[str, Tuple[int, int, int]]):
    """
    map color to label id, start from 1, ignore color map is 0.
    """
    for label, rgb in color_maps.items():
        r = (np_rgb_ann[:, :, 0] == rgb[0])
        g = (np_rgb_ann[:, :, 1] == rgb[1])
        b = (np_rgb_ann[:, :, 2] == rgb[2])

        print(label, np.count_nonzero(r & g & b))


def main():
    cfg = get_merged_config()

    with open(cfg.ymir.input.training_index_file, 'r') as fp:
        lines = fp.readlines()

    label_map_file = '/in/annotations/labelmap.txt'
    with open(label_map_file, 'r') as fp:
        label_map_lines = fp.readlines()

    color_maps: Dict[str, Tuple[int, int, int]] = {}
    for idx, line in enumerate(label_map_lines):
        label, rgb = line.split(':')[0:2]
        r, g, b = [int(x) for x in rgb.split(',')]
        color_maps[label] = (r, g, b)

    for line in lines:
        img_path, ann_path = line.split()
        pil_ann_img = Image.open(ann_path)
        np_ann_img = np.array(pil_ann_img)
        print(ann_path)
        count_ann_image(np_ann_img, color_maps)


if __name__ == '__main__':
    main()
