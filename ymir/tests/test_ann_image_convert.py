"""
read annotation image and convert, test in ymir docker enviroment
"""

from PIL import Image
from PIL.ImagePalette import ImagePalette
from ymir_exc.util import get_merged_config
import numpy as np
from typing import Dict, List, Tuple, Union, Any
from timer import timer
from numba import njit

# def rgb2label_id(rgb_ann: Union[Image.Image, str], color_maps: List[Tuple]) -> Image.Image:
#     if isinstance(rgb_ann, str):
#         pil_rgb_ann = Image.Open(rgb_ann)
#     else:
#         pil_rgb_ann = rgb_ann

#     mode = pil_rgb_ann.mode
#     palette = []
#     for rgb in color_maps:
#         palette.extend(list(rgb))

#     img_palette = ImagePalette(mode, palette, size=len(color_maps) * 3)

#     label_id_ann = pil_rgb_ann.convert(
#         mode='P',
#         palette=img_palette,
#     )

#     return label_id_ann


def mask_convert(np_rgb_ann, color_maps_list, dtype=np.uint8):
    """
    map color to label id, start from 1, ignore color map is 0.
    """
    height, width = np_rgb_ann.shape[0:2]
    np_label_id = np.zeros(shape=(height, width), dtype=dtype)

    for idx, rgb in enumerate(color_maps_list):
        r = np_rgb_ann[:, :, 0] == rgb[0]
        g = np_rgb_ann[:, :, 1] == rgb[1]
        b = np_rgb_ann[:, :, 2] == rgb[2]

        np_label_id[r & g & b] = idx + 1
    return np_label_id


def ymir_convert(pil_rgb_ann, color_maps_dict, dtype=np.uint8):
    """
    use pillow getpixel() methods to convert image
    """
    width, height = pil_rgb_ann.size
    np_label_id = np.zeros(shape=(height, width), dtype=dtype)

    for x in range(width):
        for y in range(height):
            r, g, b = pil_rgb_ann.getpixel((x, y))
            np_label_id[y, x] = color_maps_dict[(r, g, b)]

    return np_label_id


def violence_convert(np_rgb_ann, color_maps_dict, dtype=np.uint8):
    height, width = np_rgb_ann.shape[0:2]
    np_label_id = np.zeros(shape=(height, width), dtype=dtype)

    for x in range(width):
        for y in range(height):
            np_label_id[y, x] = color_maps_dict[tuple(np_rgb_ann[y, x, :])]

    return np_label_id


def rgb2label_id(rgb_ann: Union[Image.Image, str],
                 color_maps: List[Tuple],
                 mode: str = 'mask',
                 dtype: Any = np.uint8) -> np.ndarray:
    if isinstance(rgb_ann, str):
        pil_rgb_ann = Image.Open(rgb_ann)
    else:
        pil_rgb_ann = rgb_ann

    np_rgb_ann = np.array(pil_rgb_ann)
    height, width = np_rgb_ann.shape[0:2]
    np_label_id = np.zeros(shape=(height, width), dtype=np.uint)

    color_maps_dict = {}
    for idx, rgb in enumerate(color_maps):
        color_maps_dict[rgb] = idx

    color_maps_dict[(0, 0, 0)] = 255

    if mode == 'mask':
        np_label_id = mask_convert(np_rgb_ann, color_maps, dtype)
    elif mode == 'ymir':
        np_label_id = ymir_convert(pil_rgb_ann, color_maps_dict, dtype)
    else:
        np_label_id = violence_convert(np_rgb_ann, color_maps_dict, dtype)

    # Image.fromarray(np_label_id, mode='I')
    return np_label_id


if __name__ == '__main__':
    cfg = get_merged_config()

    with open(cfg.ymir.input.training_index_file, 'r') as fp:
        lines = fp.readlines()

    img_path, ann_path = lines[0].split()
    pil_ann_img = Image.open(ann_path)
    print(ann_path, pil_ann_img.mode)

    np_ann_img = np.array(pil_ann_img)

    print(np.unique(np_ann_img[:, :, 0]))

    label_map_file = '/in/annotations/labelmap.txt'
    with open(label_map_file, 'r') as fp:
        label_map_lines = fp.readlines()

    color_maps: List[Tuple] = []
    for idx, line in enumerate(label_map_lines):
        label, rgb = line.split(':')[0:2]
        r, g, b = [int(x) for x in rgb.split(',')]
        color_maps.append((r, g, b))

    print(color_maps)

    for dtype in [np.uint8, np.int32]:
        for mode in ['mask', 'ymir', 'violence']:
            with timer() as t:
                np_ann = rgb2label_id(pil_ann_img, color_maps, mode, dtype)
                print(f'{mode} convert: {t.elapse} s')
            print('before: ', np.count_nonzero(np_ann))
            print(np.unique(np_ann))
            pil_ann = Image.fromarray(np_ann, mode='I' if dtype == np.int32 else 'L')
            x_np_ann = np.array(pil_ann, dtype=dtype)
            print('after: ', np.count_nonzero(x_np_ann))
            print(np.unique(x_np_ann))
