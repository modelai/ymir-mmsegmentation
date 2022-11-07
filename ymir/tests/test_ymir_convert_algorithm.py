import os
from PIL import Image
from typing import Set, Dict, Tuple
import numpy as np
from timer import timer


def get_labelmap(labelmap_file='ymir/tests/labelmap.txt') -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    # build color map, map all unknown classes to background (0, 0, 0).
    map_color_pixel: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {(0, 0, 0): (0, 0, 0)}

    with open(labelmap_file, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        label, rgb = line.split(':')[0:2]
        r, g, b = [int(x) for x in rgb.split(',')]
        map_color_pixel[(r, g, b)] = (r, g, b)

    return map_color_pixel


class Converter(object):

    def __init__(self):
        self.map_color_pixel = get_labelmap()
        self.map_color_cid: Dict[Tuple[int, int, int], int] = {}
        for idx, color in enumerate(self.map_color_pixel):
            self.map_color_cid[color] = idx

    def ymir_convert(self, mask_image):
        width, height = mask_image.size
        unexpected_color: Set[Tuple[int, int, int]] = set()
        img_class_ids: Set[int] = set()

        for x in range(width):
            for y in range(height):
                color = mask_image.getpixel((x, y))
                if color in self.map_color_pixel:
                    unexpected_color.add(color)

                # map_color_cid (known class names) is subset of map_color_pixel (including known/unknown).
                if color in self.map_color_cid:
                    img_class_ids.add(self.map_color_cid[color])
                elif color != (0, 0, 0):  # map unknown color to (0,0,0).
                    mask_image.putpixel((x, y), (0, 0, 0))

        print(unexpected_color, img_class_ids)

        return np.array(mask_image)

    def mask_convert(self, mask_image):
        width, height = mask_image.size
        img = np.array(mask_image)

        unexpected_color: Set[Tuple[int, int, int]] = set()
        img_class_ids: Set[int] = set()

        new_mask = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        for color in self.map_color_pixel:
            r = img[:, :, 0] == color[0]
            g = img[:, :, 1] == color[1]
            b = img[:, :, 2] == color[2]

            mask = r & g & b
            if not np.any(mask):
                continue

            if color in self.map_color_pixel:
                unexpected_color.add(color)

            # map_color_cid (known class names) is subset of map_color_pixel (including known/unknown).
            if color in self.map_color_cid:
                img_class_ids.add(self.map_color_cid[color])
                new_mask[mask] = color
            elif color != (0, 0, 0):  # map unknown color to (0,0,0).
                pass

        print(unexpected_color, img_class_ids)
        return new_mask


def main():
    c = Converter()
    img_file = 'ymir/tests/lindau_000032_000019_leftImg8bit.png'
    mask_image = Image.open(img_file)
    mask_image = mask_image.convert('RGB')

    with timer() as t:
        ymir_mask = c.ymir_convert(mask_image)
        print(f'convert: {t.elapse} s')

    with timer() as t:
        np_mask = c.mask_convert(mask_image)
        print(f'convert: {t.elapse} s')

    if np.all(np_mask == ymir_mask):
        print('all the same')
    else:
        print('not all same')


if __name__ == '__main__':
    main()
