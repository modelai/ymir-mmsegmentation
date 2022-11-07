import logging
import os
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc.util import find_free_port, get_merged_config

from ymir.ymir_util import convert_annotation_dataset, write_last_ymir_result_file


def main() -> int:
    ymir_cfg: edict = get_merged_config()
    gpu_id: str = str(ymir_cfg.param.get('gpu_id', '0'))
    gpu_count: int = ymir_cfg.param.get('gpu_count', None) or len(gpu_id.split(','))

    # preprocess, convert ymir dataset to trainable format
    convert_annotation_dataset(ymir_cfg)
    config_file: str = ymir_cfg.param.get('config_file')
    work_dir: str = ymir_cfg.ymir.output.models_dir
    if gpu_count == 0:
        # view https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#training-on-cpu
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', "-1")
        cmd = f"python3 tools/train.py {config_file} " + \
            f"--work-dir {work_dir}"
    elif gpu_count == 1:
        cmd = f"python3 tools/train.py {config_file} " + \
            f"--work-dir {work_dir} --gpu-id {gpu_id}"
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        port = find_free_port()
        os.environ.setdefault('PORT', str(port))
        cmd = f"bash ./tools/dist_train.sh {config_file} {gpu_count} " + \
            f"--work-dir {work_dir}"

    args_options: str = ymir_cfg.param.get("args_options", '')
    cfg_options: str = ymir_cfg.param.get("cfg_options", '')
    if args_options:
        cmd += f" {args_options}"

    if cfg_options:
        cmd += f" --cfg-options {cfg_options}"

    logging.info(f"training command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    write_last_ymir_result_file(ymir_cfg, id='last')
    return 0


if __name__ == '__main__':
    sys.exit(main())
