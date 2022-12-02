# ymir-mmsegmentation

## training

| hyper-parameter | type | example | usage |
| - | - | - | - |
| config_file | str | configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py | the basic config file |
| max_iters | int | 20000 | training iters |
| interval | int | 2000 | evaluation and checkpoint interval |
| samplers_per_gpu | int | 2 | batch size per gpu |
| workers_per_gpu | int | 2 | num_workers per gpu |
| max_keep_ckpts | int | -1, 3 | the number of saved weight file |
| save_least_file | bool | True | save all the weight file or last weight file only |
| cfg_options | str | optimizer.lr=0.02 | view utils/train.py for detail |
| args_options | str | --seed 25 | view utils/train.py for detail |
| export_format | str | seg-mask:raw | view ymir for detail |

### result.yaml demo

```
best_stage_name: best
map: 0.6434000000000001
model_stages:
  best:
    files:
    - fast_scnn_lr0.12_8x4_160k_cityscapes.py
    - best_mIoU_iter_4000.pth
    mAP: 0.6434000000000001
    stage_name: best
    timestamp: 1667802743
  last:
    files:
    - fast_scnn_lr0.12_8x4_160k_cityscapes.py
    - latest.pth
    mAP: 0.6434
    stage_name: last
    timestamp: 1667802761
```

## mining
- [ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation (CVPR 2020)](https://github.com/nihalsid/ViewAL)
