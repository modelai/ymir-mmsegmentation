import os
import sys

from ymir_exc.executor import Executor


def main() -> int:
    apps = dict(training='python3 ymir/ymir_training.py',
                mining='python3 ymir/ymir_mining.py',
                infer='python3 ymir/ymir_infer.py')
    executor = Executor(apps)
    executor.start()

    return 0


if __name__ == '__main__':
    # fix mkl-service error, view https://github.com/pytorch/pytorch/issues/37377#issuecomment-629530272 for detail
    os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    sys.exit(main())
