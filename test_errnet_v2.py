"""Test script for ERRNet V2 with Transformer backbone + frequency enhancement.

Same interface as test_errnet.py, but uses V2 model and options.
"""

import sys
from os.path import join

import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
from engine import Engine
from options.errnet.train_options_v2 import TrainOptionsV2

from test_errnet import (
    EVAL_DATASETS,
    TEST_DATASETS,
    parse_test_args,
    build_eval_dataloader,
    build_test_dataloader,
)


def main():
    cli_args = parse_test_args()
    option_parser = TrainOptionsV2()
    option_parser.isTrain = False
    opt = option_parser.parse()
    opt.isTrain = False
    opt.model = 'errnet_model_v2'

    cudnn.benchmark = len(opt.gpu_ids) > 0
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False

    engine = Engine(opt)

    if cli_args.dataset in EVAL_DATASETS:
        spec, dataloader = build_eval_dataloader(
            opt,
            cli_args.data_root,
            cli_args.dataset,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or spec["save_subdir"]
        res = engine.eval(
            dataloader,
            dataset_name=spec["dataset_name"],
            savedir=join(cli_args.result_dir, save_subdir),
        )
        print(res)
    else:
        default_save_subdir, dataloader = build_test_dataloader(
            opt,
            cli_args.dataset,
            cli_args.input_dir,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or default_save_subdir
        engine.test(
            dataloader,
            savedir=join(cli_args.result_dir, save_subdir),
        )


if __name__ == "__main__":
    main()
