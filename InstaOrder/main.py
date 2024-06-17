import multiprocessing as mp
import argparse
import os
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from utils import dist_init, dist_init_
from trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

    BASE_DIR = config['data']['base_dir']

    for k, v in config.items():
        for kkk, vvv in v.items():
            if type(vvv) == str and '/data/' in vvv:
                v[kkk] = f"{BASE_DIR}{vvv}"
        setattr(args, k, v)

    # exp path
    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    # dist init
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    dist_init_(args.launcher, backend='nccl', dist_url=args.dist_url)
    #     dist_init(args.launcher, backend='gloo')

    if "Partial" not in args.model['algo']:
        args.load_pretrain = True

    # train
    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InstaOrder')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--launcher', default='pytorch', type=str)
    parser.add_argument('--load-iter', default=None, type=int)
    parser.add_argument('--load_pretrain', default=None, type=str)
    parser.add_argument('--load_model', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate-save', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:1234')
    args = parser.parse_args()

    main(args)
