import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol


def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        ckpt = trainer.train(ts)
        # if 'sample' in config.keys():
        #     config.ckpt = ckpt
        #     sampler = Sampler(config) 
        #     sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
    elif work_type_args.type == 'search':
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config)
        
        #0.01, 0.05, 
        # for snr in [0.1, 0.15, 0.2, 0.25, 0.3]:
        #     for seps in [0.6, 0.7, 0.8, 0.9, 1.0]:
        
        # sampler.config.sampler.corrector = 'None'
        # sampler.sample()
        sampler.config.sampler.corrector = 'Langevin'
        for snr in [0.15, 0.2, 0.25, 0.3]:
            for seps in [0.02,0.04,0.06,0.08]:
                sampler.config.sampler.snr = snr
                sampler.config.sampler.scale_eps = seps
                sampler.sample()
    elif work_type_args.type == 'search_seed':
        sampler = Sampler(config)
        for seed in [23,57,97,55,34,89,97,129,41,37]:
            sampler.config.sample.seed = seed
            sampler.sample()

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
