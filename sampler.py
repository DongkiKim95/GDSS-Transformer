import os
import time
import pickle
import math
import torch

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
                         load_ema_from_ckpt, load_sampling_fn, load_eval_settings
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from evaluation.molsets import get_all_metrics


# -------- Sampler for generic graph generation tasks --------
class Sampler(object):
    def __init__(self, config):
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        load_seed(self.configt.seed)
        self.train_graph_list, self.val_graph_list, self.test_graph_list = load_data(self.configt, get_graph_list=True)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt, self.ckpt_dict['params'])
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema = load_ema_from_ckpt(self.model, self.ckpt_dict['ema'], self.configt.train.ema)
            self.ema.copy_to(self.model.parameters())

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(len(self.test_graph_list) / self.config.sample.batch_size)
        gen_graph_list = []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            self.init_flags = init_flags(self.train_graph_list, self.configt, self.config.sample.batch_size).to(self.device[0])

            x, adj, _ = self.sampling_fn(self.model, self.init_flags)

            logger.log(f"Round {r} : {time.time()-t_start:.2f}s")

            samples_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))
            
        gen_graph_list = gen_graph_list[:len(self.test_graph_list)]

        # -------- Evaluation --------
        methods, kernels = load_eval_settings(self.config.data.data, kernel=self.config.sample.kernel)
        result_dict = eval_graph_list(self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
        logger.log(f'MMD_full {result_dict}', verbose=False)
        logger.log('='*100)

        # -------- Save samples --------
        save_dir = save_graph_list(self.log_folder_name, self.log_name, gen_graph_list)
        with open(save_dir, 'rb') as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(graphs=sample_graph_list, title=f'{self.config.ckpt}', max_num=16, save_dir=self.log_folder_name)


# -------- Sampler for molecule generation tasks --------
class Sampler_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt, self.ckpt_dict['params'])
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)
        
        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)     # for init_flags
        with open(f'data/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
            self.test_graph_list = pickle.load(f)                                   # for NSPDK MMD


        num_sampling_rounds = math.ceil(10000 / self.config.sample.batch_size)
        xs, adjs = [], []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            self.init_flags = init_flags(self.train_graph_list, self.configt, self.config.sample.batch_size).to(self.device[0])
            x, adj, _ = self.sampling_fn(self.model, self.init_flags)
        
            samples_int = quantize_mol(adj)

            samples_int = samples_int - 1
            samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

            adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
            x = torch.where(x > 0.5, 1, 0)
            x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

            xs.append(x), adjs.append(adj)

        xs, adjs = torch.concat(xs, dim=0), torch.concat(adjs, dim=0)

        gen_mols, num_mols_wo_correction = gen_mol(xs, adjs, self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
                
        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
            for smiles in gen_smiles:
                f.write(f'{smiles}\n')

        # -------- Evaluation --------
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles, train=train_smiles)
        scores_nspdk = eval_graph_list(self.test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

        logger.log(f'Number of molecules: {num_mols}')
        logger.log(f'validity w/o correction: {num_mols_wo_correction / num_mols}')
        for metric in ['FCD/Test', 'Scaf/Test', 'Frag/Test', 'SNN/Test', f'unique@{len(gen_smiles)}', 'Novelty', 'valid']:
            logger.log(f'{metric}: {scores[metric]}')
        logger.log(f'NSPDK MMD: {scores_nspdk}')
        logger.log('='*100)
