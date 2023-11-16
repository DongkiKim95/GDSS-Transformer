import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log
from utils.graph_utils import rand_perm 


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        
        if self.config.data.data in ['QM9', 'ZINC250k']:
            self.train_loader, self.test_loader = load_data(self.config)
        else:
            self.train_loader, self.val_loader, self.test_loader, train_feat_dim = load_data(self.config)
            self.config.data.feat.dim = train_feat_dim
            self.config.data.max_feat_num = sum(train_feat_dim)

        self.params = load_model_params(self.config)
    
    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train, self.device)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config, self.params)

        self.loss_fn = load_loss_fn(self.config) 

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):
            train_loss_x = []
            train_loss_adj = []
            t_start = time.time()
            self.model.train()
            for _, train_b in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, adj = load_batch(train_b, self.device) 
                y = None
                loss_subject = (x, adj, y) if not self.config.data.perm_mix else (*rand_perm(x, adj), y)

                loss, loss_x, loss_adj = self.loss_fn(self.model, *loss_subject)
                loss.backward()

                if self.config.train.grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                else:
                    grad_norm = 0
                self.optimizer.step()

                # -------- EMA update --------
                self.ema.update(self.model.parameters())

                train_loss_x.append(loss_x.item())
                train_loss_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            mean_train_x = np.mean(train_loss_x)
            mean_train_adj = np.mean(train_loss_adj)

            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | '
                        f'grad_norm: {grad_norm:.2e} |', verbose=False)

            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''

                torch.save({ 
                    'epoch': epoch,
                    'config': self.config,
                    'params' : self.params,
                    'state_dict': self.model.state_dict(),
                    'ema': self.ema.state_dict(),
                    }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
                torch.save(self.optimizer.state_dict(), f'./checkpoints/{self.config.data.data}/{self.ckpt}_optimizer.pth')
        print(' ')
        return self.ckpt