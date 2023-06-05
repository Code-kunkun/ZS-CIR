import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
import multiprocessing
import torch
from torch.utils.data import DataLoader 
from torch import nn
import random 
import numpy as np 
from trainer import Trainer
from config import Config
import datetime
from utils import get_model, set_grad, get_preprocess, get_laion_cirr_dataset, get_laion_fiq_dataset, extract_index_features, collate_fn, get_optimizer

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.deterministic = True


def main(cfg):
    # setup_seed(1024) 
    
    # get the corresponding model 
    model = get_model(cfg)
    set_grad(cfg, model)
    model.pretrained_model.eval().float()

    
    # input_dim = combiner.blip_model.visual.input_resolution
    if cfg.model_name.startswith('blip'):
        input_dim = 384
    elif cfg.model_name.startswith('clip'):
        input_dim = model.pretrained_model.visual.input_resolution
    preprocess = get_preprocess(cfg, model, input_dim)

    if cfg.dataset == 'fiq':
        val_dress_types = ['dress', 'toptee', 'shirt']
        relative_train_dataset, relative_val_dataset, classic_val_dataset, idx_to_dress_mapping = get_laion_fiq_dataset(preprocess, val_dress_types, cfg.laion_type) 
    # get dataset and dataloader 
    elif cfg.dataset == 'cirr':
        relative_train_dataset, relative_val_dataset, classic_val_dataset = get_laion_cirr_dataset(preprocess, cfg.laion_type)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=cfg.batch_size,
                                       num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over the epochs
    kwargs = {}
    if cfg.dataset == 'fiq':
        kwargs['val_index_features'] = []
        kwargs['val_index_names'] = []
        kwargs['val_total_index_features'] = []
        kwargs['idx_to_dress_mapping'] = idx_to_dress_mapping
    if cfg.dataset == 'cirr' and (cfg.encoder == 'text' or cfg.encoder == 'neither'):
        val_index_features, val_index_names, val_total_index_features = extract_index_features(classic_val_dataset, model, return_local=False)
        kwargs['val_index_features'], kwargs['val_index_names'], kwargs['val_total_index_features'] = val_index_features, val_index_names, val_total_index_features
    elif cfg.dataset == 'fiq' and (cfg.encoder == 'text' or cfg.encoder == 'neither'):
        for classic_val_dataset_ in classic_val_dataset:
            val_index_features, val_index_names, _ = extract_index_features(classic_val_dataset_, model, return_local=False)
            kwargs['val_index_features'].append(val_index_features)
            kwargs['val_index_names'].append(val_index_names)
            kwargs['val_total_index_features'].append(_) 

    # Define the optimizer, the loss and the grad scaler
    optimizer = get_optimizer(model, cfg)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.num_epochs, eta_min=1e-2 * cfg.learning_rate, last_epoch=-1)
    crossentropy_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(cfg, model, relative_train_loader, optimizer, lr_scheduler, crossentropy_criterion, classic_val_dataset, relative_val_dataset, **kwargs)
    trainer.train()

    """
    if you just want to eval
        (1) model.load_state_dict(torch.load(model_path))
        (2) trainer.eval_cirr() or trainer.eval_fiq()
    """

if __name__ == '__main__':
    cfg = Config()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    cfg.save_path = f"{cfg.save_path_prefix}/{current_time}_{cfg.comment}_best_arithmetic.pth" 

    wandb_config = vars(cfg)

    wandb.init(project='ZeroShot-CIR', notes=cfg.comment, config=wandb_config, name=cfg.comment)

    main(cfg)

    wandb.finish()
