import torch 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import random 
from torch import optim
from model.model import TransAgg
from transform import targetpad_transform, squarepad_transform
from data.cirr_dataset import CIRRDataset
from data.fiq_dataset import FashionIQDataset
from data.laion_dataset_template import LaionDataset_Template
from data.laion_dataset_llm import LaionDataset_LLM
from data.laion_dataset_combined import LaionDataset_Combined

def get_model(cfg):
    model = TransAgg(cfg)
    model = model.to(cfg.device)
    return model

def set_grad(cfg, model):
    if cfg.encoder == 'text':
        print('Only the text encoder will be fine-tuned')
        if cfg.model_name.startswith("blip"):
            for param in model.pretrained_model.visual_encoder.parameters():
                param.requires_grad = False
            for param in model.pretrained_model.vision_proj.parameters():
                param.requires_grad = False 
        elif cfg.model_name.startswith('clip'):
            for param in model.pretrained_model.visual.parameters():
                param.requires_grad = False
    elif cfg.encoder == 'both':
        print('Both encoders will be fine-tuned')
    elif cfg.encoder == 'neither':
        for param in model.pretrained_model.parameters():
            param.requires_grad = False
    else:
        raise ValueError("encoder parameter should be in ['text', 'both', 'neither']")


def get_preprocess(cfg, model, input_dim):
    if cfg.transform == "clip":
        preprocess = model.preprocess
        print('CLIP default preprocess pipeline is used')
    elif cfg.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif cfg.transform == "targetpad":
        target_ratio = cfg.target_ratio
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    return preprocess

def get_laion_cirr_dataset(preprocess, laion_type):
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    if laion_type == 'laion_template':
        relative_train_dataset = LaionDataset_Template('train', preprocess)
    elif laion_type == 'laion_llm':
        relative_train_dataset = LaionDataset_LLM('train', preprocess)
    elif laion_type == 'laion_combined':
        relative_train_dataset = LaionDataset_Combined('train', preprocess)
    else:
        raise ValueError("laion_type should be in ['laion_template', 'laion_llm', 'laion_combined']")

    return relative_train_dataset, relative_val_dataset, classic_val_dataset

def get_laion_fiq_dataset(preprocess, val_dress_types, laion_type):

    if laion_type == 'laion_template':
        relative_train_dataset = LaionDataset_Template('train', preprocess)
    elif laion_type == 'laion_llm':
        relative_train_dataset = LaionDataset_LLM('train', preprocess)
    elif laion_type == 'laion_combined':
        relative_train_dataset = LaionDataset_Combined('train', preprocess)
    else:
        raise ValueError("laion_type should be in ['laion_template', 'laion_llm', 'laion_combined']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        classic_val_datasets.append(classic_val_dataset)
    return relative_train_dataset, relative_val_datasets, classic_val_datasets, idx_to_dress_mapping


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def extract_index_features(dataset, model, return_local=True):
    feature_dim = model.feature_dim 
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(model.device, non_blocking=True) 
    index_total_features = []
    index_names = []

    for names, images in tqdm(classic_val_loader):
        images = images.to(model.device, non_blocking=True)
        with torch.no_grad():
            batch_features, batch_total_features = model.pretrained_model.encode_image(images, return_local)
            index_features = torch.vstack((index_features, batch_features))
            index_total_features.append(batch_total_features)
            index_names.extend(names)
    if return_local:
        with torch.no_grad():
            index_total_features = torch.cat(index_total_features, dim=0).to(model.device, non_blocking=True)
    else:
        index_total_features = None 
    return index_features, index_names, index_total_features


def get_optimizer(model, cfg):
    pretrained_params = list(map(id, model.pretrained_model.parameters()))
    optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if p.requires_grad and id(p) not in pretrained_params], 'weight_decay': cfg.weight_decay},
      {'params': [p for n, p in model.named_parameters() if p.requires_grad and id(p) in pretrained_params], 'weight_decay': cfg.weight_decay, 'lr': 1e-6},
      ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    return optimizer 


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    train_running_results['accumulated_train_loss'] += loss.item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch

def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    if train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'] < 0:
        print(train_running_results['accumulated_train_loss'], train_running_results['images_in_epoch'])
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss : {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )

def generate_randomized_fiq_caption(flattened_captions: list[str]) -> list[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions

