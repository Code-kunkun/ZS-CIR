import json
import multiprocessing
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from model.model import TransAgg
from utils import get_preprocess, extract_index_features
from data.cirr_dataset import CIRRDataset


def generate_cirr_test_submissions(file_name, model, preprocess, device):
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)
    index_features, index_names, _ = extract_index_features(classic_test_dataset, model, return_local=False)
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset,index_features, 
                                                                                  index_names, model, device)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    print(f"Saving CIRR test predictions")
    with open(f"./submission/recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(f"./submission/recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset, index_features, index_names, model, device):
    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(relative_test_dataset, model, device)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(relative_test_dataset: CIRRDataset, model, device) -> Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    print(f"Compute CIRR test predictions")
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = []
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members, reference_images in tqdm(
            relative_test_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            reference_images = reference_images.to(device)
            batch_predicted_features = model.combine_features(reference_images, captions)
            predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))

        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    predicted_features = torch.cat(predicted_features, dim=0)

    return predicted_features, reference_names, group_members, pairs_id


def main():
    cfg = Config()
    model = TransAgg(cfg)
    device = cfg.device 
    model = model.to(device)
    model.load_state_dict(torch.load(cfg.eval_load_path))

    # model.load_state_dict({k.replace('blip_model.', 'pretrained_model.'): v for k, v in torch.load(cfg.eval_load_path).items()})
    # input_dim = model.clip_model.visual.input_resolution
    if cfg.model_name.startswith("blip"):
        input_dim = 384
    elif cfg.model_name.startswith("clip"):
        input_dim = model.pretrained_model.visual.input_resolution

    preprocess = get_preprocess(cfg, model, input_dim=input_dim)

    model.eval()

    generate_cirr_test_submissions(cfg.submission_name, model, preprocess, device=device)


if __name__ == '__main__':
    main()
