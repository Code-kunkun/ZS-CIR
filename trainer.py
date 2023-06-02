from tqdm import tqdm 
import torch 
from torch.utils.data import DataLoader 
import numpy as np 
from statistics import mean
import wandb 
from utils import set_train_bar_description, update_train_running_results, extract_index_features, collate_fn


class Trainer():

    def __init__(self, cfg, model, train_dataloader, optimizer, scheduler, criterion, classic_val_dataset, relative_val_dataset, **kwargs):
        self.num_epochs = cfg.num_epochs
        self.dataset = cfg.dataset
        if self.dataset == 'fiq':
            self.idx_to_dress_mapping = kwargs['idx_to_dress_mapping']
        self.model = model
        self.train_dataloader = train_dataloader 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = cfg.device
        self.use_amp = cfg.use_amp
        self.criterion = criterion 
        self.encoder = cfg.encoder 
        self.classic_val_dataset = classic_val_dataset
        self.relative_val_dataset = relative_val_dataset
        self.validation_frequency = cfg.validation_frequency
        self.save_path = cfg.save_path 
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        if self.encoder == 'text' or self.encoder == 'neither':
            self.store_val_features = kwargs 

    def train(self):
        best_score = 0
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

            if epoch % self.validation_frequency == 0:
                results_dict = {}
                if self.dataset == 'cirr':
                    results = self.eval_cirr()
                    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
                    results_dict = {
                        'group_recall_at1': group_recall_at1,
                        'group_recall_at2': group_recall_at2,
                        'group_recall_at3': group_recall_at3,
                        'recall_at1': recall_at1,
                        'recall_at5': recall_at5,
                        'recall_at10': recall_at10,
                        'recall_at50': recall_at50,
                        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                        'arithmetic_mean': mean(results),
                    }
                    
                    print('recall_inset_top1_correct_composition', group_recall_at1)
                    print('recall_inset_top2_correct_composition', group_recall_at2)
                    print('recall_inset_top3_correct_composition', group_recall_at3)
                    print('recall_top1_correct_composition', recall_at1)
                    print('recall_top5_correct_composition', recall_at5)
                    print('recall_top10_correct_composition', recall_at10)
                    print('recall_top50_correct_composition', recall_at50)

                elif self.dataset == 'fiq':
                    results10, results50 = self.eval_fiq()
                    for i in range(len(results10)):
                        results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at10'] = results10[i]
                        results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at50'] = results50[i]
                        print(f'{self.idx_to_dress_mapping[i]}_recall_at10: {results10[i]}')
                        print(f'{self.idx_to_dress_mapping[i]}_recall_at50: {results50[i]}')
                    print('average_recall_at10', mean(results10))
                    print('average_recall_at50', mean(results50))

                    results_dict.update({
                        'average_recall_at10': mean(results10),
                        'average_recall_at50': mean(results50),
                        'average_recall': (mean(results10) + mean(results50)) / 2
                    })

                wandb.log(results_dict)
                if self.dataset == 'cirr':
                    score = mean(results)
                elif self.dataset == 'fiq':
                    score = results_dict['average_recall']
                if score > best_score:
                    best_score = score 
                    self.save_checkpoint(self.save_path)

                    

    def train_epoch(self, epoch):
        self.model.train() 
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(self.train_dataloader, ncols=150)
        iters = len(train_bar)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            self.optimizer.zero_grad()
            reference_images = reference_images.to(self.device, non_blocking=True)
            target_images = target_images.to(self.device, non_blocking=True)

            if not self.use_amp:
                logits = self.model(reference_images, captions, target_images)
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=self.device)
                loss = self.criterion(logits, ground_truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                self.scheduler.step(epoch + idx / iters)
            else:
                with torch.cuda.amp.autocast():
                    logits = self.model(captions, reference_images, target_images)
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=self.device)
                    loss = self.criterion(logits, ground_truth)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.scaler.step(self.optimizer)
                self.scheduler.step(epoch + idx / iters)
                self.scaler.update()

            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, self.num_epochs, train_running_results)
            # wandb to log 
        train_epoch_loss = float(train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
        wandb.log({'train_epoch_loss': train_epoch_loss})

    def get_val_index_features(self, index=None):

        with torch.no_grad():
            if (self.encoder == 'both' or self.encoder == 'image') and self.dataset == 'cirr':
                val_index_features, val_index_names, _ = extract_index_features(self.classic_val_dataset, self.model, return_local=False)
            elif self.dataset == 'cirr':
                val_index_features, val_index_names, _ = self.store_val_features['val_index_features'], self.store_val_features['val_index_names'], self.store_val_features['val_total_index_features']
            elif (self.encoder == 'both' or self.encoder == 'image') and self.dataset == 'fiq':
                val_index_features, val_index_names, _ = extract_index_features(self.classic_val_dataset[index], self.model, return_local=False)
            elif self.dataset == 'fiq':
                val_index_features, val_index_names, _ = self.store_val_features['val_index_features'][index], self.store_val_features['val_index_names'][index], self.store_val_features['val_total_index_features'][index]
        
        return val_index_features, val_index_names, _


    def eval_cirr(self):
        self.model.eval()
        val_index_features, val_index_names, _ = self.get_val_index_features()
        results = self.compute_cirr_val_metrics(val_index_names, val_index_features)
        return results 


    def eval_fiq(self):
        self.model.eval()
        recalls_at10 = []
        recalls_at50 = []
        for idx in self.idx_to_dress_mapping:
            val_index_features, val_index_names, val_index_total_features = self.get_val_index_features(index=idx)
            recall_at10, recall_at50 = self.compute_fiq_val_metrics(val_index_names, val_index_features, val_index_total_features, idx)
            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
        results_dict = {}
        for i in range(len(recalls_at10)):
            results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
            results_dict[f'{self.idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
        wandb.log(results_dict)
        return recalls_at10, recalls_at50     

    def get_val_dataloader(self, index=None):
        if index == None:
            dataset = self.relative_val_dataset
        else:
            dataset = self.relative_val_dataset[index]
        relative_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        return relative_val_loader

    def compute_fiq_val_metrics(self, val_index_names, val_index_features, val_total_index_features, index):
        relative_val_loader = self.get_val_dataloader(index)
        target_names = []
        predicted_features = []

        for batch_reference_names, batch_target_names, captions, reference_images in tqdm(relative_val_loader):

            flattened_captions: list = np.array(captions).T.flatten().tolist()
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
                i in range(0, len(flattened_captions), 2)]
            with torch.no_grad():
                reference_images = reference_images.to(self.device)
                batch_predicted_features = self.model.combine_features(reference_images, input_captions)
                predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))
            
            target_names.extend(batch_target_names)
        predicted_features = torch.cat(predicted_features, dim=0)
        val_index_features = val_index_features / val_index_features.norm(dim=-1, keepdim=True)

        distances = 1 - predicted_features @ val_index_features.T
        results = self.compute_results(distances, val_index_names, target_names)
        
        return results 
            

    def compute_cirr_val_metrics(self, val_index_names, val_index_features):

        relative_val_loader = self.get_val_dataloader()
        
        target_names = []
        group_members = []
        reference_names = []
        predicted_features = []
        for batch_reference_names, batch_target_names, captions, batch_group_members, reference_images in tqdm(relative_val_loader):
            batch_group_members = np.array(batch_group_members).T.tolist()
            with torch.no_grad():
                reference_images = reference_images.to(self.device)
                batch_predicted_features = self.model.combine_features(reference_images, captions)
                predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))

            target_names.extend(batch_target_names)
            group_members.extend(batch_group_members)
            reference_names.extend(batch_reference_names)
        
        predicted_features = torch.cat(predicted_features, dim=0)

        val_index_features = val_index_features / val_index_features.norm(dim=-1, keepdim=True)

        distances = 1 - predicted_features @ val_index_features.T

        results = self.compute_results(distances, val_index_names, target_names, reference_names, group_members)
        
        return results 
    
    def compute_results(self, distances, val_index_names,  target_names, reference_names=None, group_members=None):
        sorted_indices = torch.argsort(distances, dim=-1).cpu()
        sorted_index_names = np.array(val_index_names)[sorted_indices]

        if reference_names == None:
            labels = torch.tensor(
                sorted_index_names == np.repeat(np.array(target_names), len(val_index_names)).reshape(len(target_names), -1))

            recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
            recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

            return recall_at10, recall_at50

        elif reference_names != None:
            reference_mask = torch.tensor(
                sorted_index_names != np.repeat(np.array(reference_names), len(val_index_names)).reshape(len(target_names), -1))
            sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                            sorted_index_names.shape[1] - 1)

            labels = torch.tensor(
                sorted_index_names == np.repeat(np.array(target_names), len(val_index_names) - 1).reshape(len(target_names), -1))

            group_members = np.array(group_members)
        
            group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
            group_labels = labels[group_mask].reshape(labels.shape[0], -1)

            assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
            assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

            # Compute the metrics
            recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
            recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
            recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
            recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
            group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
            group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
            group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

            return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)




        