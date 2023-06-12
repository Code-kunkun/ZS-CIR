import torch 
from dataclasses import dataclass

@dataclass
class Config:
    dropout: float = 0.5 
    num_layers: int = 2 
    model_name: str = "blip" # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 64  # you can adjust it according to your GPU memory
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_combined' # ['laion_combined', 'laion_template', 'laion_llm'] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 100
    save_best: bool = True 
    use_amp: bool = True 
    validation_frequency: int = 1
    comment: str = "cirr_TransAgg_finetune_blip_text_combined"
    dataset: str='cirr' # ['fiq', 'cirr']
    save_path_prefix = "/GPFS/data/yikunliu/image_retrieval_runs/wandb"
    # eval related
    eval_load_path: str="xxx"
    submission_name: str='cirr_test_finetune_blip_text_combined'
    
    