import torch 
from dataclasses import dataclass

@dataclass
class Config:
    emb_dim: int = 256  # 256 for blip, 512 for CLIP Vit-base, 768 for CLIP vit-large
    dropout: float = 0.5 
    num_layers: int = 2 
    model_name: str = "blip" # [blip, clip-Vit-B/32, clip-Vit-L/14]
    device: torch.device = torch.device('cuda')
    batch_size: int = 64  # you can adjust it according to your GPU memory
    encoder: str = 'text' # ['neither', 'text', 'both']
    laion_type: str = 'laion_template' # ['laion_combined', 'laion_template', 'laion_llm'] choose different dataset
    transform: str = 'targetpad'
    target_ratio: float = 1.25
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_epsilon: float = 1e-8
    num_epochs: int = 100
    save_best: bool = True 
    use_amp: bool = True 
    validation_frequency: int = 1
    comment: str = "fiq_TransAgg_finetune_blip_text_combined"
    dataset: str='fiq' # ['fiq', 'cirr']
    save_path_prefix = "./wandb/"
    # eval related
    eval_load_path: str="/GPFS/data/yikunliu/image_retrieval_runs/wandb/2023-05-26-16-10-13_cirr_combiner_final_1e-4_finetune_blip_text_combined_31337_best_arithmetic.pth"
    submission_name: str='cirr_test_finetune_blip_text_combined_31337'
    
    