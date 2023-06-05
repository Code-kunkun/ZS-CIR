import torch 
import torch.nn as nn
from model.clip import clip 
import torch.nn.functional as F
from model.BLIP.models.blip_retrieval import blip_retrieval


class TransAgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.model_name = cfg.model_name
        if self.model_name == 'blip':
            self.pretrained_model = blip_retrieval(pretrained="/GPFS/data/yikunliu/cache/model_base_retrieval_coco.pth")
            self.feature_dim = 256
        elif self.model_name == 'clip-Vit-B/32':
            self.pretrained_model, self.preprocess = clip.load("/GPFS/data/yikunliu/cache/ViT-B-32.pt", device=cfg.device, jit=False)
            self.feature_dim = self.pretrained_model.visual.output_dim 
        elif self.model_name == 'clip-Vit-L/14':
            self.pretrained_model, self.preprocess = clip.load("/GPFS/data/yikunliu/cache/ViT-L-14.pt", device=cfg.device, jit=False)
            self.feature_dim = self.pretrained_model.visual.output_dim 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.logit_scale = 100
        self.dropout = nn.Dropout(cfg.dropout)
        self.combiner_layer = nn.Linear(self.feature_dim + self.feature_dim, (self.feature_dim + self.feature_dim) * 4)
        self.weighted_layer = nn.Linear(self.feature_dim, 3)
        self.output_layer = nn.Linear((self.feature_dim + self.feature_dim) * 4, self.feature_dim)
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))


    def forward(self, texts, reference_images, target_images):
        img_text_rep = self.combine_features(reference_images, texts)
        target_features, _ = self.pretrained_model.encode_image(target_images)
        target_features = F.normalize(target_features, dim=-1)
        logits = self.logit_scale * (img_text_rep @ target_features.T)
        return logits 
    
    def combine_features(self, reference_images, texts):
        reference_image_features, reference_total_image_features = self.pretrained_model.encode_image(reference_images, return_local=True)
        batch_size = reference_image_features.size(0)
        reference_total_image_features = reference_total_image_features.float()
        if self.model_name.startswith('blip'):
            tokenized_texts = self.pretrained_model.tokenizer(texts, padding='max_length', truncation=True, max_length=35,
                                                              return_tensors='pt').to(self.device)
            mask = (tokenized_texts.attention_mask == 0)
        elif self.model_name.startswith('clip'):
            tokenized_texts = clip.tokenize(texts, truncate=True).to(reference_image_features.device)
            mask = (tokenized_texts == 0)

        text_features, total_text_features = self.pretrained_model.encode_text(tokenized_texts)

        num_patches = reference_total_image_features.size(1)
        sep_token = self.sep_token.repeat(batch_size, 1, 1)

        combine_features = torch.cat((total_text_features, sep_token, reference_total_image_features), dim=1)

        image_mask = torch.zeros(batch_size, num_patches + 1).to(reference_image_features.device)
        mask = torch.cat((mask, image_mask), dim=1)
        
        img_text_rep = self.fusion(combine_features, src_key_padding_mask=mask) 
        
        if self.model_name.startswith('blip'):
            multimodal_img_rep = img_text_rep[:, 36, :] 
            multimodal_text_rep = img_text_rep[:, 0, :]
        elif self.model_name.startswith('clip'):
            multimodal_img_rep = img_text_rep[:, 78, :]
            multimodal_text_rep = img_text_rep[torch.arange(batch_size), tokenized_texts.argmax(dim=-1), :]

        concate = torch.cat((multimodal_img_rep, multimodal_text_rep), dim=-1)
        f_U = self.output_layer(self.dropout(F.relu(self.combiner_layer(concate))))
        weighted = self.weighted_layer(f_U) # (batch_size, 3)
        
        query_rep = weighted[:, 0:1] * text_features + weighted[:, 1:2] * f_U + weighted[:, 2:3] * reference_image_features
        
        query_rep = F.normalize(query_rep, dim=-1)

        return query_rep 
