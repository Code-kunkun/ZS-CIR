# Zero-shot Composed Text-Image Retrieval

This repository contains the official Pytorch implementation of TransAgg: xxx

## Environment
Create the environment for running our code as follow:

```
conda create --name transagg python=3.9.16
conda activate transagg
pip install -r requirements.txt
```

## Datasets

**Laion-CIR-Template、Laion-CIR-LLM and Laion-CIR-Combined**: please refer to this [link](https://drive.google.com/drive/folders/1EGpylkOMj9tduUjAhTLtaX5UqjPMyN3X?usp=sharing)

**FashionIQ**: Please refer to the [FashionIQ repo](https://github.com/XiaoxiaoGuo/fashion-iq) to get the datasets.

**CIRR**: Please refer to the [CIRR repo](https://github.com/Cuberick-Orion/CIRR#download-cirr-dataset) for instructions.

## Model Zoo

### Pretrained Model

**clip-Vit-B/32**: please refer to this [link](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)

**clip-Vit-L/14**: please refer to this [link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)

**blip**: please refer to this [link](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth)

### Checkpoints
[https://drive.google.com/drive/folders/1EGpylkOMj9tduUjAhTLtaX5UqjPMyN3X?usp=sharing](https://drive.google.com/drive/folders/1EGpylkOMj9tduUjAhTLtaX5UqjPMyN3X?usp=sharing)



## Train 
**note that**, you can modify the relevant parameters in the `config.py` file
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Test CIRR Dataset
**note that**, you can modify the relevant parameters in the `config.py` file
```
python cirr_test_submission.py
```

## Citation
if you use this code for your research or project, please cite:


## Acknowledgements
Many thanks to the code bases from [CLIP4CIR](https://github.com/ABaldrati/CLIP4Cir), [CLIP](https://github.com/openai/CLIP), [BLIP](https://github.com/salesforce/BLIP)