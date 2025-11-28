# CoGenT: A Unified Contrastive-Generative Framework for Time Series Classification. 
![Work in Progress](https://img.shields.io/badge/status-work_in_progress-orange)


> Published in IEEE Transactions on AI | [arXiv:2508.09451](https://arxiv.org/pdf/2508.09451)   
> Authors: Ziyu Liu (ziyu.liu2@student.rmit.edu.au), Azadeh Alavi (azadeh.alavi@rmit.edu.au), Minyi Li, Xiang Zhang.

# Overview

CoGenT is a unified self-supervised learning framework for time series that brings together the strengths of both contrastive and generative representation learning. Instead of relying on a single paradigm, CoGenT combines representation alignment with masked reconstruction within one architecture, enabling the model to learn features that are simultaneously discriminative and structure-aware. This unified design makes CoGenT broadly effective across diverse time-series datasets and tasks, while remaining simple, lightweight, and easy to integrate into existing pipelines.  

Framework of the proposed CoGenT:  

![Framework of the proposed CoGenT.](img/CoGenT_framework.PNG)

# Key Contributions & Results

- Unified contrastive–generative framework: Combines representation alignment and masked reconstruction in one architecture to learn both discriminative and structure-aware time-series features.

- Consistent improvements across six datasets: CoGenT outperforms the standard SimCLR and MAE on all evaluated datasets covering different channels, frequencies, and class counts.

- Strong overall performance: Achieves top F1 scores such as 0.9652 on FD and 0.9131 on FordA, with CoGenT delivering substantial gains over contrastive-only and generative-only baselines.

# Installation

```
# Clone
git clone https://github.com/DL4mHealth/cogent.git
cd cogent

# Create a Python environment
python -m venv .venv
source .venv/bin/activate    # mac/linux
# .venv\Scripts\activate     # windows

pip install -r requirements.txt
```

`requirements.txt` should include:
```
einops==0.8.1
numpy==1.24.3
pandas==2.0.3
PyYAML==6.0.3
scikit_learn==1.3.0
scipy==1.10.1
sktime==0.29.1
timm==0.6.12
torch==2.4.1
torchmetrics==1.4.0.post0
tqdm==4.66.5
ucimlrepo==0.0.7
```

# Quick start

This example shows how to run both self-supervised pretraining and supervised finetuning on the FordA dataset.  
The UCR datasets are loaded automatically via:
```
from sktime.datasets import load_UCR_UEA_dataset
```

No manual downloads are required.

### 1. Select the dataset in the UCR config

Open:
```
config/UCR_config.yaml
```

Uncomment the FordA section and comment out the others:
```yaml
# FordA
dataset_: "FordA"
n_class: 2
...

## other datasets...
#dataset_: "ChlorineConcentration"
#n_class: 3
#...
```
### 2. Pretraining

Set `pretrain: True` in the same config file:
```yaml
pretrain: True
# pretrain: False
```

Run:
```bash
python CoGenT_pretrain.py --config config/UCR_config.yaml
```

This performs self-supervised pretraining on FordA.

### 3. Finetuning WITH pretraining

To finetune using the pretrained model, keep the same flag:
```yaml
pretrain: True
```

Run:
```bash
python CoGenT_finetune.py --config config/UCR_config.yaml
```

This performs supervised finetuning using the pretrained checkpoint.

### 4. (Optional) Finetuning WITHOUT pretraining

If you want supervised finetuning from scratch, switch the flag:
```yaml
pretrain: False
```

And run:
```bash
python CoGenT_finetune.py --config config/UCR_config.yaml
```

This runs finetuning without loading any pretrained weights.


# Citation

If you find this work useful for your research, please consider citing this paper:
```
@article{liu2025unified,
  title={A Unified Contrastive-Generative Framework for Time Series Classification},
  author={Liu, Ziyu and Alavi, Azadeh and Li, Minyi and Zhang, Xiang},
  journal={arXiv preprint arXiv:2508.09451},
  year={2025}
}
```

# License

This repository is released under the Apache-2.0 license. Please see the LICENSE file for details.

# Contact

For questions regarding the code, please contact the author Ziyu Liu (ziyu.liu2@student.rmit.edu.au).  
The paper states the intent to release full code for reproducibility.
