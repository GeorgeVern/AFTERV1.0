# AFTER
This repository contains source code for our EMNLP 2020 Findings paper: [Domain Adversarial Fine-Tuning as an Effective Regularizer](https://arxiv.org/abs/2009.13366).

## Introduction
In this work, we propose a new type of regularizer for the fine-tuning process of pretrained Language Models (LMs). We identify the loss of general-domain representations of pretrained LMs during fine-tuning as a form of catastrophic forgetting. To address this, we complement the task-specific loss with an adversarial loss term that enforces invariance of text representations across different domains, while fine-tuning. The adversarial term acts as a regularizer that preserves most of the knowledge captured by the LM during pretraining, preventing catastrophic forgetting. 

Empirical results on 4 natural language understanding tasks (CoLA, MRPC, SST-2 and RTE) from the GLUE benchmark wιth two different pretrained LMs (BERT and XLNet) demonstrate improved performance over standard fine-tuning.

## Model


## Reference
    @misc{vernikos2020domain,
          title={Domain Adversarial Fine-Tuning as an Effective Regularizer}, 
          author={Giorgos Vernikos and Katerina Margatina and Alexandra Chronopoulou and Ion Androutsopoulos},
          year={2020},
          eprint={2009.13366},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
    
## Prerequisites
### Dependencies
* Python 3.6
* Pytorch 1.1.0
* Numpy 1.16.4
* Transformers 2.5.2
* Sklearn 0.0
### Install Requirements
*Create Environment (Optional):* Ideally, you should create an environment for the project.

    conda create -n after_env python=3.6
    conda activate after_env
Install PyTorch `1.1.0` with the desired Cuda version if you want to use the GPU:

`conda install pytorch==1.1.0 torchvision -c pytorch`

Clone the project:

```
git clone https://github.com/alexandra-chron/AFTERV1.0.git
cd AFTERV1.0
```

Then install the rest of the requirements:

`pip install -r requirements.txt`

### Download Data
### Main Data
To download the **Main** datasets we use the `download_glue_data.py` script from [here](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e).

For the EUROPARL Auxiliary Data
