# AFTER
This repository contains source code for our EMNLP 2020 Findings paper: [Domain Adversarial Fine-Tuning as an Effective Regularizer](https://arxiv.org/abs/2009.13366).

## Introduction
In this work, we propose a new type of regularizer for the fine-tuning process of pretrained Language Models (LMs). We identify the loss of general-domain representations of pretrained LMs during fine-tuning as a form of catastrophic forgetting. To address this, we complement the task-specific loss with an adversarial loss term that enforces invariance of text representations across different domains, while fine-tuning. The adversarial term acts as a regularizer that preserves most of the knowledge captured by the LM during pretraining, preventing catastrophic forgetting. 

Empirical results on 4 natural language understanding tasks (CoLA, MRPC, SST-2 and RTE) from the GLUE benchmark wιth two different pretrained LMs (BERT and XLNet) demonstrate improved performance over standard fine-tuning.

## Model
We extend the standard fine-tuning process of pretrained LMs with with an adversarial objective. This additional loss term is related to an adversarial classifier, that aims to discriminate between *in-domain* and *out-of-domain* text representations. In-domain refers to the labeled dataset of the task (**Main**) at hand while out-of-domain refers to **unlabeled data** from a different domain (**Auxiliary**). 

Hence, we minimize the task-specific loss and at the same time maximize the loss of the domain classifier using a [Gradient Reversal Layer](https://jmlr.org/papers/v17/15-239.html):

![AFTER_fig-1](https://user-images.githubusercontent.com/30960204/95763721-b88d2500-0caf-11eb-9220-c8d1df3b62ee.jpg)

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
To download the **Main** datasets we use the `download_glue_data.py` script from [here](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). You can choose the datasets used in the paper by running the following command:

` python download_glue_data.py --data_dir './Datasets' --tasks 'CoLA,SST,RTE,MRPC'`

*The default path for the datasets is AFTERV1.0/Datasets but any other path can be used (should agree with the `DATA_DIR` path specified in the `sys_config` script)*

### Auxiliary Data


For the EUROPARL Auxiliary Data
