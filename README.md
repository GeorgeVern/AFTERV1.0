# *AFTER* - *A*dversarial *F*ine-*T*uning as an *E*ffective *R*egularizer
This repository contains source code for our EMNLP 2020 Findings paper: [Domain Adversarial Fine-Tuning as an Effective Regularizer](https://www.aclweb.org/anthology/2020.findings-emnlp.278/).

## Introduction
In this work, we propose a new type of regularizer for the fine-tuning process of pretrained Language Models (LMs). We identify the loss of general-domain representations of pretrained LMs during fine-tuning as a form of **catastrophic forgetting**.
 The adversarial term acts as a regularizer that preserves most of the knowledge captured by the LM during pretraining, preventing catastrophic forgetting. 
 
## Model
To address it, we extend the standard fine-tuning process of pretrained LMs with with an adversarial objective. This additional loss term is related to an adversarial classifier, that discriminates between *in-domain* and *out-of-domain* text representations. 

- *In-domain*: labeled dataset of the task (**Main**) at hand 

- *Out-of-domain*: **unlabeled data** from a different domain (**Auxiliary**)

We *minimize* the task-specific loss and at the same time *maximize* the loss of the domain classifier using a [Gradient Reversal Layer](https://jmlr.org/papers/v17/15-239.html).

The loss function we propose is the following:

L<sub>after</sub> = L<sub>main</sub> - λL<sub>domain</sub>

where L<sub>main</sub> is the task-specific loss and L<sub>domain</sub> an adversarial loss that enforces invariance of text representations across different domains, while fine-tuning. 
λ is a tunable hyperparameter.
 

![AFTER_fig-1](https://user-images.githubusercontent.com/30960204/95763721-b88d2500-0caf-11eb-9220-c8d1df3b62ee.jpg)

## Results 

Experiments on 4 GLUE datasets (CoLA, MRPC, SST-2 and RTE) wιth two different pretrained LMs (BERT and XLNet) demonstrate improved performance over standard fine-tuning.
We show empirically that the adversarial term acts as a regularizer that preserves most of the knowledge captured by the LM during pretraining, preventing catastrophic forgetting. 


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
#### Main Data
To download the **Main** datasets we use the `download_glue_data.py` script from [here](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). You can choose the datasets used in the paper by running the following command:

` python download_glue_data.py --data_dir './Datasets' --tasks 'CoLA,SST,RTE,MRPC'`

*The default path for the datasets is AFTERV1.0/Datasets but any other path can be used (should agree with the `DATA_DIR` path specified in the `sys_config` script)*

#### Auxiliary Data
As **Auxiliary** data we use corpora from various domains. We provide scripts to download and preprocess the corpora used in our experiments, while any other corpora can be used as well.


## AFTER - Fine-tune a pretrained model

To run AFTER with BERT, you need the following command:

` python after_fine-tune.py -i afterBert_finetune_cola_europarl --lambd 0.1`

`lambd` refers to lambda, the weight of the joint loss function that we use.


In `configs/`, you can see a list of yaml files we used for the experiments and can also change their hyperparameters. 

## Reference
If you use this repo in your research, please cite the paper:

    @inproceedings{vernikos-etal-2020-domain,
        title = "{D}omain {A}dversarial {F}ine-{T}uning as an {E}ffective {R}egularizer",
        author = "Vernikos, Giorgos  and
          Margatina, Katerina  and
          Chronopoulou, Alexandra  and
          Androutsopoulos, Ion",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
        year = "2020",
        url = "https://www.aclweb.org/anthology/2020.findings-emnlp.278",
        doi = "10.18653/v1/2020.findings-emnlp.278",
        pages = "3103--3112",
    }
    