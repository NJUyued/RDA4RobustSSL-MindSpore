# RDA4RobustSSL-MindSpore

This repo is the official MindSpore implementation of our paper:

> ***RDA: Reciprocal Distribution Alignment for Robust Semi-supervised Learning***  
**Authors**: Yue Duan, Lei Qi, Lei Wang, Luping Zhou and Yinghuan Shi  

 
 - Quick links: [[arXiv](https://arxiv.org/abs/2208.04619) | [Published paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900527.pdf) | [Poster](/figures/poster.jpg) | [Code download](https://github.com/NJUyued/RDA4RobustSSL/archive/refs/heads/master.zip)]  | [PyTorch Repo](https://github.com/NJUyued/RDA4RobustSSL/)]
 - Latest news:
     - Our paper is accepted by **European Conference on Computer Vision (ECCV) 2022** 🎉🎉. Thanks to users. 
 - Related works:
     - 🆕 Interested in the conventional SSL or more application of complementary label in SSL? 👉 Check out our TNNLS paper **MutexMatch** [[arXiv](https://arxiv.org/abs/2203.14316) | [Repo](https://github.com/NJUyued/MutexMatch4SSL/)].

<div align=center>

<img width="750px" src="/figures/framework.jpg"> 
 
</div>


## Introduction

**Reciprocal Distribution Alignment (RDA) is a semi-supervised learning (SSL) framework working with both the matched (conventionally) and the mismatched class distributions.** Distribution mismatch is an often overlooked but more general SSL scenario where the labeled and the unlabeled data do not fall into the identical class distribution. This may lead to the model not exploiting the labeled data reliably and drastically degrade the performance of SSL methods, which could not be rescued by the traditional distribution alignment. RDA achieves promising performance in SSL under a variety of scenarios of mismatched distributions, as well as the conventional matched SSL setting.
