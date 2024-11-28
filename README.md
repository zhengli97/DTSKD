# Dual Teachers for Self-Knowledge Distillation

> **Dual Teachers for Self-Knowledge Distillation** <br>
> Zheng Li, Xiang Li, Lingfeng Yang, Renjie Song, Jian Yang#, Zhigeng Pan. <br>
> Nankai University<br>
> Pattern Recognition 2024 <br>
> [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320324001730)] [[中文解读](https://zhuanlan.zhihu.com/p/690877571)]


## Abstract

We introduce an efficient self-knowledge distillation framework, Dual Teachers for Self-Knowledge Distillation (DTSKD), 
where the student receives self-supervisions by dual teachers from two substantially different fields, 
i.e., the past learning history and the current network structure. 
Specifically, DTSKD trains a considerably lightweight multi-branch network and acquires predictions from each, 
which are simultaneously supervised by a historical teacher from the previous epoch and a structural teacher under the current iteration. 


## Implementation

### Requirements
- Python3
- Pytorch >=1.7.0
- torchvision >= 0.8.1
- numpy >=1.18.5
- tqdm >=4.47.0

### Training 

In this code, you can reproduce the experiment results of classification task on CIFAR-100.
For example:
- Running TESKD for ResNet18 on CIFAR-100 dataset. 

(Running based on one NVIDIA Titan XP GPU. It requires at least 8GB memory.)

~~~
python3 main.py --lr 0.1 \
                --lr_decay_schedule 150 225 \
                --HSKD 1\
                --experiments_dir 'The directory name where the model files are stored' \
                --experiments_name 'Experiment name' \
                --classifier_type 'resnet18_dtkd' \
                --data_path 'The directory where the CIFAR-100 dataset is located' \
                --data_type 'cifar100' \
                --backbone_weight 3.0 \
                --b1_weight 1.0 \
                --b2_weight 1.0 \
                --b3_weight 1.0 \
                --ce_weight 0.2 \
                --kd_weight 0.8 \
                --coeff_decay 'cos' \
                --cos_max 0.9 \
                --cos_min 0.0 \
                --rank 0 \
                --tsne 0 \
                --world_size 1
~~~


## Contact

If you have any questions, you can submit an [issue](https://github.com/zhengli97/DTSKD/issues) on GitHub, leave a message on [Zhihu Article](https://zhuanlan.zhihu.com/p/690877571) (if you can speak Chinese), or contact me by email (zhengli97[at]qq.com).

## Citation

If you find our paper or repo helpful for your research, please consider citing our paper and giving this repo a star⭐. Thank you!

```
@article{li2024dual,
  title={Dual teachers for self-knowledge distillation},
  author={Li, Zheng and Li, Xiang and Yang, Lingfeng and Song, Renjie and Yang, Jian and Pan, Zhigeng},
  journal={Pattern Recognition},
  volume={151},
  pages={110422},
  year={2024},
  publisher={Elsevier}
}
```