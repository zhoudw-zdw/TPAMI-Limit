
# Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks  (LIMIT)

The code repository for ["Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks"](https://ieeexplore.ieee.org/document/9864267) (TPAMI 2022) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

       
    @ARTICLE{zhou2022few,
    author={Zhou, Da-Wei and Ye, Han-Jia and Ma, Liang and Xie, Di and Pu, Shiliang and Zhan, De-Chuan},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks}, 
    year={2022},
    volume={},
    number={},
    pages={1-16},
    doi={10.1109/TPAMI.2022.3200865}}



## Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks


New classes arise frequently in our ever-changing world, e.g., emerging topics in social media and new types of products in e-commerce. A model should recognize new classes and meanwhile maintain discriminability over old classes. Under severe
circumstances, only limited novel instances are available to incrementally update the model. The task of recognizing few-shot new classes without forgetting old classes is called few-shot class-incremental learning (FSCIL). In this work, we propose a new paradigm for FSCIL based on meta-learning by LearnIng Multi-phase Incremental Tasks (LIMIT), which synthesizes fake FSCIL tasks from the base dataset. The data format of fake tasks is consistent with the ‘real’ incremental tasks, and we can build a generalizable feature space for the unseen tasks through meta-learning. Besides, LIMIT also constructs a calibration module based on transformer, which calibrates the old class classifiers and new class prototypes into the same scale and fills in the semantic gap. The calibration module also adaptively
contextualizes the instance-specific embedding with a set-to-set function. LIMIT efficiently adapts to new classes and meanwhile resists forgetting over old classes. Experiments on three benchmark datasets (CIFAR100, miniImageNet, and CUB200) and large-scale dataset, i.e., ImageNet ILSVRC2012 validate that LIMIT achieves state-of-the-art performance.

<img src='imgs/teaser.png' width='1000' height='300'>

## Results
<img src='imgs/result.png' width='900' height='778'>

Please refer to our [paper](https://arxiv.org/abs/2203.17030) for detailed values.

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- tqdm

- Download the [pretrained models](https://drive.google.com/drive/folders/11liacLA6F4uz_SWAPLYQshlzLMXMzDkB?usp=sharing) and put them in ./params. Note that these pre-trained models are only trained with cross-entropy with the base dataset, and it should be distinguished from the models pre-trained on large-scale datasets.

## Dataset
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.

The split of ImageNet100/1000 is availabel at [Google Drive](https://drive.google.com/drive/folders/1IBjVEmwmLBdABTaD6cDbrdHMXfHHtFvU?usp=sharing).

## Code Structures
There are four parts in the code.
 - `models`: It contains the backbone network and training protocols for the experiment.
 - `data`: Images and splits for the data sets.
- `dataloader`: Dataloader of different datasets.
 - `checkpoint`: The weights and logs of the experiment.
 - `params`: Pretrained model weights.
 
## Training scripts

- Train CIFAR100

  ```
    python train.py -project limit -dataset cifar100 -epochs_base 20  -lr_base 0.0002 -lrg 0.0002  -gamma 0.3 -gpu 3 -model_dir ./params/pretrain_CIFAR.pth -temperature 16 -schedule Milestone -milestones 2 4 6 -num_tasks 32 >>cifar.txt
  ```
  
- Train CUB200
    ```
    python train.py -project limit -dataset cub200 -epochs_base 40  -lr_base 0.0002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 2 -model_dir ./params/pretrain_CUB.pth  -dataroot YOURDATAROOT -num_tasks 32 >>cub.txt
    ```

- Train miniImageNet
    ```
    python train.py -project limit -dataset mini_imagenet -epochs_base 20  -lr_base 0.0002 -lrg 0.0002 -gamma 0.3 -gpu 3 -model_dir ./params/pretrain_MINI.pth  -dataroot YOURDATAROOT -num_tasks 32 -temperature 0.5 -schedule Milestone -milestones 3 6 9 12 >>mini.txt
    ```

Remember to change `YOURDATAROOT` into your own data root, or you will encounter errors.

  

 
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [Awesome Few-Shot Class-Incremental Learning](https://github.com/zhoudw-zdw/Awesome-Few-Shot-Class-Incremental-Learning)

- [PyCIL: A Python Toolbox for Class-Incremental Learning](https://github.com/G-U-N/PyCIL)

- [Fact](https://github.com/zhoudw-zdw/CVPR22-Fact)

- [FEAT](https://github.com/Sha-Lab/FEAT)

- [Castle](https://github.com/Sha-Lab/aCASTLE)

- [CEC](https://github.com/icoz69/CEC-CVPR2021)



## Contact 
If there are any questions, please feel free to contact with the author: Da-Wei Zhou (zhoudw@lamda.nju.edu.cn). Enjoy the code.
