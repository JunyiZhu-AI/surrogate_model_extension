## Surrogate Model Extension (SME): A Fast and Accurate Weight Update Attack on Federated Learning [Accepted at ICML 2023]

### Abstract
In Federated Learning (FL) and many other distributed training frameworks, collaborators can hold their private data locally and only share the network weights trained with the local data after multiple iterations. Gradient inversion is a family of privacy attacks that recovers data from its generated gradients. Seemingly, FL can provide a degree of protection against gradient inversion attacks on weight updates, since the gradient of a single step is concealed by the accumulation of gradients over multiple local iterations. In this work, we propose a principled way to extend gradient inversion attacks to weight updates in FL, thereby better exposing weaknesses in the presumed privacy protection inherent in FL. In particular, we propose a surrogate model method based on the characteristic of two-dimensional gradient flow and low-rank property of local updates. Our method largely boosts the ability of gradient inversion attacks on weight updates containing many iterations and achieves state-of-the-art (SOTA) performance. Additionally, our method runs up to $100\times$ faster than the SOTA baseline in the common FL scenario. Our work re-evaluates and highlights the privacy risk of sharing network weights.

<p align="center">
      <img width="902" height="291" src=".illustration.png" alt>
</p>
<p align="center">
    <em>Figure 1: An overview of our confidence-aware personalized federated learning framework.</em>
</p>

<p align="center">
      <img width="1000" height="463" src=".visualization.png" alt>
</p>
<p align="center">
    <em> Figure 2: Visualization of the reconstructed images. </em>
</p>


### Download
Make sure that conda is installed.
```sh
git clone git@github.com:JunyiZhu-AI/surrogate_model_extension.git
cd surrogate_model_extension
conda create -n sme python==3.9.12
conda activate sme
conda install pip
pip install -r requirement.txt
```
Next, prepare the FEMNIST dataset.
```sh
mkdir data
cd data
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
cd ../../../
mv leaf/data/femnist ./
rm -rf leaf
cd ../
```

### Run
To run the experiments, follow these instructions:

1. For label distribution skew on CIFAR-10:
```sh
python3 -m experiment.run_experiment --config experiment/configs/cifar10.json
```

2. For label distribution skew on CIFAR-10 with 5,000 data points:
```sh
python3 -m experiment.run_experiment --config experiment/configs/cifar10_data5000.json
```

3. For label concept drift on CIFAR-100:
```sh
python3 -m experiment.run_experiment --config experiment/configs/cifar100.json
```

4. Example of running an experiment by passing arguments directly (the following hyperparameters are not tuned and are for demonstration purposes only):
```sh
python3 -m experiment.run_experiment \
      --model CNNCifar \
      --dataset CIFAR10 \
      --batch_size 10 \
      --lr_head 0.001 \
      --lr_base 0.001 \
      --momentum 0.9 \
      --n_labels 5 \
      --head_epochs 10 \
      --base_epochs 5 \
      --n_rounds 100 \
      --max_data 20000 \
      --n_clients 50 \
      --sampling_rate 0.1 \
      --seed 42 \
      --n_mc 1 \
      --beta 0.9 \
      --scale 1
```

### Modify
The explanation of hyperparameters can be found in the ```experiment\run_experiment.py``` file. Our method employs a head-base architecture, making it easily adaptable to other types of networks. If you wish to modify the network, we recommend fine-tuning the hyperparameters. In our experience, it is efficient to use the hyperparameters of Federated Averaging (FedAvg) for the base network, while only tuning the head network specifically. However, full grid search can often obtain better performance.

### Citation
```
@inproceedings{Zhu2023a,
  TITLE = {Confidence-aware Personalized Federated Learning via Variational Expectation Maximization},
  AUTHOR = {Zhu, Junyi and Ma, Xingchen and Blaschko, Matthew B.},
  BOOKTITLE = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  YEAR = {2023},
}
```
