# Spectral-Spatial Classification for Hyperspectral Image Based on a Single GRU



## Paper

[Spectral-Spatial Classification for Hyperspectral Image Based on a Single GRU](https://doi.org/10.1016/j.neucom.2020.01.029)

Please cite our paper if you find it useful for your research.

```
Bibtex: @article{pan2020spectral,
title={Spectral-spatial classification for hyperspectral image based on a single GRU},
author={Pan, Erting and Mei, Xiaoguang and Wang, Quande and Ma, Yong and Ma, Jiayi},
journal={Neurocomputing},
volume={387},
pages={150--160},
year={2020},
publisher={Elsevier}
}

@inproceedings{pan2019gru,
  title={GRU with spatial prior for hyperspectral image classification},
  author={Pan, Erting and Ma, Yong and Dai, Xiaobing and Fan, Fan and Huang, Jun and Mei, Xiaoguang and Ma, Jiayi},
  booktitle={IGARSS 2019-2019 IEEE International Geoscience and Remote Sensing Symposium},
  pages={967--970},
  year={2019},
  organization={IEEE}
}
```

## Installation

- Install `Tensorflow 1.9.0` with `Python 3.6`.

- Clone this repo

  `git clone https://github.com/EtPan/SPGRU`

## Usage

**1. Change the file path**

​	Replace the file path for the hyperspectral data in `save_indices.py` and `indices.py`;Replace the file path for the `ckpt` files of the model in `spgru.py` and `spgru_test.py`.

**2. Split the dataset**

​	Run `save_indices.py` .

**3. Training**

​	Run `spgru.py`.

**4. Testing and Evaluation**

​	Run `spgru_test.py`.

## Contact

panerting@whu.edu.cn 

