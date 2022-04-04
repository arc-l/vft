# Visual Foresight Trees for Object Retrieval from Clutter with Nonprehensile Rearrangement

**Paper:** https://arxiv.org/abs/2105.02857

**Videos:**

[![IMAGE ALT TEXT](http://img.youtube.com/vi/7cL-hmgvyec/sddefault.jpg)](https://www.youtube.com/watch?v=7cL-hmgvyec "VFT")

**Citation:**
If you use this code in your research, please cite the paper:

```
@ARTICLE{huang2021visual,
  author={Huang, Baichuan and Han, Shuai D. and Yu, Jingjin and Boularias, Abdeslam},
  journal={IEEE Robotics and Automation Letters}, 
  title={Visual Foresight Trees for Object Retrieval From Clutter With Nonprehensile Rearrangement}, 
  year={2022},
  volume={7(1)},
  pages={231-238},
  doi={10.1109/LRA.2021.3123373}
}
```

## Installation
We recommand [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
```shell
git clone https://github.com/arc-l/vft.git
cd vft
conda env create -n vft --file env-vft.yml
conda activate vft
pip install graphviz
```
or 
```shell
git clone https://github.com/arc-l/vft.git
cd vft
conda env create -n vft --file env-vft-cross.yml
conda activate vft
```

## Quick Start (Simulation)
1. Download models (download folders and unzip) from [Google Drive](https://drive.google.com/drive/folders/1mqP3qgUoYHCaHfsW8jkA4kFWKuAnMoQ_?usp=sharing) and put them in `vft` folder
2. `bash mcts_main_run.sh`

# Training networks
This paper shares many common code to https://github.com/arc-l/dipn. Except the environment was changed from CoppeliaSim (V-REP) to PyBullet.

## Acknowledgement
The part of simulation environment was adapted from https://github.com/google-research/ravens
