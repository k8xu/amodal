## Amodal Completion via Progressive Mixed Context Diffusion (CVPR 2024)
### [Project Page](https://k8xu.github.io/amodal/) | [Paper](https://arxiv.org/pdf/2312.15540.pdf) | [arXiv](https://arxiv.org/abs/2312.15540) | [Bibtex](#bibtex)

[Katherine Xu](https://k8xu.github.io)$^{1}$, [Lingzhi Zhang](https://owenzlz.github.io)$^{2}$, [Jianbo Shi](https://www.cis.upenn.edu/~jshi)$^1$<br>
$^1$ University of Pennsylvania, $^2$ Adobe Inc.

![teaser](images/teaser.png)
Our method can recover the hidden pixels of objects in diverse images. Occluders may be co-occurring (a person on a surfboard), accidental (a cat in front of a microwave), the image boundary (giraffe), or a combination of these scenarios.
The pink outline indicates an occluder object.

We use pretrained diffusion inpainting models, and no additional training is required!


## 🚀 Updates
- Stay tuned for our code release!


## Table of Contents
* [Requirements](#requirements)
* [Setup](#setup)
* [Citation](#citation)


## Requirements
* Python 3.10
* Docker


## Setup

1. Clone this `amodal` repository, and run `cd Grounded-Segment-Anything`.

2. In the Dockerfile, change all instances of `/home/appuser` to your path for the `amodal` repository.

3. Run `make build-image`.

4. Start and attach to a docker container from the image `gsa:v0`. Then, navigate to the `amodal` repository.

5. Run `./install.sh` to finish setup and download model checkpoints.


## Dataset

1. Run `./download_dataset.sh` to download the COCO dataset.


## Usage

### Progressive Occlusion-aware Completion Pipeline

1. In `./main.sh`, modify `input_dir` to your folder path for the images.

2. Run `./main.sh`. You may need to use `chmod` to change the file permissions first.


## Citation

If you find our work useful, please cite our paper:
```
@inproceedings{xu2024amodal,
  title={Amodal completion via progressive mixed context diffusion},
  author={Xu, Katherine and Zhang, Lingzhi and Shi, Jianbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9099--9109},
  year={2024}
}
```
