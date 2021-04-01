<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## CVPR'21 Weakly Supervised Action Selection Learning in Video
[[paper](http://www.cs.toronto.edu/~guangweiyu/pdfs/CVPR2021_asl.pdf)]

Authors: [Jeremy Junwei Ma*](https://scholar.google.com/citations?user=LyoH1SMAAAAJ&hl=en), [Satya Krishna Gorti*](http://www.cs.toronto.edu/~satyag/), [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs), [Guangwei Yu](http://www.cs.toronto.edu/~guangweiyu/)
<a name="intro"/>

## Introduction
This repository contains the implementation of ASL (Action Selection Learning) on the THUMOS-14 dataset.

<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* Python 3.6
* Pytorch 1.4.0

Experiments on THUMOS-14 dataset were run on a single NVIDIA TITAN V GPU with 12 GB GPU memory.

<a name="dataset"/>

## Dataset

1. Download the THUMOS-14 dataset [here](http://crcv.ucf.edu/THUMOS14/download.html)

2. In `./scripts/train.sh` and `./scripts/inference.sh`, replace the `--data_path` argument with the downloaded THUMOS-14 path

## Training

```
    bash ./scripts/train.sh
```

## Inference

```
    bash ./scripts/inference.sh
```

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{ma2021asl,
      title={Weakly Supervised Action Selection Learning in Video},
      author={Ma, Junwei and Gorti, Satya Krishna and Volkovs, Maksims and Yu, Guangwei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2021}
    }

