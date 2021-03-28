<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## CVPR'21 Weakly Supervised Action Selection Learning in Video
[[paper]()]

Authors: [Jeremy Junwei Ma](https://scholar.google.com/citations?user=LyoH1SMAAAAJ&hl=en), Satya Krishna Gorti, Guangwei Yu, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

<a name="intro"/>

## Introduction
This repository contains the implementation of ASL (Action Selection Learning) on the Thumos 14' dataset.

<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* Python 3.6
* Pytorch 1.4.0

Experiments on Thumos 14' dataset were run on a single NVIDIA TITAN V GPU with 12 GB GPU memory.

<a name="dataset"/>

## Dataset

1. Download the Thumos 14' dataset [here](http://crcv.ucf.edu/THUMOS14/download.html)

2. In `./scripts/train.sh` and `./scripts/inference.sh`, replace the `--data_path` argument with the downloaded Thumos 14' path

## Training

```
    bash ./scripts/train.sh
```

## Inference

```
    bash ./scripts/inference.sh
```
