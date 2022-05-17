<!-- 
SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
 -->

# Quickstart

You can follow these steps to quickly recreate the models covered in the tutorial.

## Step 1 - Train a PyTorch model on the CIFAR10 dataset

Execute the following command on a machine with an NVIDIA GPU.

```bash
python3 train.py model_bn --checkpoint_path=data/model_bn.pth
```

## Step 2 - Export the trained model to ONNX

Execute the following command on a machine with an NVIDIA GPU.

```bash
python3 export.py model_bn data/model_bn.onnx --checkpoint_path=data/model_bn.pth
```

> Tip: Once exported to ONNX, the models can be profiled using the ``trtexec`` tool as described in [TUTORIAL.md](TUTORIAL.md)

## Step 3 - Build the TensorRT engine

Execute the following command on a machine with an NVIDIA GPU.  To use the DLA, you must call this on a machine with a DLA, like Jetson Orin.

```bash
python3 build.py data/model_bn.onnx --output=data/model_bn.engine --int8 --dla_core=0 --gpu_fallback --batch_size=32
```

> 

## Step 4 - Evaluate the model on the CIFAR10 test dataset

Execute the following command on a machine with an NVIDIA GPU.  You must call this on 
the same machine that you called ``build.py``.

```bash
python3 eval.py data/model_bn.engine --batch_size=32
```