# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import argparse
import os
import subprocess
from calibrator import DatasetCalibrator
import tensorrt as trt
import torch
import torchvision
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('onnx', type=str, help='Path to the ONNX model.')
parser.add_argument('--output', type=str, default=None, help='Path to output the optimized TensorRT engine')
parser.add_argument('--max_workspace_size', type=int, default=1<<25, help='Max workspace size for TensorRT engine.')
parser.add_argument('--int8', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dla_core', type=int, default=None)
parser.add_argument('--gpu_fallback', action='store_true')
parser.add_argument('--dataset_path', type=str, default='data/cifar10')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, 
    train=True,
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, 
    train=False,
    download=True, 
    transform=transform
)

data = torch.zeros(args.batch_size, 3, 32, 32).cuda()

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
builder.max_batch_size = args.batch_size
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(args.onnx, 'rb') as f:
    parser.parse(f.read())

profile = builder.create_optimization_profile()
profile.set_shape(
    'input',
    (args.batch_size, 3, 32, 32),
    (args.batch_size, 3, 32, 32),
    (args.batch_size, 3, 32, 32)
)

config = builder.create_builder_config()

config.max_workspace_size = args.max_workspace_size

if args.fp16:
    config.set_flag(trt.BuilderFlag.FP16)

if args.int8:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = DatasetCalibrator(data, train_dataset)

if args.dla_core is not None:
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = args.dla_core

if args.gpu_fallback:
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    
config.add_optimization_profile(profile)
config.set_calibration_profile(profile)

engine = builder.build_serialized_network(network, config)

if args.output is not None:
    with open(args.output, 'wb') as f:
        f.write(engine)

