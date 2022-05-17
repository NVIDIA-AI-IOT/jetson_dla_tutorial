# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import argparse
import torch
import os
from models import MODELS


parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--checkpoint_path', type=str, default=None)
args = parser.parse_args()

data = torch.zeros(1, 3, 32, 32).cuda()

model = MODELS[args.model_name]().cuda().eval()

if args.checkpoint_path is not None:
    model.load_state_dict(torch.load(args.checkpoint_path))

torch.onnx.export(
    model,
    data,
    args.output,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
