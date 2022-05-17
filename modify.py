# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import argparse
import onnx_graphsurgeon as gs
import onnx


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to the ONNX model.')
parser.add_argument('output', type=str, help='Path to output the modified ONNX model.')
args = parser.parse_args()

graph = gs.import_onnx(onnx.load(args.input))

for node in graph.nodes:
    if node.op == 'GlobalAveragePool':
        node.op = 'AveragePool'
        node.attrs['kernel_shape'] = [2, 2]

onnx.save(gs.export_onnx(graph), args.output)