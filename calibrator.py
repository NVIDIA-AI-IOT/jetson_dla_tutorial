# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import torch
import tensorrt as trt


__all__ = [
    'DatasetCalibrator'
]

    
class DatasetCalibrator(trt.IInt8Calibrator):
    
    def __init__(self, 
            input, dataset, 
            algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        super(DatasetCalibrator, self).__init__()
        self.dataset = dataset
        self.algorithm = algorithm
        self.buffer = torch.zeros_like(input).contiguous()
        self.count = 0
        
    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):
            for buffer_idx in range(self.get_batch_size()):
                dataset_idx = self.count % len(self.dataset) # roll around if not multiple of dataset
                image, _ = self.dataset[dataset_idx]
                image = image.to(self.buffer.device)
                self.buffer[buffer_idx].copy_(image)

                self.count += 1
            return [int(self.buffer.data_ptr())]
        else:
            return []
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_batch_size(self):
        return int(self.buffer.shape[0])
    
    def read_calibration_cache(self, *args, **kwargs):
        return None
    
    def write_calibration_cache(self, cache, *args, **kwargs):
        pass