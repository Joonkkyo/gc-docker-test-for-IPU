# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import copy
from PIL import Image
from pathlib import Path
import poptorch
import numpy as np
import logging
import torch
import base64
import horovod.torch as hvd
import io
import import_helper
from torchvision import transforms
from loguru import logger

transform_val = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def inference(model, data):
    logger.info("inference resnet model")
    seed = data.seed
    binary_image = data.binary_image
    set_seed(seed)
    buf = io.BytesIO(base64.b64decode(binary_image))
    image = Image.open(buf)
    data_img = transform_val(image).unsqueeze(0)
    outputs = model(data_img)
    _, result = torch.max(outputs, 1)
    candidate = int(result)

    return {'candidate': candidate}