#!/usr/bin/env bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

RESOURCES_DIR='/root/inference/utils/resources/'
GET_FILE='get.py'
IMAGES='images224.tar.gz'

python3 "${RESOURCES_DIR}${GET_FILE}" ${IMAGES}
echo "Unpacking ${IMAGES}"
mkdir -p ../data
tar xzf ${IMAGES} -C ../data
rm ${IMAGES}