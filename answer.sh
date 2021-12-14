#!/bin/bash

python3 tools/gdget.py 1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
unzip -qq dataset.zip -d ./
rm -rf dataset.zip
python3 inference.py --model-path detectron2_nuc_model.pth \
	--testset-path dataset/test --testset-info dataset/test_img_ids.json \
	--output-path results --config mrcnn_R50_config.yaml

