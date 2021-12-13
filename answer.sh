#!/bin/bash

#python3 tools/gdget.py 1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
#unzip -qq dataset.zip -d ./
#rm -rf dataset.zip
python3 inference.py --model-path best_total_loss_model.pth \
	--testset-path dataset/test --testset-info dataset/test_img_ids.json \
	--output-path ./ --config mrcnn_R50_config.yaml

