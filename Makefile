all:
    pip install torch
    pip install torchvision

    # detectron2 repo
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2; pip install -e .

    # Other library
    pip install opencv-python
    pip install scikit-learn
    pip install scikit-image
    pip install tensorflow
    pip install pandas
    pip install clodsa
