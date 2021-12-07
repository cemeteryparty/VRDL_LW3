all:
	python3 gdget.py 1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
	unzip -qq dataset.zip -d ./
	rm -f dataset.zip
	mkdir results logs
install:
	pip install tensorflow==1.13.1
	pip install keras==2.0.8
	pip install h5py==2.10.0
	pip install mrcnn
clean:
	rm -r dataset __pycache__

