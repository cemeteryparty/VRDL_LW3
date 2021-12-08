all:
	git clone https://github.com/cemeteryparty/VRDL_LW3.git
	python3 VRDL_LW3/tools/gdget.py \
		1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
	unzip -qq dataset.zip -d ./
	rm -rf dataset.zip VRDL_LW3
install:
	#pip install tensorflow==1.13.1
	#pip install keras==2.0.8
	#pip install h5py==2.10.0
	#git clone https://github.com/cemeteryparty/VRDL_LW3.git
build:
	mv VRDL_LW3/nucleus/nucleus.py ./
	mv VRDL_LW3/mrcnn/ ./
clean:
	rm nucleus.py
	rm -r dataset
	rm -rf mrcnn

