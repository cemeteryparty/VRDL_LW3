all:
	git clone https://github.com/cemeteryparty/VRDL_LW3.git
	python3 VRDL_LW3/tools/gdget.py \
		1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
	unzip -qq dataset.zip -d ./
	rm -rf dataset.zip VRDL_LW3
build:
	mv VRDL_LW3/nucleus/nucleus.py ./
	mv VRDL_LW3/mrcnn/ ./
clean:
	rm nucleus.py
	rm -r dataset
	rm -rf mrcnn

