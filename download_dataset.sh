#!/bin/bash
mkdir dataset
cd dataset

# Download COCO dataset
wget -O train2014.zip http://images.cocodataset.org/zips/train2014.zip
wget -O val2014.zip http://images.cocodataset.org/zips/val2014.zip

unzip train2014.zip
unzip val2014.zip

rm train2014.zip
rm val2014.zip

cd ..