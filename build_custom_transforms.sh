#!/bin/bash
if [ -z "$1" ]; then
    echo "Path to libtorch is unset or set to the empty string"
    exit 1
fi

echo "$1"

cd transforms/dilate
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=$1 ..
make -j

cd ../../erode
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=$1 ..
make -j

cd ../../rotate
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=$1 ..
make -j

cd ../../scale
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=$1 ..
make -j

cd ../../translate
mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_PREFIX_PATH=$1 ..
make -j

