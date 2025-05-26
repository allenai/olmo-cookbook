#!/bin/bash

set -ex


# Install aria2 for parallel downloads
RELEASE=1.37.0

wget https://github.com/aria2/aria2/releases/download/release-$RELEASE/aria2-$RELEASE.tar.gz
tar -xzvf aria2-$RELEASE.tar.gz

sudo yum install gcc-c++ openssl-devel libxml2-devel -y

pushd aria2-$RELEASE
./configure
make -j$(nproc)
sudo make install
popd

# clean up
rm -rf aria2-$RELEASE.tar.gz

# # # # # # # # # # # # # # # # # # # #

# Install hfd for faster HuggingFace downloads

wget https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/6891c4b02f5cf3d014c7b1523556e15d9a3dd00f/hfd.sh
chmod a+x hfd.sh
sudo mv hfd.sh /usr/local/bin/hfd
