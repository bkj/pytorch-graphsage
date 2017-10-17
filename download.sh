#!/bin/bash

# download.sh
#
# download datasets

# --
# Download

mkdir -p data
cd data

wget http://snap.stanford.edu/graphsage/reddit.zip
unzip reddit.zip && rm reddit.zip

wget http://snap.stanford.edu/graphsage/ppi.zip
unzip ppi.zip && rm ppi.zip

# --
# Change names

cd reddit
rename 's/reddit-//' *
cd ..

cd ppi
rename 's/ppi-//' *
cd ..