#!/bin/bash

# download.sh
#
# download datasets

mkdir -p data
cd data

wget http://snap.stanford.edu/graphsage/reddit.zip
unzip reddit
cd reddit
rename 's/reddit-//' *