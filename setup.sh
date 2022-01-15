#!/bin/bash
apt install python3.8-venv git -y
source venv/bin/activate

#pip3 install tqdm librosa tensorflow-io --use-feature=2020-resolver
apt update
apt install libsndfile1 ffmpeg llvm -q -y