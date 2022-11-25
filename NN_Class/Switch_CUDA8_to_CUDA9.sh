sudo rm -rf /usr/local/cuda
sudo rm -rf /usr/local/cuda/bin/gcc
sudo rm -rf /usr/local/cuda/bin/g++
sudo ln -sf /usr/local/cuda-9.0 /usr/local/cuda
sudo ln -sf /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
sudo ln -sf /usr/bin/g++-6 /usr/local/cuda/bin/g++

