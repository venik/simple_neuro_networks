#!/bin/bash

# Sasha Nikiforov nikiforov.al@gail.com
# fetch MNIST DB from the Yann Lecun site

clean() {
    rm -vrf t10k-* train-*
}

fetch() {
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
}

unpack () {
    gzip -vd *.gz
}

clean
fetch
unpack
