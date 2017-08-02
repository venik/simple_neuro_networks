#!/bin/bash

# Sasha Nikiforov nikiforov.al@gail.com
# fetch Census data set from https://archive.ics.uci.edu/ml/datasets/Census+Income

clean() {
    rm -vrf adult* old.adult.names
}

fetch() {
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names
}

clean
fetch
