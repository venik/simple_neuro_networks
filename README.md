# Simple Neural networks

src/one_layer_mnist/ - implementation of the 1 layer perceptron. it was trained against MNIST data set with 60k samples, checked against 10k samples with ~70% success

## Books that I've read and can recommend
[Cristiani-SVM] Nello Cristiani, John Shawe-Taylor Support Vector Machines and other kernel based learning methods. Comment will appear later, when I finish the book.

[Haykin-NNLM] S.Haykin Neural Networks and Learning Machines (3rd Edition) - quite classical from a famous author, has a lot of references. I think NND is better to start with neural networks

[ISLR] [An Introduction to Statistical Learning with Applications in R](http://www-bcf.usc.edu/~gareth/ISL/) - it's a really good book about statistical learning, doesnt require deep knowledge in linear agebra, yet very informative. Data sets are widly avaliable, examples are great, book is free.

[NND] [M.Hagan H.Demuth - Neural Network Design (2nd Edition)](http://hagan.okstate.edu/NNDesign.pdf) - extremally cheap (~25$ on Amazon) but super usefull, also contains introduction into linear algebra

## Prepare environemnt
1) Install python virtualenv
2) Init and activate virtual env
```
  # virtualenv ./.neuro -p python3
  # source ./.neuro/bin/activate
```
3) Install python libraries
```
  # pip install -r .requirements
```

Update environment
1) store python libraries list
```
  # pip freeze -r .requirements
```