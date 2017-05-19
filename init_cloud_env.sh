#!/usr/bin/env bash

PKG_INSTALL='apt install '
PKG_LIST='gcc python3-dev virtualenv git'

# install packages
sudo $PKG_INSTALL $PKG_LIST

# init virtualenv
PYTHON=python3
VENV_FOLDER='./.venv_cloud'

PYTHON_EXEC=$(which $PYTHON 2>/dev/null | grep -v "not found" | wc -l)
if [ $PYTHON_EXEC -eq 0 ] ; then
    echo '[ERR] Cannot find '$PYTHON' Exit...'
    exit -1
fi

virtualenv $VENV_FOLDER --python=$PYTHON

source $VENV_FOLDER/bin/activate
pip install --upgrade -r ./.requirements_cloud
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp34-cp34m-linux_x86_64.whl
