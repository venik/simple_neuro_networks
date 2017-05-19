#!/usr/bin/env bash

PKG_INSTALL='apt install '
PKG_LIST='gcc python3-dev git'

# install packages
sudo $PKG_INSTALL

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
pip install -r ./.requirements_cloud
