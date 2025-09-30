#!/bin/bash
#
# Environment definition for installation
#
###############################################################################################

export MEROPE_PYBIND_REPO=https://github.com/pybind/pybind11
export MEROPE_VOROPP_REPO=https://math.lbl.gov/voro++/download/
export EIGEN_REPO=https://gitlab.com/libeigen/eigen

echo "------------------------------"
echo "------------------------------"
echo "Please verify the location of MKL"
echo "------------------------------"
echo "------------------------------"
export MKL_ROOT_LIB="/usr/lib/x86_64-linux-gnu"
export MKL_INCLUDE_DIRS="/usr/include/mkl" 


VOROPP_NAME_DIR=voro-plus-plus




