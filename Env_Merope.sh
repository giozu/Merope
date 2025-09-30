#!/bin/bash
# 
# Marc Josien
#
# Environment for Merope librairies

# 1) Source Mérope Paths
# Get the path of this script (assumed to be in the root)
pushd . > /dev/null
SCRIPT_PATH="${BASH_SOURCE[0]}"
if ([ -h "${SCRIPT_PATH}" ]); then
  while([ -h "${SCRIPT_PATH}" ]); do cd `dirname "$SCRIPT_PATH"`;
  SCRIPT_PATH=`readlink "${SCRIPT_PATH}"`; done
fi
cd `dirname ${SCRIPT_PATH}` > /dev/null
SCRIPT_PATH=`pwd`;
popd  > /dev/null

# get the environment compiler of the installation
source ${SCRIPT_PATH}/Installation/Install_environment.sh
# Merope &  RSA_algo
export PYTHONPATH=${SCRIPT_PATH}/INSTALL-DIR/lib:$PYTHONPATH
# Scripts
export PATH=$PATH:${SCRIPT_PATH}/scripts
### tools
export PYTHONPATH=${SCRIPT_PATH}/tools/python/:$PYTHONPATH

# 2) Source AMITEX-FFTP and/or TMFFT
# AMITEX_FOLDER=/usr/lib/amitex_fft-v8.17.8
AMITEX_FOLDER=/usr/lib/amitex_fftp-v8.17.14
# amitex si scarica da internet, lo si copia e incolla in una cartella
# poi si installa eseguendo ./install (prima si deve rendere eseguibile ./install e anche altre cose tipo)
# # go into FoX and fix configure
# cd /usr/lib/amitex_fftp-v8.17.14/lib_extern/FoX-4.1.2_modif2018
# sudo chmod +x configure

# # go back to the top-level dir and fix the clean scripts
# cd /usr/lib/amitex_fftp-v8.17.14
# sudo chmod +x clean_results.sh clean_results_PF.sh clean_all.sh


### use amitex_fftp
export PATH=${AMITEX_FOLDER}/libAmitex/bin:$PATH
export LD_LIBRARY_PATH=${AMITEX_FOLDER}/libAmitex/src/materiauxK:$LD_LIBRARY_PATH
source ${AMITEX_FOLDER}/env_amitex.sh

