#!/bin/bash

# setup rdsdev env
ENVIRONMENT=ml4risk
conda create -y -n $ENVIRONMENT python=3.7
conda install nb_conda_kernels ipykernel --name $ENVIRONMENT --yes

# install packages
source activate "$ENVIRONMENT"
pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt --use-deprecated=legacy-resolver

# pip install --no-use-pep517 pyarrow
conda deactivate

# install kernel
source activate JupyterSystemEnv
python -m ipykernel install --user --name $ENVIRONMENT --display-name "$ENVIRONMENT"
conda deactivate