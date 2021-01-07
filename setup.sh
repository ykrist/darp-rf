#!/bin/bash -i
set -e
CONDA_ENV="darp"
cp config-utils.yaml pkgs/phd-utils/config.yaml
echo "Creating Conda environment..."
conda env create -f env.yml
echo "Created Conda environment \"${CONDA_ENV}\""
echo "installing pkgs/oru"
export CONDA_ENV
( conda activate $CONDA_ENV && cd pkgs/oru && ./bin/conda-pkg-setup install -y )
echo "installing pkgs/phd-utils"
( conda activate $CONDA_ENV && cd pkgs/phd-utils && conda-pkg-setup install -y )

