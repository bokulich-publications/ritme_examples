# until package is not available via conda install it like this from within ritme repos:
mamba create -y -n ritme_model -c qiime2 -c conda-forge -c bioconda -c pytorch -c defaults $(python get_requirements.py ci/recipe/meta.yaml conda)

conda activate ritme_model

make dev
