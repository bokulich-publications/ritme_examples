# ritme_examples
This repos contains fully reproducible workflows to reproduce the use cases accompanying the ritme publication.

## Setup
To run the notebooks in `use_cases` we suggest you set up a conda environment as follows:

* Install [mamba](https://github.com/mamba-org/mamba)
* Create and activate a conda environment with the required dependencies:
```shell
mamba env create -f environment.yml
conda activate ritme_examples
conda install -c https://packages.qiime2.org/qiime2/2024.5/metagenome/released/ -c conda-forge -c bioconda -c defaults q2-fondue -y
pip install -e .
qiime dev refresh-cache
```
* Run the `vdb-config` tool and exit by pressing x (needed to initialize the wrapped SRA Toolkit for more information see [here](https://github.com/ncbi/sra-tools/wiki/05.-Toolkit-Configuration))
```shell
vdb-config -i
```

## Contact

In case of questions or comments feel free to raise an issue in this repository.


## License

This repository  is released under a BSD-3-Clause license. See LICENSE for more details.
