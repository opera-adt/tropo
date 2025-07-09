# OPERA TROPO

Tropospheric Delay Estimation workflow for global OPERA TROPO products.

Creates the science application software (SAS) using the [RAiDER](https://github.com/dbekaert/RAiDER) library.

## Development setup


### Prerequisite installs
1. Download source code:
```bash
git clone https://github.com/opera-adt/opera_tropo.git
```
2. Install dependencies, either to a new environment:
```bash
mamba env create --name opera_tropo --file tropo/conda-env.yml
conda activate opera_tropo
```
or install within your existing env with mamba.

3. Install `tropo` via pip in editable mode
```bash
python -m pip install --no-deps -e  tropo
```

### Usage

There are 5 entrypoints for the OPERA-TROPO workflow

1. Download HRES model *.nc from s3 bucket to local directory
```bash
opera_tropo download -s3 "bucket_path" --date 20190613 --hour 06
```
2. Run troposphere phase delay estimation, require configuration file
   default configs can be found in opera_tropo/config/default
```bash
opera_tropo config -input input_data/D06130600061306001.zz.nc -out output/ -c runconfig.yaml
```

3. Run troposphere phase delay estimation, require configuration file
   default configs can be found in opera_tropo/config/default
   NOTE: processing datetime is changing for each output filename
```bash
opera_tropo run runconfig.yaml
```

4. Make browser image. NOTE. browse-image is created druing run routine
   Default height level 800m, plot image at different level
```bash
opera_tropo make-browse -i OPERA_L4_TROPO_GLOBAL_20190101T000000Z_20250130T232942Z_HRES_0.1_v0.1.nc --height 100
```

5. Validate: golden_dataset vs output.
```bash
opera_tropo validate OPERA_L4_TROPO_20190613T060000Z_20250206T182940Z_HRES_0.1_v0.1.nc output/OPERA_L4_TROPO_20190613T060000Z_20250206T201820Z_HRES_0.1_v0.1.nc
```

### Setup for contributing


We use [pre-commit](https://pre-commit.com/) to automatically run linting, formatting, and [mypy type checking](https://www.mypy-lang.org/).
Additionally, we follow [`numpydoc` conventions for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
To install pre-commit locally, run:

```bash
pre-commit install
```
This adds a pre-commit hooks so that linting/formatting is done automatically. If code does not pass the checks, you will be prompted to fix it before committing.
Remember to re-add any files you want to commit which have been altered by `pre-commit`. You can do this by re-running `git add` on the files.

Since we use [black](https://black.readthedocs.io/en/stable/) for formatting and [flake8](https://flake8.pycqa.org/en/latest/) for linting, it can be helpful to install these plugins into your editor so that code gets formatted and linted as you save.

### Running the unit tests

After making functional changes and/or have added new tests, you should run pytest to check that everything is working as expected.

First, install the extra test dependencies:
```bash
python -m pip install --no-deps -e .[test]
```

Then run the tests:

```bash
pytest
```


### Building the docker image

To build the docker image, run:
```bash
docker build -f docker/Dockerfile -t opera_tropo .
```
which will print out instructions for running the image.

or
```bash
./docker/build-docker-image.sh --tag my-tag
```
#### Running the docker image
Run the built docker image with:
```
docker run \
  --rm \
  --user $(id -u):$(id -g) \
  --volume <local host directory>:/home/ops \
  opera_tropo:latest \
  opera_tropo run runconfig.yaml

```
