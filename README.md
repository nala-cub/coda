<div align="center">

<h1>The World of an Octopus: How Reporting Bias Influences a Language Model's Perception of Color</h1>

</div>

## Overview

Code and dataset for *The World of an Octopus: How Reporting Bias Influences a Language Model's Perception of Color*.

This repository is roughly split into 2 parts:

- [`probing`](/projects/coda/probing): The probing implementations, including code for generating CoDa.
- [`mturk-survey`](/projects/coda/mturk-survey): Instruction pages and used for crowdsourcing annotations.

## How to use

### Using CoDa

If you'd like to use CoDa, we highly recommend using the version hosted on the [Huggingface Hub](https://huggingface.co/datasets/corypaik/coda) as it requires no additional dependencies.

```python
from datasets import load_dataset

ds = load_dataset('corypaik/coda')
```

You can find more details about how to use Huggingface Datasets [here](https://github.com/huggingface/datasets).

### Running experiments

This repository is developed and tested on linux systems and uses [Bazel](https://docs.bazel.build/versions/4.1.0/install.html). If you are on other platforms, you might consider running Bazel in a docker container. If you'd like more guidance on this, please open an Issue on [GitHub](https://github.com/nala-cub/coda/issues/new).

First, clone the project

```bash
# clone project
git clone https://github.com/nala-cub/coda

# goto project
cd coda
```

You can run the specific tasks as:

```bash
# run zeroshot
bazel run //projects/coda/probing/zeroshot
# representation probing
bazel run //projects/coda/probing/representations
# ngrams
bazel run //projects/coda/probing/ngram_stats
# generate dataset from annotations (relative to workspace root)
bazel run //projects/coda/probing/dataset:create_dataset -- \
  --coda_ds_export_dir=<export_dir>
```

To see help for any of the commands, use:

```bash
bazel run <target> -- --help
# for example:
# bazel run //projects/coda/probing/zeroshot -- --help
```

### Annotation Instructions

Annotations were collected using an Angular app on Firebase. The included files contain all instructions, but not the app itself. If you're interested in the latter please open an issue on [GitHub](https://github.com/nala-cub/coda/issues/new).

## Citation

If this code was useful, please cite the paper:

```
@misc{paik2021world,
      title={The World of an Octopus: How Reporting Bias Influences a Language Model's Perception of Color},
      author={Cory Paik and Stéphane Aroca-Ouellette and Alessandro Roncone and Katharina Kann},
      year={2021},
      eprint={2110.08182},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

CoDa is licensed under the Apache 2.0 license. The text of the license can be found [here](/LICENSE).
