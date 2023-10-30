[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/reasoningschema.svg?branch=main)](https://cirrus-ci.com/github/<USER>/reasoningschema)
[![ReadTheDocs](https://readthedocs.org/projects/reasoningschema/badge/?version=latest)](https://reasoningschema.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/reasoningschema/main.svg)](https://coveralls.io/r/<USER>/reasoningschema)
[![PyPI-Server](https://img.shields.io/pypi/v/reasoningschema.svg)](https://pypi.org/project/reasoningschema/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/reasoningschema.svg)](https://anaconda.org/conda-forge/reasoningschema)
[![Monthly Downloads](https://pepy.tech/badge/reasoningschema/month)](https://pepy.tech/project/reasoningschema)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/reasoningschema)
-->

# Nostalgia

> Add a short description here!

A longer description of your project goes here...

## Installation

In order to set up the necessary environment:

### __Virtualenv__

1. Install [virtualenv] and [virtualenvwrapper].
2. Create virtualenviroment for the project:

    ```bash
    mkvirtualenv nostalgia
    ```

3. Install the project in edit mode:

    ```bash
    python setup.py develop
    ```

Optional and needed only once after `git clone`:

1. install several [pre-commit] git hooks with:

   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```

   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

2. install [nbstripout] git hooks to remove the output cells of committed notebooks with:

   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```

   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.


## __Minimal Example__

### __Model Training__

To train a model, use the script `scripts/train_model.py`. Which model to train, specific parameters and data/output directories are provided by `config.yaml`
files. See section [__Config Files__](#config-files) for details.

Train a model from a config by `python scripts/train_model.py --config path/to/config.yaml`. We provide some default config files:

- `configs/train/llama2_commonsenseqa.yaml`:
  Fine-tune Llama2 model on the Commonsense QA dataset
- ...

Additional arguments to `scripts/train_model.py`:
- `--quiet / --verbose / --very-verbose`:
  Set log-level, i.e. number of logging messages shown during training.
- `--resume`: Use to resume a previous training.


### __Starting Tensorboard__

In the framework we provide out of the box tensorboard logging. In order to see the training progress in tensorboard, first you need to start the tensorboard:

```bash
tensorboard --logdir results/logging/tensorboard
```

![Tensorboard Logging Example](doc/source/images/tensorboard.png)

### __Config Files__

For reproducibility and tractability of the experiments done as well as for convenience we store all the models' hyperparameters into a __yaml__ config file.
All of the configuration files used to train the models are stored in `configs` folder. Each configuration file contains 5 main parts. The first part of the
yaml configuration file is:

```yaml
experiment:
  name: llama2-7B-CSQA-LoRA-D
  seed: [1]
  device_map: null # auto, cuda, cpu
```

- `name:` Key holds the name of the experiment. The user can use any name that finds suitable for the experiment.
- `num_works:` How many processes should be used for the training.
- `device_map:` Which gpus to be used. `null` will use GPU, `auto` will use all the GPUs.
- `seed:` Value of the initial seed.

The second part of the configuration file is the model.

```yaml
model:
  name: LLMCausal
  backbone: meta-llama/Llama-2-7b-chat-hf
  backbone_path: null # if null it uses the default cache path set for huggingface
  load_in_8bit: false
  load_in_4bit: false
  use_bf16: false # not sure if this will produce good results. Problem with grad scaling
  freeze_first_n_layers: -1
  peft:
    method: null # null, lora, ...
    r: 32
    lora_alpha: 32
    target_modules: !!python/tuple ["q_proj", "v_proj"]
    bias: none
    task_type: CAUSAL_LM
    lora_dropout: 0.05
    inference_mode: false
```

In this part the user can define which model will be used for the training as well as the hyperparameters. In the example above, we use __LLMCausal__ model.

The third part of the yaml file is the data loader part.

```yaml
dataset:
  name: commonsense_qa
  root_dir: null # if null it uses the default cache path set for huggingface
  batch_size: 16
  test_batch_size: 32
  num_workers: 2
  supervised: true # if false it appends "\nA: The answer is " else "\nA: The answer is <(a) correct answer>"
  prompt_path: null # prompts/commonsenseQA_SP.txt # path to a text file where the prompts are stored
  max_padding_length: 130
  output_fields: !!python/tuple ["input_ids", "attention_mask", "labels"]
  force_download: false
```

The forth part is the optimizer that we are going to use during the training.

```yaml
optimizers: !!python/tuple
  - optimizer: # name of the optimizer
      name: torch.optim.AdamW
      lr: 0.00005
      weight_decay: 0.0
      gradient_norm_clipping: 0.1
      schedulers: !!python/tuple
        - name: torch.optim.lr_scheduler.StepLR
          step_size: 10
          gamma: 0.8
```

The last part that we have to define is the _trainer_. In this part we set all the parameters that are used for training and logging.

```yaml
trainer:
  name: Trainer
  debug_iterations: null
  precision: bf16 # bf16_mixed, fp16_mixed, fp32_policy
  epochs: 20
  detect_anomaly: false
  gradient_accumulation_steps: 2
  save_every: 2
  best_metric: ppl
  logging_format: "RANK_%(rank)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s"
  experiment_dir: ./results/
  schedulers: !!python/tuple
    - name: reasoningschema.utils.param_scheduler.PeriodicScheduler # ExponentialIncrease, ConstantScheduler, PeriodicScheduler
      label: beta_scheduler_kl_rws
    - name: reasoningschema.utils.param_scheduler.ExponentialSchedulerGumbel
      label: temperature_scheduler_rws
      init_temperature: 1
      min_temperature: 0.5
      training_fraction_to_reach_min: 0.7
```

In this library we have an object that is called _Trainer_ that is responsible for training the models and logging and creating checkpoints during training.

### __Inference and Evaluation of a Trained Model__


To evaluate a trained model, one has to run `scripts/inference.py`.

```bash
python scripts/inference.py -c configs/inference/personality_test.yaml
```