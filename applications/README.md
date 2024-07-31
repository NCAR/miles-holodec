# Using the Lightning Trainers
The pytorch lightning trainers are designed to be hardware agnostic and easily scalable. The HolodecCLI is intended to make
configuration options very easy and keep the training and inference very modular.

### Configuration
A default config can be obtained by executing `python main.py fit --print_config` replacing `main` with your lightning application.
This will show all the available configuration options and their respective default values, and can be saved with `python main.py fit --print_config > default.yml`
This file can then be altered to your needs. For config options that accept multiple classes, you can specify specific parameters
in combination to recieve the default config when using that class, i.e. 
`python main.py fit --print_config --lr_scheduler.class_path ReduceLROnPlateau`

### Training
Training can be performed by executing the `fit` command and providing a configuration file.
`python main.py fit --config my_config.yml`. You can also provide multiple configuration files, or append config changes at command line, such as
`python main.py fit --config my_config.yml --trainer.fast_dev_run 1` The fast_dev_run is a good way to quickly test if things are configured
as expected. The trainer will automatically save checkpoints, hyperparameters, and the configuration file used during training, but can be
further customized with the trainer.logger config options.

### Inference
Inference can be performed by executing the `predict` command and configuration settings. An example would be:
`python main.py predict --config my_config.yml --ckpt_path results/version_6/checkpoints/epoch\=9-step\=5000.ckpt` Note you can include the checkpoint
in the config file or provide it at command line. To save predictions, use the config settings found under
`trainer.callbacks.class_path HolodecWriter`. Post-processing is currently available in the notebooks folder, with `lightning_figures.ipynb` loading
inference results, performing 3D clustering, and generating many figures on the results.

### Scalability
Altering the trainer configuration settings can enable multi-GPU training, and using the DDP strategy will work out of the box 
with training and prediction. More complex scaling, such as FSDP, would require additional setup and configuration with environment variables.
