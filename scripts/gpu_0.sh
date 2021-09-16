# delete_incomplete

python -m domainbed.scripts.sweep launch \
       --data_dir=./domainbed/data \
       --output_dir=./train_output \
       --algorithms ERM \
       --datasets ColoredMNIST \
       --n_trials 10000 \
       --n_hparams 5 \
       --n_trials 1 \
       --wandb \
       --command_launcher multi_gpu \
       --sweep_name "erm_cmnist_sweep" 

python -m domainbed.scripts.sweep launch \
       --data_dir=./domainbed/data \
       --output_dir=./train_output/irm \
       --algorithms IRM \
       --datasets ColoredMNIST \
       --n_trials 10000 \
       --n_hparams 5 \
       --n_trials 1 \
       --wandb \
       --command_launcher multi_gpu \
       --sweep_name "irm_cmnist_sweep" 