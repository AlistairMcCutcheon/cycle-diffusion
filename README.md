Code originally from https://github.com/ChenWu98/cycle-diffusion

```
pip install -r requirements.txt
git clone git@github.com:CompVis/taming-transformers.git
pip install -e taming-transformers/
```
Download the two DDPMs I trained, one for smoke, one for images without smoke. Trained using https://github.com/openai/improved-diffusion
```
gdown https://drive.google.com/uc?id=1ZLlsgTerOxH4xliKljB0n_3vDkX2-wX6
gdown https://drive.google.com/uc?id=18P2-l3NE8ybPXHwqGliMPdmHLHPHZaC-
mv smoke_ema_0.9999_380000.pt cycle-diffusion/ckpts/ddpm/
mv empty_ema_0.9999_380000.pt cycle-diffusion/ckpts/ddpm/
```

Set up wandb for logging (registration is required). You should modify the setup_wandb function in cycle-diffusion/main.py to accomodate your wandb credentials. You may want to run something like:

```
wandb login
```
Prepare the dataset
```
gdown https://drive.google.com/uc?id=19LSrZHYQqJSdKgH8Mtlgg7-i-L3eRhbh
unzip ./D-Fire.zip
mkdir data/ data/DFire/ data/DFire/raw data/DFire/clean data/DFire/clean/train data/DFire/clean/train/empty data/DFire/clean/train/smoke data/DFire/clean/test data/DFire/clean/test/empty data/DFire/clean/test/smoke 
mv train data/DFire/raw/
mv test data/DFire/raw/
python prepare_dfire.py
```
Generate outputs using the two already trained ddpms:
```
cd cycle-diffusion
export RUN_NAME=translate_empty64_to_smoke64_ddim_eta01
export SEED=42
torchrun --nproc_per_node 1 --master_port 1446 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_predict --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```

See image outputs here: cycle-diffusion/output/translate_empty64_to_smoke64_ddim_eta0142/temp_gen

