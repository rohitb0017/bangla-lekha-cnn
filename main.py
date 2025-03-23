import os, sys, mlconfig, shutil
from pathlib import Path
from train_model import train_model
from inference import run_inference

parent = '/content/banglalekha_dataset/BanglaLekha_Dataset/'

# Sample argument : python main.py train
num_args = len(sys.argv)
if num_args > 2:
    sys.exit(f"Too many arguments : Expected atmost 1, got {num_args-1}")
if num_args < 2 :
    sys.exit(f"Expected atleast 1 argument\nUsage :\npython main.py <mode : train | inference>")
if num_args == 2:
    if sys.argv[1] not in ['train', 'inference']:
        sys.exit(f"Mode '{sys.argv[1]}' is invalid!")

mode = sys.argv[1]

if mode == 'train' :
    config = 'train_config.yaml'


    fn = os.path.join(parent, config)
    config = mlconfig.load(fn)
    
    train_model(config = config)

if mode == 'inference' :
    config = 'inference_config.yaml'

    
    fn = os.path.join(parent, config)
    config = mlconfig.load(fn)
    
    checkpoint_path = os.path.join('/content/banglalekha_dataset/BanglaLekha_Dataset/', 'Checkpoints')
    models = [i for i in os.listdir(checkpoint_path) if i.endswith('.pth')]
    if len(models) < 1:
        sys.exit(f"No pre-trained models available. Train a model first.")
    if not os.path.exists('/content/banglalekha_dataset/BanglaLekha_Dataset/Checkpoints'):
        sys.exit(f"Data directory not available!")
    model_name = f"{config.model_name}.pth"
    if model_name not in models:
        sys.exit(f"No model available with name : {model_name}")
    model_name = model_name.rsplit('.', maxsplit=1)[0]

    logs_dir = os.path.join('/content/banglalekha_dataset/BanglaLekha_Dataset/', 'Plots & Outputs')
    os.makedirs(logs_dir, exist_ok=True)  #  Create the directory if it doesn’t exist

    logs = os.path.join(logs_dir, f'inference_logs_{model_name}.txt')
    output = open(logs, 'w')  # Now it won't fail

    run_inference(config, output)
    output.close()