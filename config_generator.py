import os 
import json 
from itertools import product 
 
base_config = { 
    "model_config": {
        "model_type": "BiDeepRNN",
        "args" : {
            "dim_input": 200,
            "dim_hidden": 128,
            "num_layers": 1,
            "dim_output": 2,
            "embedding_strategy": "word2vec",
            "pretrained_path": "glove-twitter-200",
            "embedding_frozen": True,
            "context_window": 5,
            "oov_handing": "using_unk"
        }
    }, 
    "tokenizer_config": { 
        "tokenizer_type": "nltk", 
        "args": { 
            "dataset": "rotten_tomatoes" 
        } 
    }, 
    "trainer_args": { 
        "task": "classification", 
        "training_batch_size": 32, 
        "validation_batch_size": 32, 
        "learning_rate": 0.0001, 
        "epoch": 30 
    }, 
    "metric_config": { 
        "metrics": [ 
            {"name": "accuracy", "args": {}}, 
            {"name": "f1", "args": {}}, 
            {"name": "precision", "args": {}}, 
            {"name": "recall", "args": {}} 
        ] 
    }, 
    "data_config": { 
        "name": "rotten_tomatoes", 
        "is_huggingface": True, 
        "type": "classification" 
    }, 
    "analysis_config": {
        "output_dir": "experiments/n_layers/bideeprnn_layer=1",
        "record_metrics": True,
        "record_gradients": True,
        "save_interval": 1000
    }
} 
 
# Parameters to vary 
batch_sizes = [16, 32, 64] 
learning_rates = [0.001, 0.0001, 0.00001] 
architectures = ['BiDeepRNN', 'DeepRNN'] 
num_layers_list = [1, 2, 3, 4, 5] 
 
# config_dir = './configs/BiDeepRNNandDeepRNN/' 
 
# os.makedirs(config_dir, exist_ok=True) 
 
shell_script_lines = [] 
 
for batch_size, learning_rate, architecture, num_layers in product(batch_sizes, learning_rates, architectures, num_layers_list): 
    if architecture == 'BiDeepRNN':
        config_dir = './configs/birnn'
    elif architecture == 'DeepRNN':
        config_dir = './configs/deeprnn'

    os.makedirs(config_dir, exist_ok=True) 

    config = json.loads(json.dumps(base_config))
     
    config['model_config']['model_type'] = architecture 
    config['model_config']['args']['num_layers'] = num_layers 
    config['trainer_args']['training_batch_size'] = batch_size 
    config['trainer_args']['validation_batch_size'] = batch_size 
    config['trainer_args']['learning_rate'] = learning_rate 
     
    output_dir = f"experiments/{architecture}_layers={num_layers}_bs={batch_size}_lr={learning_rate}" 
    config['analysis_config']['output_dir'] = output_dir 
     
    config_filename = f"{architecture}_layers={num_layers}_bs={batch_size}_lr={learning_rate}.json" 
    config_filepath = os.path.join(config_dir, config_filename) 
     
    with open(config_filepath, 'w') as f: 
        json.dump(config, f, indent=4) 
     
    shell_script_lines.append(f"python train.py --config {config_filepath}") 
     
shell_script_content = '\n'.join(shell_script_lines) 
shell_script_path = 'run_all_configs.sh' 
 
with open(shell_script_path, 'w') as f: 
    f.write(shell_script_content) 
 
os.chmod(shell_script_path, 0o755) 
 
print(f"Generated {len(shell_script_lines)} configurations.") 
print(f"Configurations are saved in {config_dir}") 
print(f"Shell script to run all configurations is saved as {shell_script_path}")