import subprocess
import time

# List of commands to run
commands = [
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=2_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=3_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=4_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=5_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=1_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=2_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=3_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=4_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=5_bs=64_lr=0.0001.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=1_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=2_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=3_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=4_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/birnn/BiDeepRNN_layers=5_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=1_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=2_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=3_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=4_bs=64_lr=1e-05.json",
    "python train.py --config ./configs/deeprnn/DeepRNN_layers=5_bs=64_lr=1e-05.json"
]

# Parameters for batch processing
max_concurrent = 3  # Number of processes to run at the same time

def run_in_batches(commands, max_concurrent):
    processes = []
    for i, cmd in enumerate(commands):
        # Start the command
        print(f"Starting: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

        # Check if we need to wait for batch to complete
        if (i + 1) % max_concurrent == 0 or i == len(commands) - 1:
            for p in processes:
                p.wait()  # Wait for all processes in the batch to complete
            processes = []  # Clear the list for the next batch
            time.sleep(2)  # Optional: add a delay between batches

    print("All commands have been executed.")

# Run the function
run_in_batches(commands, max_concurrent)
