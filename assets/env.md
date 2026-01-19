Below is the instruction to **setup** and **test** the environment for 🌊LaMer:

### Install veRL
The first step is to install veRL:
```
conda create -n lamer python==3.12 -y
conda activate lamer

pip3 install vllm==0.8.5

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```

### Environment
For Sokoban and MineSweeper, no extra packages are needed. For Webshop and ALFWorld, please follow the instructions from [verl-agent](https://github.com/langfengQ/verl-agent/tree/master?tab=readme-ov-file#install-supported-environments) which are copied below:

#### ALFWorld
Install with pip:
```
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```
Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in ~/.cache/alfworld/):
```
alfworld-download -f
Use --extra to download pre-trained checkpoints and seq2seq data.
```
Play a Textworld game:
```
alfworld-play-tw
```

#### Webshop
Webshop requires Python <=3.10, so begin by creating a new verl-agent-webshop environment
```
conda create -n webshop python==3.10 -y
conda activate webshop
```
Install Webshop
```
cd ./agent_system/environments/webshop/webshop
./setup.sh -d all
```
Note: If you encounter issues with gdown, you may need to visit https://drive.google.com/, get your Google Drive cookie, and paste it into .cache/gdown/cookies.txt. Or you may need to manually download the files.

After Webshop is installed, return to the root directory of the repository and install the verl package in verl-agent:
```
cd repo_root/
pip3 install vllm==0.8.5

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```
Please also refer to the original [Webshop](https://github.com/princeton-nlp/WebShop?tab=readme-ov-file#-setup) repository for more detailed information.

### Test Environment
We provide the script `examples/test_env.py` for testing environments and inspecting game states, prompts, and reward signals.
```bash
python -m examples.test_env
```
You can specify the environment by modifying the `env_name`:
```python
env_name = 'sokoban'  # 'minesweeper' or 'sokoban' or 'webshop' or 'alfworld'
```
This script simulates random policy on the environment using only CPU and does not require any GPU resources.