## 🛠️ Installation

- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl python=3.9 -y
  conda activate internvl
  ```

- My experiments runs on cuda-12.1 and torch:2.5.1+cu121

  ```
  conda install -c nvidia cudatoolkit=12.1
  export CUDA_PATH=/usr/local/cuda-12.1 #replace it with your path where cuda-12.1 is installed
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  ```

- Ensure Correct CUDA & PyTorch Versions: it should be 12.1
  ```
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
  ```


- Install dependencies using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

  By default, our `requirements.txt` file includes the following dependencies:

  - `-r requirements/internvl_chat.txt`
  - `-r requirements/streamlit_demo.txt`
  - `-r requirements/classification.txt`
  - `-r requirements/segmentation.txt`

  The `clip_benchmark.txt` is **not** included in the default installation. If you require the `clip_benchmark` functionality, please install it manually by running the following command:

  ```bash
  pip install -r requirements/clip_benchmark.txt
  ```

### Additional Instructions

- Install `flash-attn==2.7.4.post1`:

  ```bash
  pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
  ```

- Install `mmcv-full==1.6.2` (optional, for `segmentation`):

  ```bash
  pip install -U openmim
  mim install mmcv-full==1.6.2
  ```

- Install `apex` (optional, for `segmentation`):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

  If you encounter `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, it is because apex's CUDA extensions are not being installed successfully. You can try uninstalling apex and the code will default to the PyTorch version of RMSNorm. Alternatively, if you prefer using apex, try adding a few lines to `setup.py` and then recompiling.

  <img src=https://github.com/OpenGVLab/InternVL/assets/23737120/c04a989c-8024-49fa-b62c-2da623e63729 width=50%>


- Solution to some potential errors:

  if you encounter `ImportError: cannot import name 'log' from 'torch.distributed.elastic.agent.server.api'`, it is because DeepSpeed is trying to import log from torch.distributed.elastic.agent.server.api, but it's either been removed or moved in the latest PyTorch versions.

  or 

  The error `"AssertionError: It is illegal to call Engine.step() inside no_sync context manager"` occurs when DeepSpeed tries to execute Engine.step() inside a no_sync context, which prevents gradient synchronization across processes.

  ```
  pip install deepspeed==0.15.4 accelerate==0.34.2
  ```

  if you encounter error: `packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: -9) local_rank: 0 (pid: 382608) of binary: `, it could be because of small CPU memory, you should put 200 GB memory
  add `--mem=200GB` to the slurm cmd

