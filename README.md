## Cluster setup
```bash
getnode -p gpu
module add soft/nvidia/cuda/12.0
/usr/bin/python3.9 -m venv .env
source .env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
cd lltm-extension
python setup.py install
python benchmark.py
```
