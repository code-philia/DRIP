
# Start a new conda environment

```commandline
  conda create -n advprompter python=3.11.4
  conda activate advprompter
  pip install -r requirements.txt
```

Check whether the torchrl has been successfully installed
```commandline
python -c "import torchrl; from torchrl.data import ReplayBuffer; print('TorchRL OK')"
```
