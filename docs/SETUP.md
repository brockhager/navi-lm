# Setup checklist: WSL2 + PyTorch (Blackwell)

## 1. WSL2 (PowerShell, as admin)

```powershell
wsl --install -d Ubuntu
wsl --update
```

Reboot if asked. Keep the Windows NVIDIA driver (610.88). Do not install a Linux GPU driver inside WSL.

## 2. Cap WSL memory

Create `C:\Users\brock\.wslconfig`:

```ini
[wsl2]
memory=11GB
processors=8
swap=8GB
```

Then `wsl --shutdown` and reopen Ubuntu.

## 3. Confirm the GPU is visible

```bash
nvidia-smi
```

It should list the RTX 5060.

## 4. Python environment

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
python3 -m venv ~/venvs/navilm
source ~/venvs/navilm/bin/activate
```

## 5. PyTorch with CUDA 12.8+

Blackwell needs a CUDA 12.8 or newer build. Confirm the current command on pytorch.org, typically:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expect `True`, the RTX 5060, and capability `(12, 0)`.

## 6. NCM

```bash
git clone https://github.com/HazyResearch/ncm.git
cd ncm
pip install -r requirements.txt
```

Check that `requirements.txt` did not replace the CUDA 12.8 torch build. Re-run the check in step 5 afterwards.

## 7. Evaluate a released checkpoint

Commands from the repo README:

```bash
hf download hazyresearch/ncm-checkpoints --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --local-dir data
python evaluate.py --config config/eval/lm1b-small.yaml --checkpoint-dir artifacts/lm1b/small
```

Verify the exact config and checkpoint paths against the repo after download. The README uses `config/eval/<dataset>.yaml` and `artifacts/<dataset>`.

Keep the working copy inside the WSL filesystem (`~/`), not under `/mnt/c`, for speed.
