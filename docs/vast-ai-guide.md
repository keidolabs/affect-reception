# Vast.ai Cloud GPU Guide

Running v2 mechanistic experiments (8B/9B models) on Vast.ai from your Mac.

---

## One-Time Setup

### 1. Vast.ai account

1. Create account at [vast.ai](https://vast.ai) and add billing (credit card or crypto)
2. Add credit — $20 is a solid start for several sessions

### 2. SSH key

```bash
# Generate a dedicated key for Vast.ai (don't reuse your GitHub key)
ssh-keygen -t ed25519 -f ~/.ssh/vast_key -C "vast-mechlab"

# Copy the public key
cat ~/.ssh/vast_key.pub
```

Go to **Vast.ai → Account → SSH Keys → Add SSH Key** and paste the public key.

### 3. Install the Vast CLI (optional but useful)

```bash
pip install vastai
vastai set api-key YOUR_API_KEY   # from vast.ai account page
```

You can manage instances from the CLI instead of the browser:
```bash
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 disk_space>=150'
vastai create instance <offer_id> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 150
vastai show instances
vastai destroy instance <id>
```

---

## Daily Session Guide

### Step 1 — Commit and push any local changes

```bash
cd ~/dev/keidolabs/research/emotion-circuits
git add -p                          # review what's changed
git commit -m "ready for cloud run"
git push
```

> Never edit code on the cloud instance. All code lives in git on your Mac.

---

### Step 2 — Spin up a Vast.ai instance

**Via browser (easier):**
1. Go to [vast.ai/create](https://vast.ai/create)
2. Filter: **GPU: RTX 3090** (or A5000), **RAM: ≥48GB**, **Disk: ≥150GB**
3. Template: `pytorch/pytorch:2.x-cuda12.x-cudnn8-runtime` (preinstalled CUDA)
4. Disk: set to **150 GB**
5. Click **Rent** → note the **instance ID**, **IP**, and **port**

**Via CLI:**
```bash
# Find a cheap RTX 3090 with enough RAM and disk
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 cpu_ram>=48 disk_space>=150 rentable=true' --limit 5

# Rent it (replace OFFER_ID)
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --disk 150 \
  --ssh

# Get connection details
vastai show instances
```

---

### Step 3 — Connect and set up (first time only per instance)

```bash
# SSH in (get IP and port from Vast.ai console)
ssh -i ~/.ssh/vast_key root@<IP> -p <PORT>
```

Once inside the instance:
```bash
# Set your HF token (paste it here — it won't be saved to git)
export HF_TOKEN=hf_yourtoken

# Clone the repo
git clone https://github.com/<your-org>/emotion-circuits ~/emo-circuits
cd ~/emo-circuits

# Run the setup script (installs uv, tmux, syncs env, writes .env)
bash scripts/cloud_setup.sh
```

The setup script takes ~2–3 minutes. It will print "Setup complete" when done.

---

### Step 4 — Start a tmux session (always do this)

```bash
tmux new -s exp
```

This keeps your experiment running if your SSH connection drops. If you get
disconnected, reconnect with:
```bash
ssh -i ~/.ssh/vast_key root@<IP> -p <PORT>
tmux attach -t exp
```

---

### Step 5 — Pull latest code and run

```bash
cd ~/emo-circuits
git pull       # get any changes you pushed from your Mac

# Run the experiment (example: knockout, llama8b_inst)
uv run python experiments/16_v2_knockout/run.py llama8b_inst
```

Other common runs:
```bash
# Set A extraction — run one model at a time
uv run python experiments/11_v2_extract_seta/extract_llama8b_base.py
uv run python experiments/11_v2_extract_seta/extract_llama8b_inst.py
uv run python experiments/11_v2_extract_seta/extract_gemma9b_base.py
uv run python experiments/11_v2_extract_seta/extract_gemma9b_inst.py

# Set B extraction
uv run python experiments/12_v2_extract_setb/extract_llama8b_base.py
uv run python experiments/12_v2_extract_setb/extract_llama8b_inst.py

# Patching (needs exp 11 + 12 outputs to already exist)
uv run python experiments/14_v2_patching/run.py llama8b_base
uv run python experiments/14_v2_patching/run.py llama8b_inst

# Knockout
uv run python experiments/16_v2_knockout/run.py llama8b_base
uv run python experiments/16_v2_knockout/run.py llama8b_inst
```

---

### Step 6 — Monitor while it runs (open a second tmux pane)

```bash
# In tmux: Ctrl+b, then " (split horizontally) or % (split vertically)
# Then in the new pane:
watch -n 2 nvidia-smi
```

You should see VRAM usage climbing to ~20–22GB for 8B models. If it stays near 0,
something is wrong (model didn't load to GPU).

---

### Step 7 — Download results to external SSD

Once an experiment finishes (or while another is running in tmux), from your Mac:

```bash
# Download extraction outputs for a specific model (activations = bulk data)
./scripts/sync_down.sh <IP> <PORT> 11_v2_extract_seta llama8b_base
./scripts/sync_down.sh <IP> <PORT> 11_v2_extract_seta llama8b_inst

# Download full experiment output (CSV, plots, manifests — no .npz)
./scripts/sync_down.sh <IP> <PORT> 16_v2_knockout
./scripts/sync_down.sh <IP> <PORT> 14_v2_patching
```

Check what landed:
```bash
ls -lh /Volumes/mechlab/outputs/11_v2_extract_seta/activations/llama8b_base/
```

---

### Step 8 — Destroy the instance (do NOT forget)

```bash
# Via CLI
vastai destroy instance <id>

# Or via browser: Vast.ai console → Instances → Destroy
```

Verify in the console that no instances are running. An A100 at $1.20/hr left
running overnight = $10+ wasted. **Set a phone alarm** for expected runtime + 30 min.

---

## Experiment Sequencing

v2 phases have dependencies. Run in this order:

```
Exp 11 (extract Set A) ─┐
                         ├──▶ Exp 13 (probes — local, no model)
Exp 12 (extract Set B) ─┘         │
                                   ▼
                         Exp 14 (patching — needs 11 + 12)
                         Exp 16 (knockout — independent, just needs model)
                         Exp 15 (attention — local, no model)
                         Exp 17 (geometry — local, no model)
                         Exp 18 (summary — local, no model)
```

**Efficient cloud session:** run exp 11 → exp 12 → exp 14 → exp 16 sequentially
in one tmux session, download all at the end, destroy.

**Rough timings (RTX 3090, 8B models):**

| Experiment | Per model | Notes |
|---|---|---|
| 11 extract Set A | ~15–25 min | 80 stimuli × 32 layers |
| 12 extract Set B | ~30–45 min | 192 stimuli × 32 layers |
| 14 patching | ~10–15 min | Loads activations from 11/12 |
| 16 knockout | ~20–35 min | Live forward passes, no saved acts needed |

Run all 4 experiments for 2 models (base + inst) = ~3–4 hrs total. At $0.35/hr
that's ~$1–1.50.

---

## Troubleshooting

**Model download takes forever**

8B models are ~16GB. On Vast.ai datacenter links this takes 5–15 min. Normal.
Start the download, open a second tmux pane to monitor GPU while you wait.

**CUDA OOM mid-experiment**

The scripts save each stimulus incrementally, so partial outputs survive.
Check which `.npz` files landed, find the last `stimulus_id` in `manifest.csv`,
and restart the script — it will overwrite already-saved files (idempotent).

If you keep hitting OOM: filter to a GPU with more VRAM (A5000 24GB, or A100).

**`uv sync` fails with lock conflict**

```bash
uv sync --frozen   # use exact lock file, don't resolve
```

**SSH connection refused after spinning up**

Vast.ai instances take 1–3 min to boot. Wait, then retry. If still failing,
check the port — it's random and shown in the console.

**`attn_implementation="eager"` warning on CUDA**

Safe to ignore. It just means eager mode is explicitly enabled (required for
`output_attentions=True`). Without it, attention weights return `None`.

**Forgot to destroy — check now**

```bash
vastai show instances   # should return empty if nothing running
```

---

## Cost Cheat Sheet

| GPU | $/hr | 8B extraction (Set A + B) | Full exp 11–16 (4 models) |
|---|---|---|---|
| RTX 3090 24GB | ~$0.30–0.40 | ~1 hr → ~$0.35 | ~8–10 hrs → ~$3–4 |
| RTX A5000 24GB | ~$0.35–0.45 | ~1 hr → $0.40 | ~8–10 hrs → ~$4 |
| A100 80GB | ~$1.20–1.50 | ~30 min → $0.75 | (overkill for 8B) |

For 8B/9B models: RTX 3090 is the sweet spot. A100 is for 70B when we get there.
