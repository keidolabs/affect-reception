# Makefile — emotion-circuits cloud GPU orchestration
# See: lab-journal/thinking-about-cloud-gpu.md for full workflow docs
#
# Quick start:
#   1. cp .env.session.sample .env.session  # fill in INSTANCE and PORT
#   2. make session  EXP=16_v2_knockout MODEL=llama8b_inst
#   3. make status
#   4. make grab     EXP=16_v2_knockout MODEL=llama8b_inst
#   5. make shutdown

-include .env.session

INSTANCE ?= $(error Set INSTANCE= (ip from Vast.ai console) or add to .env.session)
PORT     ?= $(error Set PORT= (port from Vast.ai console) or add to .env.session)

# EXP and MODEL can be set on command line or in .env.session
# SCRIPT defaults to "run.py $(MODEL)" for experiments that take a model arg.
# Override for extraction experiments: make run SCRIPT=extract_llama8b_inst.py
EXP    ?= $(error Set EXP= (e.g. 16_v2_knockout))
MODEL  ?= $(error Set MODEL= (e.g. llama8b_inst))
SCRIPT ?= run.py $(MODEL)

SSH = ssh -i ~/.ssh/vast_key -p $(PORT) root@$(INSTANCE)

.DEFAULT_GOAL := help

.PHONY: cloud-setup sync-up push-acts run session status logs gpu grab shutdown help

# ── First-time setup ──────────────────────────────────────────────────────────

cloud-setup:
	@echo "Setting up new instance $(INSTANCE):$(PORT)..."
	$(SSH) 'bash -s' < scripts/cloud_setup.sh
	@echo "$$(date +%Y-%m-%dT%H:%M) | cloud-setup | $(INSTANCE):$(PORT)" >> runs.log

# ── Code sync ─────────────────────────────────────────────────────────────────

sync-up:
	./scripts/sync_up.sh $(INSTANCE) $(PORT)
	@echo "$$(date +%Y-%m-%dT%H:%M) | sync-up | $(INSTANCE):$(PORT)" >> runs.log

# ── Push saved activations to cloud ──────────────────────────────────────────
#
# Use when .npz activation files already exist locally and you need them on
# the cloud instance for experiments that read saved activations (Exp 14 patching).
# Faster than re-running extraction on cloud (~seconds vs ~hours).
#
# Usage (push one model's activations from both sets):
#   make push-acts MODEL=llama8b_inst
#
# This pushes:
#   experiments/11_v2_extract_seta/outputs/activations/<MODEL>/
#   experiments/12_v2_extract_setb/outputs/activations/<MODEL>/

push-acts:
	@echo "Pushing $(MODEL) activations to $(INSTANCE):$(PORT)..."
	rsync -avz --progress \
	  -e "ssh -i ~/.ssh/vast_key -p $(PORT)" \
	  experiments/11_v2_extract_seta/outputs/activations/$(MODEL)/ \
	  root@$(INSTANCE):~/emo-circuits/experiments/11_v2_extract_seta/outputs/activations/$(MODEL)/
	rsync -avz --progress \
	  -e "ssh -i ~/.ssh/vast_key -p $(PORT)" \
	  experiments/12_v2_extract_setb/outputs/activations/$(MODEL)/ \
	  root@$(INSTANCE):~/emo-circuits/experiments/12_v2_extract_setb/outputs/activations/$(MODEL)/
	@echo "$$(date +%Y-%m-%dT%H:%M) | push-acts | $(MODEL)" >> runs.log

# ── Run experiments ───────────────────────────────────────────────────────────
#
# Launches in a tmux session named 'exp'. Tees stdout to outputs/run.log so
# you can tail it remotely with: make logs EXP=<exp>
#
# Single run.py experiments (pass model as arg):
#   make run EXP=16_v2_knockout MODEL=llama8b_inst
#
# Extraction experiments (one script per model):
#   make run EXP=11_v2_extract_seta SCRIPT=extract_llama8b_inst.py

run:
	$(SSH) 'cd ~/emo-circuits \
	  && mkdir -p experiments/$(EXP)/outputs \
	  && tmux new-session -d -s exp \
	     "uv run python experiments/$(EXP)/$(SCRIPT) \
	      2>&1 | tee experiments/$(EXP)/outputs/run.log"'
	@echo "$$(date +%Y-%m-%dT%H:%M) | run | $(EXP) | $(MODEL) | $(SCRIPT)" >> runs.log
	@echo "Experiment launched. Monitor with: make status"

# Sync code then immediately launch — the common case
session: sync-up run

# ── Monitoring ────────────────────────────────────────────────────────────────

status:
	@$(SSH) 'tmux capture-pane -t exp -p 2>/dev/null | tail -30 \
	  || echo "(no tmux session named exp)"'

logs:
	$(SSH) 'tail -50 ~/emo-circuits/experiments/$(EXP)/outputs/run.log 2>/dev/null \
	  || echo "(no run.log yet — experiment may still be starting)"'

gpu:
	$(SSH) 'nvidia-smi'

# ── Download results ──────────────────────────────────────────────────────────
#
# Downloads experiment outputs to /Volumes/mechlab/outputs/<exp>/<model>/
# Requires the Samsung T7 to be mounted at /Volumes/mechlab/

grab:
	./scripts/sync_down.sh $(INSTANCE) $(PORT) $(EXP) $(MODEL)
	@echo "$$(date +%Y-%m-%dT%H:%M) | grab | $(EXP) | $(MODEL)" >> runs.log

# ── Shutdown ──────────────────────────────────────────────────────────────────

shutdown:
	@echo ""
	@echo "  MANUAL STEP: destroy the Vast.ai instance to stop billing."
	@echo ""
	@echo "  vast destroy instance <id>"
	@echo "  Or visit: https://cloud.vast.ai/instances/"
	@echo ""
	@echo "  Verify no running instances before closing your terminal."
	@echo ""
	@echo "$$(date +%Y-%m-%dT%H:%M) | shutdown-reminder | $(INSTANCE):$(PORT)" >> runs.log

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "emotion-circuits cloud GPU orchestration"
	@echo ""
	@echo "Typical session:"
	@echo "  make cloud-setup  INSTANCE=1.2.3.4 PORT=22222          (first time only)"
	@echo "  make session      EXP=16_v2_knockout  MODEL=llama8b_inst"
	@echo "  make status"
	@echo "  make logs         EXP=16_v2_knockout"
	@echo "  make gpu"
	@echo "  make grab         EXP=16_v2_knockout  MODEL=llama8b_inst"
	@echo "  make shutdown"
	@echo ""
	@echo "Extraction experiments (separate script per model):"
	@echo "  make run  EXP=11_v2_extract_seta  SCRIPT=extract_llama8b_inst.py  MODEL=llama8b_inst"
	@echo ""
	@echo "Tip: put INSTANCE, PORT, EXP, MODEL in .env.session to skip repeated args."
	@echo "     cp .env.session.sample .env.session"
	@echo ""
	@echo "Pushing saved activations (when .npz already exist locally):"
	@echo "  make push-acts MODEL=llama8b_inst   # pushes Exp11+12 .npz for that model"
	@echo ""
