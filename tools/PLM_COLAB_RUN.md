# Colab PLM run

This is GPU-only model training. The Mac preflight validates rows and manifests,
but it does not produce a neural-network AUC.

1. In Colab, choose **Runtime → Change runtime type → T4 GPU**.
2. Upload `tools/finetune_plm_colab.py` and
   `polyreactivity_dataset.csv` from `~/Downloads/antibody_sources/`.
3. Install the runtime packages:

   ```bash
   !pip install -q transformers accelerate scikit-learn peft
   ```

4. Run the frozen-encoder rung first:

   ```bash
   !python finetune_plm_colab.py \
     --csv polyreactivity_dataset.csv \
     --arm library_only \
     --mode frozen \
     --seeds 42,43,44 \
     --work-dir /content/drive/MyDrive/biostuff_plm \
     --manifest-out /content/drive/MyDrive/biostuff_plm/frozen_manifest.json \
     --out /content/drive/MyDrive/biostuff_plm/frozen_report.json
   ```

5. Only if frozen improves the fixed AIMS/Shehata macro AUC, repeat with
   `--mode adapter`. Only if the adapter improves again should `--mode full` be
   attempted. Use `--resume` after an interrupted run.

Repeat the comparison with `--arm diverse_only` and `--arm mixed`. Do not tune
hyperparameters against the holdout scores, and do not add GDPa3. Each run saves
checkpoints, a model, per-row held-out predictions, a resolved manifest, hashes,
and per-source ROC/PR AUCs. AIMS flu remains a tiny 97-row source and must be
reported separately.
