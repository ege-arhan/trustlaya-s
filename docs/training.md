# Training

The source dataset contains 10,000 deterministic synthetic examples across eleven categories and Turkish, English and mixed-language templates. Template family, not row, determines the split (6,972 train, 1,053 validation, 1,975 test). Values such as addresses, emails and secrets are synthetic. No public real-world safety corpus was added.

The student backbone is `ytu-ce-cosmos/turkish-medium-bert-uncased` (MIT). Embeddings and first two encoder layers are frozen. Remaining encoder layers and all heads train with supervised binary cross-entropy and categorical cross-entropy. The optional Laya multilingual teacher (Apache-2.0) provides nine typed `noul` probabilities on 128 source rows. Only teacher targets whose binary side agrees with the synthetic label enter the weak distillation term (weight 0.05). Teacher output is never a ground-truth label. Teacher probabilities lack task-specific calibration here.

Run commands in README. The training log and exact counts are in `logs/training.log`, `logs/distillation.log` and `reports/training.json`. The final model is written in safetensors format; tokenizer files are in the same directory.
