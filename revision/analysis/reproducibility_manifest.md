# Reproducibility manifest

Entries in this file are append-only. Paths are workspace-relative and no
credential material is recorded.

## 2026-07-18 — canonical original entity-split metrics (no-training reevaluation)

- Commit SHA before artifact generation: `2cc1566e8f9cb04f3d82d16457225a701905e904`
- Commit SHA after artifact generation: `89359751a8b942152d8db5d5a5782f07f87e44f0`
- Script SHA-256: `7ea342405f64153a93c60035e832bfe799e8b21de70e4eadddd621c3ef16bc7a`
- Unit-test SHA-256: `b3d56fd70afa939d8083fe3f2c3a724dd6ef298fbf9797b7378b5edf8b4c67f3`
- Command: `python revision/recompute_entity_split_metrics.py --output-dir revision/analysis --device cuda`
- Seed: `42`; requested/resolved device: `cuda` / `cuda` (NVIDIA GeForce RTX 4080); Python `3.10.14`; Torch `2.4.0+cu121`.
- Configuration: `BioInteract/configs/default.yaml`, SHA-256 `ba31fa23f1a01b5547283f1fdb06c8a84d909140e4c3d500d8f4e08aed82beba`.
- Archived checkpoints: `BioInteract/checkpoints/best_random.pt` `a91f5e75e123af4c0a66ff9302842b0bcacdce2780abb9f7f5f4170b69f3e347`; `BioInteract/checkpoints/best_cold_target.pt` `3b5051a8b69ea5208dceb41d9db8766c8dae87dcee2e33c0116c20c1ad003e4b`; `BioInteract/checkpoints/best_cold_drug.pt` `a813f4afd9dea005e8c49db46ce5bfd088a56d1644c87988ec293e0378805355`.
- Davis inputs: `interactions.csv` `996c93442822f138fabbb6553570fbddf8f4bdd863c52351776b11c3dd4be73f`; `drug_smiles.csv` `2d0e76a53ba39fcbd8d90136cacbdf8681c437fc486f0057e13e8401b25c6b9d`; `target_sequences.csv` `34e3a6d2deec0e7d17f57b9a53ad16999231a36496cee5dd3c732cc244e3b3a5`; `BioInteract/data/esm2_embeddings` (442 files; 852510612 bytes) `22588ffe20d2ba5a81a259898be570a661940df5a396f0f1069767aeb02503d2`.
- Generated artifacts: `original_entity_split_metrics.json` `b53b918a6729c6c2f2c97d7a94260bb63a2f60bc2ec1637e05531d6dbbcacdc7`; `random_validation_predictions.csv` `7a9959825af8c5790d947f320a5af9e7977275e27fb0e0b5c7ff7b59bd96fd79`; `random_test_predictions.csv` `860986156ca0e4c4360382262147afc00a77fdca6ea750d282bdfe82030f9fa4`; `target_id_held_out_validation_predictions.csv` `ec701bc6b890cd257010098055d96c39a6b57b230c9fc914391b6827d44c04b1`; `target_id_held_out_test_predictions.csv` `85de2bfa39bdca075080961f442f71fea2c036885aa8183298283ced3782b083`; `drug_id_held_out_validation_predictions.csv` `a944fb9975728ec3ae00573fb932a67c4799e95618c3c9644bf311e9b9ecab14`; `drug_id_held_out_test_predictions.csv` `9b457987694329d0067789c406b88f13cd46e043c58152c2e78b76963e5280ac`.
- Protocol: the F1 threshold was selected once from validation predictions for each archived seed-42 split and frozen for test evaluation. No training, checkpoint writes, data/split changes, or historical-result overwrites occurred.

## 2026-07-18 — AMP precision-status correction

- Commit SHA before status correction: `4f23768039c5032352628000a3d183d2611bcb2a`
- Commit SHA after status correction: `f929e8116283c12720523f9c11f9995d89c150ad`
- The recorded command used the fixed evaluator's `torch.cuda.amp.autocast(enabled=config['training'].get('amp', True))` path. Because `BioInteract/configs/default.yaml` sets `training.amp: true`, model forward passes used CUDA AMP autocast (float16), with archived FP32 checkpoint weights loaded by `torch.load(..., map_location='cuda', weights_only=False)`.
- Existing full-precision checkpoint prediction CSVs differ materially from these AMP-derived predictions. The retained JSON is therefore explicitly `canonical: false`; it is audit evidence only and does not supersede historical results or support manuscript/result edits until a precision-matched audit is complete.
- Corrected script SHA-256: `2afb91e0e1bfc043e31abb974dc6e04a8a2b28f198d2d05f173466bf7dd3627b`; corrected unit-test SHA-256: `94a035feb7d161a1dbe7fa2eb9f1aa05c3fb9a96d752aa542b68e88620fc455b`; status-corrected JSON SHA-256: `32a48009fea4bd72a1916ca5421a059e9f2a341dcf6aceda73bcf4ec7dc782f9`.

## 2026-07-18 — current-release full-precision canonical correction

- Commit SHA before current-release regeneration: `c5a0ec21290d2eeaa0f4cc70ee8a20c3f517dbb8`
- Commit SHA after current-release regeneration: `61218492ca0bd6075dc4a9cfe13ec68f30dfccff`
- Command: `python revision/recompute_entity_split_metrics.py --output-dir revision/analysis --device cuda`
- Data/split protocol retained from `BioInteract/configs/default.yaml`, SHA-256 `ba31fa23f1a01b5547283f1fdb06c8a84d909140e4c3d500d8f4e08aed82beba`: Davis data, seed `42`, `70/10/20` split proportions, `batch_size: 64`, `shuffle: false`, and `num_workers: 0`.
- Each model was instantiated from its own archived `checkpoint['config']`, not the default model section. Exact canonical-JSON architecture hashes: `best_random.pt` `16d45a2a726556a53fa7ee051296b8c7a7fe61524695d51c21247a72b8a3c581`; `best_cold_target.pt` `c51f2188e819006320f49d50aafb70440c9a1abc432dbda14d88411f34edab2c`; `best_cold_drug.pt` `16d45a2a726556a53fa7ee051296b8c7a7fe61524695d51c21247a72b8a3c581`. The full architecture objects are embedded per split in `original_entity_split_metrics.json`.
- Inference used direct `torch.inference_mode()` full precision (`amp_enabled: false`) on CUDA / NVIDIA GeForce RTX 4080, Python `3.10.14`, Torch `2.4.0+cu121`. It does not import or call the CLI evaluator collector.
- Validation-selected/frozen-threshold results: Random threshold `0.5959881544113159`, AUROC `0.903904640966352`, AUPRC `0.5601254874754175`, F1 `0.5653846153846154`; Target-ID-held-out threshold `0.6022375822067261`, AUROC `0.9299874433205442`, AUPRC `0.5244660498174324`, F1 `0.5338491295938105`; Drug-ID-held-out threshold `0.8456151485443115`, AUROC `0.7334469799554069`, AUPRC `0.16628563292349766`, F1 `0.1`.
- This is canonical for the current-release checkpoint-plus-input protocol. It supersedes legacy metric JSON and figure prediction CSVs because those CSV scores cannot be recreated from the archived current checkpoints and released current inputs (random-split Pearson `0.881`, Spearman `0.873`, maximum absolute score difference `0.620`). Legacy results were neither edited nor overwritten.
- Corrected script SHA-256: `c874a5125575dae82a37273beef49feb40c335eb5a8034a4e1645f1f8cc11114`; test SHA-256: `89872bd76622f8d9334d129191d5cf7f3b8cf9427f129a62d7b6e5fa236d9221`; canonical JSON SHA-256: `d07b84be4ca57bcddefd9de5fe96b88406557152df7bb7ea8e2286fc5e86409b`.
- Corrected generated CSV SHA-256: `random_validation_predictions.csv` `1e9cebb8cccee283594d11ea562d58f684c67adb160434a987126f9ee6c4c593`; `random_test_predictions.csv` `be54ae37ad407b610aacc4193c55244156271968a068a2517496964fad6f88ac`; `target_id_held_out_validation_predictions.csv` `c18d8e820b45b2db9eabf8ebb529ddc071ef85c3358c551ccc058d71eb37d63e`; `target_id_held_out_test_predictions.csv` `bc6e5cee61a9d76d398aac8a2d1a6c985b74f17197f12739b797c11c2acebc06`; `drug_id_held_out_validation_predictions.csv` `75bb4a046f21a6c58a38783d43eb97413671668b7433a6ead5ac46008e3467f4`; `drug_id_held_out_test_predictions.csv` `65be4bddab35fe011acb47a6644423dfe796d4c9361351974d356a50f6be9e68`.
- No training, checkpoint writes, input/split changes, manuscript edits, or legacy result-file writes occurred.

## 2026-07-18 — deterministic canonical release verification

- Command: `python revision/recompute_entity_split_metrics.py --output-dir revision/analysis --device cuda`.
- Determinism controls were enabled before CUDA initialisation: `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, disabled cuDNN benchmarking/TF32, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The release run used direct full-precision `torch.inference_mode()` only; no model training, checkpoint write, data change, or split change occurred.
- Two independent strict CUDA runs produced identical SHA-256 values for all six prediction CSV files. The JSON differs only in its self-recorded output-directory command string. The promoted `revision/analysis` CSV hashes are: Random validation `697957340dda4d3bff0ee38122c397f2e3b251f7930dff0920c62f0f083b0a65`, Random test `330d1cd929de02c658df43b74977b9623f1499165cf41a032eb81e7ccb42bb73`; Target-ID-held-out validation `f217930a811e99b21b45f667d63afe18000d1e53893d551e45b1f5cd6b944426`, Target-ID-held-out test `9cbc3992dc5c80d682b2a2bc44268a91c1e51d2ba3a3521e791fe9a732f54765`; Drug-ID-held-out validation `0a89f2c333e7cb2850fe8f0a3735cbd463d27efc70922748ed55337a1f4ddbc0`, Drug-ID-held-out test `57d77fd8145f6d1733a26bf9f58e2a72cc89b1ec15cfee483231d92eb6c92214`.
- Canonical JSON SHA-256: `6d32bbcc39be4c6d9f0199db6406efc24d1a1654a08dc7dc090b4c5594749474`; reconstruction script SHA-256: `6fdacbef633e0b0055c8fceabd56062d49e2340a776d7246f9352707dfc475d0`; unit-test SHA-256: `888b06807e97b4cdcff3db5a35eaa209e79860ee1eabf6763ea483d4e4e5ec06`.
- Validation-selected/frozen-threshold results: Random threshold `0.5959881544113159`, AUROC `0.9039055886180711`, AUPRC `0.5601694007558918`, F1 `0.5653846153846154`; Target-ID-held-out threshold `0.6022372841835022`, AUROC `0.9299874433205441`, AUPRC `0.5245233707288152`, F1 `0.5338491295938105`; Drug-ID-held-out threshold `0.8456150889396667`, AUROC `0.7334477219355293`, AUPRC `0.16723035384474041`, F1 `0.1`.
- `python revision/promote_canonical_entity_metrics.py --bootstrap-replicates 600` promoted the verified canonical prediction files and metrics to the public result JSON/CSV files and `prediction_summary.json` before regeneration of Figure 2.
