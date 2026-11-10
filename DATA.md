# Data

Set `RAT_DATA` to the folder that holds the raw datasets (default `data/` in the repository root) and `RAT_BANKS` to the folder that holds the frozen-encoder banks (default `data/banks/`).

## Standard multi-view sets (`$RAT_DATA`)

| Key | File(s) | Source |
|---|---|---|
| handwritten, scene15, caltech20, landuse, leaves, outdoor | `HandWritten.mat`, `Scene15.mat`, `Caltech101-20.mat`, `LandUse_21.mat`, `100Leaves.mat`, `OutdoorScene.mat` | the multi-view `.mat` files distributed with the trusted multi-view classification code (TMC, ETMC, RCML) |
| pie | `rat/ext/RCML/data/PIE_face_10.mat` | the RCML repository (installed by `scripts/setup_external.sh`) |
| brca, rosmap | `rat/ext/MOGONET/{BRCA,ROSMAP}/` | the MOGONET repository (installed by `scripts/setup_external.sh`); official train/test split |
| mosi, urfunny, mustard | `mosi-*/mosi_data.pkl`, `humor-*/humor/humor.pkl`, `sarcasm-*/sarcasm.pkl` | MultiBench pre-extracted features (official test split); the folder names are listed in `MB` in `rat/mv_bench.py` |

## Frozen-encoder banks (`$RAT_BANKS`)

| Bank | Raw data | Builder |
|---|---|---|
| `sun.npz` | SUN RGB-D (`SUNRGBD/`) | `python features/audit_extract.py --ds sun --root $RAT_DATA` |
| `milk.npz` | MILK10k training release (`MILK10k/`, 5,240 lesions) | `python features/audit_extract.py --ds milk --root $RAT_DATA` |
| `food.npz` | UPMC Food-101 (`UPMC-Food101/`) | `python features/audit_extract.py --ds food --root $RAT_DATA` |
| `crema.npz` | CREMA-D (audio and video) | `python features/crema_seq_extract.py --out banks/crema_seq.npz`, then `python features/crema_clip_bank.py` |

Encoders: DINOv2-base (images and video frames), BERT-base-uncased (text), wav2vec 2.0-base (audio), fp16, from the Hugging Face hub. The bank builders apply the corruption families to raw signals before encoding. They draw the bank splits once with seed 0. MD5 checksums of the banks used in the paper:

```
1ed81d5149d38108cd9f0cb4ab0c6a49  sun.npz
bc827c0fa0ffbbf015e2ecf3245b5db4  milk.npz
a50033254a2f0c92f8f9add30bd7fabd  food.npz
480c89915cf87e5689eb9276a6942483  crema.npz
```
