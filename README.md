## Installation

`pip install -e .` 

### Download Checkpoints

First, we need to download a model checkpoint. All the model basse checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
```
You can download finetuned checkpoint to checkpoint folder via:

(checkpoint)[https://drive.google.com/file/d/13Nx5mK8HXu4CKKb6NMacQBMK_urt2oEf/view?usp=sharing]

## Run inference
Replace `RGB_FOLDER`, `DEPTH_FOLDER` with your actual rgb and depth images folder

Run `python task3.py`, the output should be at `visualize_output_private` folder

## To replica the final score:
Download the dataset provided by Viettel with custom annotations via this script:
```bash
curl -L "https://app.roboflow.com/ds/kAeC7gTF3v?key=dZMrU03MK1" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
Then run this script:
```
python training/train.py -c configs/train_large21.yaml --use-cluster 0 --num-gpus 1
```


