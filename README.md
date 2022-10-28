# An Equal-Size Hard EM Algorithm for Diverse Dialogue Generation

## Training

To train a single-decoder model, use base_trainer.py:
```bash
python base_train.py \
    --train-path=path-to-the-training-csv-file \
    --val-path=path-to-the-validation-csv-file \
    --model-str=t5-small \
```
The resulting checkpoint is used as the initial checkpoint for multi-decoder training.
For Weibo, use --model-str=uer/t5-small-chinese-cluecorpussmall, --language=zh, and --multi-ref in addition.

To train a multi-decoder model, use the script multi-decoder_trainer.py:
```bash
python multi-decoder_trainer.py \
    --train-path=path-to-the-training-csv-file \
    --val-path=path-to-the-validation-csv-file \
    --model-str=t5-small \
    --init-ckpt=path-to-warmstart-ckpt \
    --freeze \
    --num-modes=10 \
    --trainer=eqhem \
    --decoder=adapter \
```
where
1. --num-modes specifies the number of decoders
2. --trainer specifies the training algorithm. eqhem for EqHard-EM, sem for Soft-EM, hem for Hard-EM, random for EqRandom-Fixed, and drandom for EqRandom-Dynamic.
3. --decoder specifies the decoder architecture.

## Monitoring Performance

The training script will automatically generate a timestamped logging directory to store the checkpoints as well as log files.
The validation performance can be monitored during training through tensorboard:
```
tensorboard --logdir=path-to-the-timestamped-logging-folder
```

## Continue Training

If the performance is still increasing at the end of training, you can resume with the following command
```bash
python train.py \
    --the-original-arguments-that-you-started-training-with
    --resume-path=path-to-the-timestamped-logging-folder
```

## Evaluation

After the performance has peaked, you can evaluate the model using evaluate_generations.py:
```bash
python evaluate_generations.py --ckpt-path=path-to-the-best-validation-checkpoint --eval-path=path-to-the-test-csv-file
```
Additionally use --language=zh and --multi-ref for evaluating on Weibo.

## Datasets
[OST](https://github.com/yq-wen/overlapping-datasets), [Weibo](https://drive.google.com/file/d/1KX-34q9kx6i9tqhyHH6jfMSR-68u_3Mn/view?usp=sharing)
