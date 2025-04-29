# CDFSL-test

## Arguments
- `m`: models to train / baseline, prompt, clsmod, posmod, clsmod_pospermute, prototype_contextualization, lowlayer_contextualization .
- `d`: miniimagenet, BSCD, FWT, all
- `checkpointdir`: checkpoint direction to test.
- `continual_layers` : only used in lowlayer_contextualization, select low layer to contextualize.

## Requirements
- python 3.8.20
- torch 2.0.1
- torchvision 0.15.2
- tqdm 4.66.5


## Example Execution Scripts

### train
```bash
python main.py -m 'bntuning' -tr -tc 'fewshot' -d 'miniimagenet' -e 100 -lr 0.01 -elr 1e-4 -bs 256 -opt 'adamW' -log '_name_' -img_size 224 -patch_size 16 -sched 'cosine'
```

### Test 
```bash
python main.py -m 'bntuning' -tc 'crossdomain' -d 'BSCD' -e 100 -log 'name' -img_size 224 -patch_size 16 -checkdir '_checkpointdir_.pt'
```

