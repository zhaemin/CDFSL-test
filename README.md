# CDFSL-test

## Arguments
- `m`: models to train / baseline, prompt, clsmod, posmod, clsmod_pospermute, prototype_contextualization, lowlayer_contextualization .
- 'd' : miniimagenet, BSCD, FWT, all
- `checkpointdir`: checkpoint direction to test.
- 'continual_layers' : only used in lowlayer_contextualization, select low layer to contextualize.


## Example Execution Scripts

### train
```bash
python main.py -m 'clsmod' -tr -tc 'fewshot' -d 'miniimagenet' -e 100 -lr 0.01 -bs 256 -opt 'adamW' -log '' -img_size 224 -patch_size 16 
```

### Test 2
```bash
python main.py -m 'clsmod' -tc 'crossdomain' -d 'BSCD'  -log '' -img_size 224 -patch_size 16 -checkpointdir 'clsmod_100ep_0.01lr.pt'
```

