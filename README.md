# Learning to grok: Emergence of in-context learning and skill composition in tree structure reasoning

## This is an official implementation of the paper [arXiv:2406.02550](https://arxiv.org/abs/2406.02550).

![Main Result](assets/fig1.png)


## Dataset generation

To generate raw training set, run
```bash
python src/ICL/datasets/generate_raw.py
```


To generate eval set, run
```bash
python src/ICL/datasets/generate_eval.py
```

## Train model
To train the transformer model, please update parameters to override in the ```src/ICL/model/conf/yourconf.yaml```
```bash
python src/ICL/model/train.py --config PATH_TO_YOUR_CONFIG_YAML
```


## Evaluation metric
We implement part of the  evalution task to colelct all the results across exp settings
```bash
python src/ICL/eval/collection.py 
```




## Requirements

- Tested Environment:
  - Python == 3.11.4
  - PyTorch >= 2.2
- Run ```pip install -r requirements.txt``` before running experiments
- Run ```pip install pip install -e .``` every time updating codes in the ```ICL``` package

