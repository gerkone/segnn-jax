# Steerable E(3) GNN in jax
Reimplementation of [SEGNN](https://arxiv.org/abs/2110.02905) in jax. Original work by Johannes Brandstetter, Rob Hesselink, Elise van der Pol, Erik Bekkers and Max Welling.

## Why jax?
**40-50% faster** inference and training compared to the [original torch implementation](https://github.com/RobDHess/Steerable-E3-GNN). Also JAX-MD.

## Installation
```
python -m pip install segnn-jax
```

Or clone this repository and build locally
```
python -m pip install -e .
```

### GPU support
Upgrade `jax` to the gpu version
```
pip install --upgrade "jax[cuda]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Validation
N-body (charged and gravity) and QM9 datasets are included for completeness from the original paper.

### Results
Charged is on 5 bodies, gravity on 100 bodies. QM9 has graphs of variable sizes, so in jax samples are padded to the maximum size. Loss is MSE for Charged and Gravity and MAE for QM9.

Times are remeasured on Quadro RTX 4000, __model only__ on batches of 100 graphs, in (global) single precision.

<table>
  <tr>
    <td></td>
    <td colspan="2"><b>torch (original)</b></td>
    <td colspan="2"><b>jax (ours)</b></td>
  </tr>
  <tr>
    <td></td>
    <td>Loss</td>
    <td>Inference [ms]</td>
    <td>Loss</td>
    <td>Inference [ms]</td>
  </tr>
  <tr>
    <td> <code>charged (position)</code> </td>
    <td>.0043</td>
    <td>21.22</td>
    <td>.0045</td>
    <td>4.47</td>
  </tr>
  <tr>
    <td><code>gravity (position)</code> </td>
    <td>.265</td>
    <td>60.55</td>
    <td>.264</td>
    <td>41.72</td>
  </tr>
  <tr>
    <td> <code>QM9 (alpha)</code> </td>
    <td>.066*</td>
    <td>82.53</td>
    <td>.082</td>
    <td>105.98**</td>
  </tr>
</table>
* rerun on same conditions

** padded (naive)

### Validation install

The experiments are only included in the github repo, so it needs to be cloned first.
```
git clone https://github.com/gerkone/segnn-jax
```

They are adapted from the original implementation, so additionally `torch` and `torch_geometric` are needed (cpu versions are enough).
```
python -m pip install -r experiments/requirements.txt
```

### Datasets
QM9 is automatically downloaded and processed when running the respective experiment.

The N-body datasets have to be generated locally from the directory [experiments/nbody/data](experiments/nbody/data) (it will take some time, especially n-body `gravity`)
#### Charged dataset (5 bodies, 10000 training samples)
```
python3 -u generate_dataset.py --simulation=charged
```
#### Gravity dataset (100 bodies, 10000 training samples)
```
python3 -u generate_dataset.py --simulation=gravity --n-balls=100
```

### Usage
#### N-body (charged)
```
python validate.py --dataset=charged --epochs=200 --max-samples=3000 --lmax-hidden=1 --lmax-attributes=1 --layers=4 --units=64 --norm=none --batch-size=100 --lr=5e-3 --weight-decay=1e-12
```

#### N-body (gravity)
```
python validate.py --dataset=gravity --epochs=100 --target=pos --max-samples=10000 --lmax-hidden=1 --lmax-attributes=1 --layers=4 --units=64 --norm=none --batch-size=100 --lr=5e-3 --weight-decay=1e-12 --neighbours=5 --n-bodies=100
```

#### QM9
```
python validate.py --dataset=qm9 --epochs=1000 --target=alpha --lmax-hidden=2 --lmax-attributes=3 --layers=7 --units=128 --norm=instance --batch-size=128 --lr=5e-4 --weight-decay=1e-8 --lr-scheduling
```

(configurations used in validation)


## Acknowledgments
- [e3nn_jax](https://github.com/e3nn/e3nn-jax) made this reimplementation possible.
- [Artur Toshev](https://github.com/arturtoshev) and [Johannes Brandsetter](https://github.com/brandstetter-johannes), for support.
