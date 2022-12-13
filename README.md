# E(3) Steerable GNN in jax
Reimplementation of [SEGNN](https://arxiv.org/abs/2110.02905) in jax. Original work by Johannes Brandstetter, Rob Hesselink, Elise van der Pol, Erik Bekkers and Max Welling.

## Installation
To install only the model requirements run
```
python -m pip install -r requirements.txt
```

The validation experiments are adapted from the original implementation, so additionally `torch` and `torch_geometric` are needed.
```
python -m pip install -r experiments/requirements.txt
```

## Experiments and datasets
The N-body (charged and gravity) and QM9 datasets are included for completeness from the original paper.

QM9 is automatically downloaded and processed when running the respective experiment. Note that for QM9 the batch size is just for convenience and is not actually the number of graphs in the batch, as the batches are filled with molecules to reduce the padding graphs.

The N-body datasets have to be generated locally from the directory [experiments/nbody/data](experiments/nbody/data)
#### Charged dataset (5 bodies, 10000 training samples)
```
python3 -u generate_dataset.py --simulation=charged
```
#### Gravity dataset (100 bodies, 10000 training samples)
```
python3 -u generate_dataset.py --simulation=gravity --n-balls=100
```
## Validation
<table>
  <tr>
    <td></td>
    <td colspan="2"><b>torch (original)</b></td>
    <td colspan="2"><b>jax (ours)</b></td>
  </tr>
  <tr>
    <td></td>
    <td>MSE</td>
    <td>Inference [ms]*</td>
    <td>MSE</td>
    <td>Inference [ms]</td>
  </tr>
  <tr>
    <td> <code>charged (position)</code> </td>
    <td>.0043</td>
    <td>40.76</td>
    <td>.0047</td>
    <td><b>28.67</td>
  </tr>
  <tr>
    <td><code>gravity** (position)</code> </td>
    <td>.265</td>
    <td>392.20</td>
    <td>.28</td>
    <td><b>240.34</td>
  </tr>
  <!-- <tr>
    <td> <code>QM9 (alpha)</code> </td>
    <td>.06</td>
    <td></td>
    <td></td>
    <td>180.85</td>
  </tr> -->
</table>
*remeasured (Quadro RTX 4000), batch of 100 graphs, single precision

** 5 neighbors

## Usage
### Nbody
#### Charged experiment
```
python3 -u main.py --dataset=charged --epochs=200 --max-samples=3000 --lmax-hidden=1 --lmax-attributes=1 --layers=4 --units=64 --norm=none --batch-size=100 --lr=5e-4 --weight-decay=1e-8
```
#### Gravity experiment
```
python3 -u main.py --dataset=gravity --epochs=100 --target=pos --max-samples=10000 --lmax-hidden=1 --lmax-attributes=1 --layers=4 --units=64 --norm=none --batch-size=100 --lr=1e-4 --weight-decay=1e-8 --neighbours=5 --n-bodies=100
```

<!-- #### QM9
```
python3 -u main.py --dataset=qm9 --epochs=500 --target=alpha --lmax-hidden=2 --lmax-attributes=3 --layers=7 --units=128 --norm=instance --batch-size=30 --lr=2e-4 --weight-decay=1e-8
``` -->


(configurations used in validation)
