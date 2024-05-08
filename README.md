# pyFiDEL (Fermi-Dirac Ensemble Learner)

This is the official repo for our paper ["Learning from Fermions: the Fermi-Dirac Distribution Provides a Calibrated Probabilitic Output for Binary Classifiers"](https://www.pnas.org/content/118/34/e2100761118) published in PNAS 2021. Here, the python implementation of the Fermi-Dirac ensemble learning (FiDEL) method is included.

Contact: sungcheol.kim78[at]gmail[dot]com

## Installation

with `Poetry`, 

```sh
git clone https://github.com/sungcheolkim78/pyFiDEL.git

pip3 install poetry
poetry install
pip3 install -e .
```

with `pip`,

```sh
git clone https://github.com/sungcheolkim78/pyFiDEL.git
cd pyFiDEL
pip install -r requirements.txt
pip install -e .
```

with `pip install -e .` you can install this package with live link to the git repository.

## Quick start

You can see the examples in `notebooks/Tutorial.ipynb` file. 

Here, we create a simulated classifier with target AUC score. And we compare the class 
probabilty at given rank from simulation to the Fermi-Dirac distribution. 

```python
from pyFiDEL import SimClassifier

c = SimClassifier(N=10000, rho=0.5)
score = c.create_gaussian_scores(auc0=0.9, tol=1e-4)

p = PCR(c.score, c.y, sample_size=100, sample_n=500)
df, info = p.build_metric()
```

Here, we test the ensemble method based on Fermi-Dirac distribution. 

```python
from pyFiDEL import SimClassifier, FiDEL
import numpy as np

c = SimClassifier(N=1000, rho=0.7)

auc_list = np.linspace(0.55, 0.65, num=30)
predictions = c.create_predictions(n_methods=30, auc_list=auc_list)

f = FiDEL()
f.add_predictions(c.pred)
f.add_label(c.y)
f.calculate_performance(alpha=1.0)
df = f.df
```

You can find more examples in `docs/cookbook.md`

- Usage for ensemble method (FiDEL)
- Usage for optimal threshold
- Usage for confidence interval 
