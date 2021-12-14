# pyFiDEL

This is the official repo for our paper ["Learning from Fermions: the Fermi-Dirac Distribution Provides a Calibrated Probabilitic Output for Binary Classifiers"](https://www.pnas.org/content/118/34/e2100761118) published in PNAS 2021. Here, the python implementation of the Fermi-Dirac ensemble learning (FiDEL) method is included.

Contact: sungcheol.kim78[at]gmail[dot]com

## Installation

- required packages: numpy, pandas, scipy, seaborn 

```{bash}
> git clone https://github.com/sungcheolkim78/pyFiDEL.git
> pip3 install -e .
```

## Usage for ensemble method (FiDEL)

## Usage for optimal threshold

## Usage for confidence interval 

## FiDEL result

## For paper figures

### Figure 1 in PNAS

```{python}
import matplotlib.pyplot as plt

from pyFiDEL import SimClassifier, PCR

# create simulator
c = SimClassifier(N=10000, rho=.5)

# generate Gaussian score with target AUC
score = c.create_gaussian_scores(auc0=.9)

# create pcr data
p = PCR(c.score, c.y, sample_size=100, sample_n=1000)

# plot PCR distribution
plt.plot(p.pcr, '.')
```

### Figure 2 in PNAS

```{python}
import numpy as np
from pyFiDEL.ranks import build_correpond_table

auclist = np.linspace(0.52, 0.98, num=47)
rholist = np.linspace(.1, .9, num=17)

# calculate beta, mu from auc, rho
df = build_correpond_table(auclist, rholist, resol=.00001, method='root')

# plot in 3D space
```

### Figure 3 in PNAS


