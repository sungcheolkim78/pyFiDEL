# How to generate figures in PNAS paper?

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
