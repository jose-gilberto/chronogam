# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%

metrics = pd.read_csv('../scripts/metrics/chrono/CricketX/3.0/lightning_logs/version_0/metrics.csv')
metrics.head()

# %%

sns.lineplot(metrics.train_energy[:400])

# %%

sns.lineplot(metrics.train_energy[400:])

# %%

metrics.train_energy[0]