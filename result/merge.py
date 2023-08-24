# %%
import pandas as pd
import numpy as np

# %%
chrono_1 = pd.read_csv('./ucr_chrono.csv')
chrono_2 = pd.read_csv('./chrono_merge.csv')

# %%
collapse = chrono_1.query('f1 == 0')
for row in collapse.values:
    dataset, model, label, acc, f1, recall, precision = row
    
    row_ = chrono_2.query('dataset == @dataset and label == @label')
    chrono_1.update(row_)

# %%
chrono_1.query('f1 == 0')

chrono_1.to_csv('chrono_merged.csv', index=False)