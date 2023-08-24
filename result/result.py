# %%
import pandas as pd

# %%
chrono = pd.read_csv('./ucr_chrono.csv')
chrono.accuracy.mean(), chrono.f1.mean()

# %%
chrono_old = pd.read_csv('./ucr_chrono_old.csv')
chrono_old.model = 'chrono_gam_old'
chrono_old.accuracy.mean(), chrono_old.f1.mean()


# %%
dagmm = pd.read_csv('./ucr_dagmm.csv')
dagmm.accuracy.mean(), dagmm.f1.mean()

# %%
deepsvdd = pd.read_csv('./ucr_deepsvdd.csv')
deepsvdd.accuracy.mean(), deepsvdd.f1.mean()

# %%
ocsvm = pd.read_csv('./ucr_ocsvm.csv')
ocsvm.accuracy.mean(), ocsvm.f1.mean()

# %%
isolation_forest = pd.read_csv('./ucr_isolation_forest.csv')
isolation_forest.accuracy.mean(), isolation_forest.f1.mean()


# %%

results = pd.concat([chrono, chrono_old, dagmm, deepsvdd, ocsvm, isolation_forest])
results.label = results.label.astype(str)

final_results = pd.DataFrame()

final_results['classifier_name'] = results.model
final_results['dataset_name'] = results.dataset.str.cat(results.label, sep='-')
final_results['f1'] = results.f1
final_results['accuracy'] = results.accuracy

final_results.to_csv('all_results.csv', index=False)
