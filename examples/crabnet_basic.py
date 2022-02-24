"""Basic usage of CrabNet regression on elasticity dataset."""
from crabnet.utils.data import get_data
from crabnet.data.materials_data import elasticity
from crabnet.crabnet_ import CrabNet

train_df, val_df = get_data(elasticity, "train.csv", dummy=True)

cb = CrabNet(mat_prop="elasticity")
cb.fit(train_df)
val_pred, val_sigma = cb.predict(val_df, return_uncertainty=True)
