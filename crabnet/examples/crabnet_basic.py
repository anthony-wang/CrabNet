"""Basic usage of CrabNet regression on elasticity dataset."""
from crabnet.model import get_data
from crabnet.data.materials_data import elasticity

train_df, val_df = get_data(elasticity, "train.csv", dummy=True)

from crabnet.crabnet_ import CrabNet

cb = CrabNet(mat_prop="elasticity")
cb.fit(train_df)
val_pred, val_sigma = cb.predict(val_df)
