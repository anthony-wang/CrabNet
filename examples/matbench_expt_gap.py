import numpy as np
import pandas as pd
from crabnet.crabnet_ import CrabNet
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

for task in mb.tasks:
    task.load()
    for fold in task.folds:

        # Inputs are either chemical compositions as strings
        # or crystal structures as pymatgen.Structure objects.
        # Outputs are either floats (regression tasks) or bools (classification tasks)
        train_inputs, train_outputs = task.get_train_and_val_data(fold)

        train_df = pd.DataFrame({"formula": train_inputs, "target": train_outputs})

        # Get testing data
        test_inputs = task.get_test_data(fold, include_target=False)
        test_df = pd.DataFrame(
            {"formula": test_inputs, "target": np.zeros(test_inputs.shape[0])}
        )

        crab = CrabNet(epochs=10, learningcurve=False, losscurve=False)
        crab.fit(train_df)
        predictions = crab.predict(test_df)

        # Record your data!
        task.record(fold, predictions)

# Save your results
mb.to_file("matbench_expt_gap.json.gz")
1 + 1
