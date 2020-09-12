import re
from pathlib import Path

import numpy as np
from rich import print
from rich.table import Table

from lib.tools.classification import calc_stable_core

base_dir = Path('./downstream_classification/results/compare_embedding_errors')
total_diff_files = list(base_dir.glob('total_diff/*.txt'))

table = Table()
table.add_column('Filename')
table.add_column('Old')
table.add_column('New')
table.add_column('Improvement')

for total_diff_file in total_diff_files:
    basename = total_diff_file.name
    corresponding_prediction_name = base_dir / 'predictions' / str(basename).replace('.txt', '.npy')

    predictions = np.load(corresponding_prediction_name)
    stable_core = calc_stable_core(predictions)

    with total_diff_file.open() as file:
        content = file.readlines()

    old_core = int(re.search('Stable Core: (\d+)', content[0]).group(1))
    improvement = (stable_core / old_core) if old_core != 0 else 'Undefined'
    table.add_row(basename, str(old_core), str(stable_core), str(improvement))

    content[0] = content[0].replace(f'Stable Core: {old_core}', f'Stable Core: {stable_core}')
    with total_diff_file.open('w') as file:
        file.writelines(content)

print(table)
