from pathlib import Path

import yaml

with open("model_editing/model_editor/rome_style/util/globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
