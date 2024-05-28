import os
import subprocess
from datetime import datetime

import pandas as pd
import pytz

import wandb

# class WandBLogger:
#     """WandB logger."""

#     def __init__(self, args, project_name="llm-rs"):
#         self.logger = wandb.init(
#             project = project_name,
#             config = vars(args),
#         )

#     def log(self, dict):
#         self.logger.log(dict)

#     def finish(self):
#         self.logger.finish()

class WandBLogger:
    """WandB logger."""

    def __init__(self, args, project_name="llm-rs"):
        # Set W&B to offline mode
        os.environ['WANDB_MODE'] = 'offline'

        # Initialize W&B run
        self.logger = wandb.init(
            project=project_name,
            config=vars(args),
        )
        self.run_id = self.logger.id  # Store the run ID
        self.run_path = self.logger.dir  # Store the run directory path

    def log(self, log_dict):
        self.logger.log(log_dict)

    def finish(self):
        self.logger.finish()
        # Sync the specific run after training
        path_wo_files = self.run_path.rsplit("/", 1)[0]
        print(f"Syncing run ID {self.run_id} from path {path_wo_files}")
        subprocess.run(["wandb", "sync", path_wo_files])


