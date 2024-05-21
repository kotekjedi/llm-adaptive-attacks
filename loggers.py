import os
from datetime import datetime

import pandas as pd
import pytz

import wandb


class WandBLogger:
    """WandB logger."""

    def __init__(self, args, project_name="llm-rs"):
        self.logger = wandb.init(
            project = project_name,
            config = vars(args),
        )

    def log(self, dict):
        self.logger.log(dict)

    def finish(self):
        self.logger.finish()

