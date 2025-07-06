import os
import yaml
import json
import numpy as np
from loguru import logger as printer


# ========== YAML Utilities ==========
def readYaml(path):
    if not os.path.exists(path):
        printer.error(f"YAML file not found: {path}")
        return None
    try:
        with open(path, 'r') as stream:
            parsed_yaml = yaml.safe_load(stream)
            printer.info(f"YAML file loaded: {path}")
            return parsed_yaml
    except yaml.YAMLError as exc:
        printer.error(f"Error reading YAML file {path}: {exc}")
        return None


# ========== JSON Utilities ==========
def readJson(path):
    if not os.path.exists(path):
        printer.error(f"JSON file not found: {path}")
        return None
    try:
        with open(path, 'r') as stream:
            data = json.load(stream)
            printer.info(f"JSON file loaded: {path}")
            return data
    except json.JSONDecodeError as e:
        printer.error(f"JSON decoding error in file {path}: {e}")
        return None
    except Exception as e:
        printer.error(f"Error reading JSON file {path}: {e}")
        return None


def writeJson(data, path):
    try:
        with open(path, "w") as outfile:
            json.dump(data, outfile, indent=4)
        printer.info(f"JSON file written: {path}")
    except Exception as e:
        printer.error(f"Error writing JSON file {path}: {e}")


# ========== Early Stopper ==========
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): Number of consecutive evaluations with no improvement to wait before stopping.
            min_delta (float): Minimum change in validation loss to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """
        Returns:
            bool: True if training should stop early.
        """
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            printer.info(f"Validation loss improved to {validation_loss:.6f}")
        else:
            self.counter += 1
            printer.warning(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                printer.error("Early stopping triggered!")
                return True
        return False
