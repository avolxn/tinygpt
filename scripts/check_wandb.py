"""Verify mandatory W&B authentication before launching training jobs."""

import argparse

from tinygpt.tracking import require_wandb_auth

parser = argparse.ArgumentParser(description="Verify W&B authentication")
parser.add_argument(
    "--non-interactive",
    action="store_true",
    help="Require preconfigured credentials instead of prompting",
)
args = parser.parse_args()

require_wandb_auth(interactive=not args.non_interactive)
print("W&B authentication verified.")
