# midas_nse/config.py  — DO NOT RUN DIRECTLY, only import this
import os

PROJECT_ROOT = r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse"
RAW  = os.path.join(PROJECT_ROOT, "data", "raw")
EXT  = os.path.join(PROJECT_ROOT, "data", "external")
PROC = os.path.join(PROJECT_ROOT, "data", "processed")
LOGS = os.path.join(PROJECT_ROOT, "logs")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints")