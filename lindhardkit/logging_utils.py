#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logging_utils.py — Centralized logging and runtime banners for NESTOR
=====================================================================
Provides standardized logging setup, author information banners, and
run-time announcements for the NESTOR electronic susceptibility toolkit.
Ensures consistent formatting, UTF-8-safe terminal output, and timestamped
log files for reproducible simulations.

Purpose
--------
•  Create timestamped log files with unified format for all NESTOR modules.  
•  Mirror messages simultaneously to the console (stdout) and disk.  
•  Remove stale log files at startup for clean session bookkeeping.  
•  Provide standard author information and startup banners for reproducibility
   and citation acknowledgement.

Main functions
---------------
- **setup_logger(name='lindhard')**  
    Configure and return a `logging.Logger` with both file and console handlers.  
    Log files are named `run_YYYY-MM-DD_HHMMSS.log` and encoded in UTF-8.

- **print_author_info(logger)**  
    Print formatted author and citation metadata to the log (UTF-8 box characters).

- **banner(logger)**  
    Display a standardized start banner for CDW/Lindhard susceptibility runs.

Logging format
---------------
- **Message format:**  
    `%(asctime)s  %(levelname)8s: %(message)s`  
- **Timestamp format:**  
    `%Y-%m-%d %H:%M:%S`  
- **Output:** both console and `run_*.log` file in the working directory.

Features
---------
•  UTF-8-safe output for Unicode box-drawing characters.  
•  Automatic cleanup of previous `run_*.log` files to reduce clutter.  
•  Explicit non-propagating logger to avoid duplicate stream outputs.  
•  Compatible with all major Python versions ≥ 3.10.  

Author:
    Chinedu E. Ekuma  
    Department of Physics, Lehigh University, Bethlehem, PA, USA

Contributors:
    Chinedu E. Ekuma  
    Chidiebere Nwaogbo

License:
    MIT License (see LICENSE file)

Version:
    1.0.0
"""


from __future__ import annotations
import logging, sys, datetime, glob, os

# Message format uses logging fields; asctime will be formatted by DATE_FMT below.
LOG_FMT  = "%(asctime)s  %(levelname)8s: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str = "lindhard") -> logging.Logger:
    # Clean up old logs in cwd
    for old in glob.glob("run_*.log"):
        try:
            os.remove(old)
        except OSError:
            pass

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # File handler (ensure UTF-8 so box-drawing chars print correctly)
    fh = logging.FileHandler(f"run_{stamp}.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))

    # Console handler to stdout (UTF-8 on modern terminals)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))

    # Reset and attach
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def print_author_info(logger: logging.Logger) -> None:
    lines = [
        "═" * 70,
        " Author     : Prof. Chinedu E. Ekuma",
        " Affiliation: Lehigh University, Bethlehem PA 18015",
        " Contact    : cekuma@lehigh.edu",
        " Homepage   : https://physics.lehigh.edu/~cekuma",
        "",
        " If you have used this code in your calculations, please cite:",
        "    Ekuma, C. E., *et al.* \"Title of Article.\" Journal Name, Year.",
        "    DOI: xx.xxxx/xxxxxxx",
        "═" * 70,
    ]
    for line in lines:
        logger.info(line)

def banner(logger: logging.Logger) -> None:
    logger.info("═"*70)
    logger.info(" Starting CDW Lindhard Susceptibility Calculation ")
    logger.info("═"*70)

