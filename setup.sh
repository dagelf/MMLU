#!/bin/bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/")"

wget https://people.eecs.berkeley.edu/~hendrycks/data.tar

tar -xvf data.tar
