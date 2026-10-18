# Sourced by every train_fred_<backbone>_<dataset>.sh shim.
#   default VAR value   sets VAR to value unless the caller already exported it
#   launch              hands over to train_fred.sh with the exported settings
set -euo pipefail
export BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
default() { export "$1=${!1:-$2}"; }
launch()  { exec bash "$BASE_DIR/run_scripts/train/train_fred.sh"; }
