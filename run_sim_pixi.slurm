#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                # MB
#SBATCH --time=1-12             # MINUTES, DAYS-HOURS

#SBATCH --job-name=MaMMoS_benchmark1
#
# Number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --nodelist=La
#
#SBATCH --export=NONE
# By default all environment variables of the shell invoking the sbatch command are propagated.
# This may cause unexpected behaviour as for example $HOME used in this script might be different
# from the expected /home/<username>. Also consider $PATH, $OCL* or $CUDA* variables. I recommend
# to set --export=NONE to avoid the propagation.

# Show some information
START_TIME=$(date +%s)
echo "START TIME:" $(date -u +"%Y-%m-%dT%H:%M:%SZ") "($START_TIME)"
echo "JOB_NAME:" $SLURM_JOB_NAME
echo "PARTITION:" $SLURM_JOB_PARTITION
echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
echo "ALLOC. NODES:" $SLURMD_NODENAME
echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR

# ========== USER PARAMETERS ==========
SIM_DIR=$HOME/slurm-$SLURM_JOB_ID
echo "SIM_DIR:" $SIM_DIR
if [ ! -d "$SIM_DIR" ]; then
  mkdir -p "$SIM_DIR"
fi

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PIXI_PROJECT_ROOT=$SLURM_SUBMIT_DIR

# ========== PIXI ENVIRONMENT SETUP (Recommended) ==========
# This section uses pixi for reproducible environment management
# Requires: pixi installed and pixi.toml in SLURM_SUBMIT_DIR

# Determine the real home directory (handle cases where $HOME differs between compute nodes)
# This is important when home directories are mounted differently (e.g., /home vs /ceph/home)
REAL_HOME=$(eval echo ~$(whoami))

# Set path to pixi executable by checking multiple possible locations
# This makes the script portable across different systems and installations
PIXI_EXE=""
for pixi_path in \
    "$(command -v pixi 2>/dev/null)" \
    "$REAL_HOME/.pixi/bin/pixi" \
    "$HOME/.pixi/bin/pixi" \
    "$REAL_HOME/.local/bin/pixi" \
    "$HOME/.local/bin/pixi" \
    "/usr/local/bin/pixi" \
    "/ceph/home/$(whoami)/.pixi/bin/pixi"
do
    if [ -n "$pixi_path" ] && [ -f "$pixi_path" ]; then
        PIXI_EXE="$pixi_path"
        break
    fi
done

if [ -z "$PIXI_EXE" ]; then
    echo "ERROR: pixi executable not found!"
    echo "Searched locations:"
    echo "  - command -v pixi"
    echo "  - $REAL_HOME/.pixi/bin/pixi"
    echo "  - $HOME/.pixi/bin/pixi"
    echo "  - $REAL_HOME/.local/bin/pixi"
    echo "  - $HOME/.local/bin/pixi"
    echo "  - /usr/local/bin/pixi"
    echo "  - /ceph/home/$(whoami)/.pixi/bin/pixi"
    echo ""
    echo "To fix this:"
    echo "  1. Install pixi: curl -sSf https://pixi.sh | bash"
    echo "  2. Or set USE_PIXI=false in this script to use mamba instead"
    exit 1
fi
echo "ENVIRONMENT: Using Pixi with manifest at $SLURM_SUBMIT_DIR/pixi.toml"
echo "PIXI EXECUTABLE: $PIXI_EXE"

# ---------- Pixi environment activation & diagnostics ----------
# Derive the pixi environment root from the active python inside pixi
PIXI_ENV_ROOT=$("$PIXI_EXE" run --manifest-path "$SLURM_SUBMIT_DIR/pixi.toml" python -c 'import sys; print(sys.prefix)')
# Prepend its bin to PATH for any direct tool calls in this script
export PATH="$PIXI_ENV_ROOT/bin:$PATH"
# Ensure no hidden Neper defaults are injected
unset NEPEROPT

echo "PIXI ENV ROOT: $PIXI_ENV_ROOT"
echo "PATH: $PATH"
# Confirm gmsh/neper are available from the pixi environment
echo "DIAGNOSTICS:"
"$PIXI_EXE" run --manifest-path "$SLURM_SUBMIT_DIR/pixi.toml" which gmsh || true
"$PIXI_EXE" run --manifest-path "$SLURM_SUBMIT_DIR/pixi.toml" gmsh -version || true
"$PIXI_EXE" run --manifest-path "$SLURM_SUBMIT_DIR/pixi.toml" which neper || true
# If Neper path detection fails, you can force gmsh via NEPEROPT (uncomment the next two lines)
# GMSH_BIN=$("$PIXI_EXE" run --manifest-path "$SLURM_SUBMIT_DIR/pixi.toml" which gmsh)
# export NEPEROPT="-gmsh $GMSH_BIN"

# Copy required files
echo "Copying project files to working directory..."
cp -r $SLURM_SUBMIT_DIR/src .
cp -r $SLURM_SUBMIT_DIR/resources .


# ============================================================================
# RUN 
# ============================================================================
cd $SIM_DIR
echo "WORKING DIR:" $PWD


$PIXI_EXE run --manifest-path $SLURM_SUBMIT_DIR/pixi.toml python -u src/loop.py --mesh resources/two_grains_interface/two_grains_interface.npz


# Clean up, remove working directory
if [[ "$SIM_DIR" != "$SLURM_SUBMIT_DIR" ]]; then
  RESULT_DIR="$SLURM_SUBMIT_DIR"/results
  mkdir -p "$RESULT_DIR"
  cp -r "$SIM_DIR" "$RESULT_DIR"
  mv "$SLURM_SUBMIT_DIR"/slurm-"$SLURM_JOB_ID".out  slurm-$SLURM_JOB_ID
  cd $SLURM_SUBMIT_DIR
  rm -r $SIM_DIR
fi



# Show end time
END_TIME=$(date +%s)
echo "END TIME:" $(date -u +"%Y-%m-%dT%H:%M:%SZ") "($END_TIME)"

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))
echo "ELAPSED TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s"