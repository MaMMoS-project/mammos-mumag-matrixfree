#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                # MB
#SBATCH --time=1-12             # MINUTES, DAYS-HOURS

#SBATCH --job-name=MaMMoS_benchmark2
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

# show some information
START_TIME=$(date +%s)
echo "START TIME:" $(date -u +"%Y-%m-%dT%H:%M:%SZ") "($START_TIME)"
SIMDIR=$HOME/slurm_$SLURM_JOB_ID
echo "JOBNAME:" $SLURM_JOB_NAME
echo "PARTITION:" $SLURM_JOB_PARTITION
echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
echo "ALLOC. NODES:" $SLURMD_NODENAME
echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR
echo "SIMDIR:" $SIMDIR

# set variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PIXI_PROJECT_ROOT=$SLURM_SUBMIT_DIR

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# Choose between pixi (default) or mamba (legacy) environment management
# Set USE_PIXI=true to use pixi environment (recommended)
# Set USE_PIXI=false to use the old mamba environment setup
# ============================================================================

USE_PIXI=true

if [ "$USE_PIXI" = false ]; then
    # ========== MAMBA ENVIRONMENT SETUP (Legacy) ==========
    # This section uses the original mamba-based approach to run simulations
    # Requires: mamba/micromamba environment to be configured
    export MAMBA_ROOT_PREFIX='/scandium/home/programs/micromamba/micromamba'
    export MAMBA_EXE='/scandium/home/programs/micromamba/bin/micromamba'
    ENV_NAME="mfree-mumag-gpu"
    
    if [ ! -x "$MAMBA_EXE" ]; then
        echo "ERROR: mamba executable not found at $MAMBA_EXE"
        exit 1
    fi
    echo "ENVIRONMENT: Using Mamba environment '$ENV_NAME'"
    echo "MAMBA EXECUTABLE: $MAMBA_EXE"
    
else
    # ========== PIXI ENVIRONMENT SETUP (Recommended) ==========
    # This section uses pixi for reproducible environment management
    # Requires: pixi installed and pixi.toml in SLURM_SUBMIT_DIR
    # Advantages: Better reproducibility, lighter weight, self-contained
    
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
fi

# create working directory
mkdir -p $SIMDIR
cd $SIMDIR
echo "WORKING DIR:" $PWD

cp -r $SLURM_SUBMIT_DIR/src .
mkdir examples/
cp -r $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/ ./examples/
cp -r $SLURM_SUBMIT_DIR/*.py .
cp -r $SLURM_SUBMIT_DIR/*.npz .
# cp -r $SLURM_SUBMIT_DIR/*.yml .

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

echo "Before RUNNING SIMULATION : sensor loop step by step.py"
if [ "$USE_PIXI" = false ]; then
    $MAMBA_EXE run -n $ENV_NAME python -u ./examples/sensor_loop_step_by_step/sensor_loop_step_by_step.py
else
    $PIXI_EXE run --manifest-path $SLURM_SUBMIT_DIR/pixi.toml python -u ./examples/sensor_loop_step_by_step/sensor_loop_step_by_step.py
fi
echo "After RUNNING SIMULATION : sensor loop step by step.py"

echo "Before RUNNING EVALUATION : sensor loop evaluation.py"
if [ "$USE_PIXI" = false ]; then
    $MAMBA_EXE run -n $ENV_NAME python -u ./examples/sensor_loop_step_by_step/sensor_loop_evaluation.py
else
    $PIXI_EXE run --manifest-path $SLURM_SUBMIT_DIR/pixi.toml python -u ./examples/sensor_loop_step_by_step/sensor_loop_evaluation.py
fi
echo "After RUNNING EVALUATION : sensor loop evaluation.py"

# copy files back if necessary
cp ./examples/sensor_loop_step_by_step/sensor_initial_state/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_initial_state/
cp ./examples/sensor_loop_step_by_step/sensor_initial_state/sensor.npz $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_initial_state/
cp ./examples/sensor_loop_step_by_step/sensor_initial_state/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_initial_state/
cp ./examples/sensor_loop_step_by_step/sensor_case-a_down/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-a_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-a_down/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-a_down/

cp ./examples/sensor_loop_step_by_step/sensor_case-b_down/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-b_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-b_down/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-b_down/

cp ./examples/sensor_loop_step_by_step/sensor_case-c_down/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-c_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-c_down/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-c_down/

cp ./examples/sensor_loop_step_by_step/sensor_case-a_down/*.vtu $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-a_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-b_down/*.vtu $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-b_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-c_down/*.vtu $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-c_down/
# Copy all sensor.p2 files back 
cp ./examples/sensor_loop_step_by_step/sensor_case-a_down/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-a_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-a_up/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-a_up/
cp ./examples/sensor_loop_step_by_step/sensor_case-b_down/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-b_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-b_up/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-b_up/
cp ./examples/sensor_loop_step_by_step/sensor_case-c_down/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-c_down/
cp ./examples/sensor_loop_step_by_step/sensor_case-c_up/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-c_up/

# Number changes with settings in .p2 cp ./examples/sensor_loop_step_by_step/sensor_case-a_down/sensor.0002.state.npz $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_case-a_down/

cp ./examples/sensor_loop_step_by_step/*.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/
cp ./examples/sensor_loop_step_by_step/*.png $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/
cp ./examples/sensor_loop_step_by_step/*.log $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/

# clean up, remove working directory
cd ..
rm -r $SIMDIR

# show end time
END_TIME=$(date +%s)
echo "END TIME:" $(date -u +"%Y-%m-%dT%H:%M:%SZ") "($END_TIME)"

ELAPSED=$((END_TIME - START_TIME))
echo "ELAPSED TIME: $((ELAPSED / 60)) minutes and $((ELAPSED % 60)) seconds"
