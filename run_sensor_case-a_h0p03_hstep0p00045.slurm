#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                # MB
#SBATCH --time=1-12             # MINUTES, DAYS-HOURS

#SBATCH --job-name=MaMMoS_min_case-b_benchmark2
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
SIMDIR=$HOME/slurm_$SLURM_JOB_ID
echo "JOBNAME:" $SLURM_JOB_NAME
echo "PARTITION:" $SLURM_JOB_PARTITION
echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
echo "ALLOC. NODES:" $SLURMD_NODENAME
echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR
echo "SIMDIR:" $SIMDIR

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# Set mesh size (h) and hstep values
# When using backup files for the initial state, make sure they exist for the chosen parameters
# available files are named like: 
#   - backup_mesh_sensor_mesh_h0p005_hstep0p00025.npz
#   - backup_mesh_sensor_mesh_h0p004_hstep0p00045.npz
#   - backup_mesh_sensor_mesh_h0p01_hstep0p00045.npz
# and backup_sensor_mesh_h0p005_hstep0p00025.0002.state.npz
# ============================================================================
# MESH_H has to match available initial state files !
# MESH_H="0.007"
MESH_H="0.03"
# MESH_H="0.01"
# HSTEP can be choosen freely
# HSTEP="0.00025"
HSTEP="0.00045"
HSTEP_INI="0.00025"
# Choose cases to run: a, b, c or combinations like ab, ac, bc, abc
# CASES="a b c"
CASES="a"
# ============================================================================

# Set variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PIXI_PROJECT_ROOT=$SLURM_SUBMIT_DIR
# Tag for result collection directory (UTC timestamp + slurm id + parameters)
RESULT_TAG=$(date -u +"%Y%m%dT%H%M%SZ")_slurm_${SLURM_JOB_ID}_h${MESH_H}_hstep${HSTEP}_case${CASES}
RESULT_BASE=$SLURM_SUBMIT_DIR/examples/sensor_loop/results/$RESULT_TAG

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

# Create working directory
mkdir -p $SIMDIR
cd $SIMDIR
echo "WORKING DIR:" $PWD

cp -r $SLURM_SUBMIT_DIR/src .
mkdir -p examples/sensor_loop/
cp -r $SLURM_SUBMIT_DIR/examples/sensor_loop/sensor_loop_step_by_step.py ./examples/sensor_loop/
cp -r $SLURM_SUBMIT_DIR/examples/sensor_loop/sensor_loop_evaluation.py ./examples/sensor_loop/

# Create sensor_initial_state folder and copy ONLY the required backup files + any existing sensor.* files
mkdir -p examples/sensor_loop/sensor_initial_state
echo "Copying backup files for initial state (h=${MESH_H}, hstep=${HSTEP_INI})..."
cp $SLURM_SUBMIT_DIR/examples/sensor_loop/sensor_initial_state/backup_mesh_sensor_mesh_h${MESH_H}_hstep${HSTEP_INI}.npz ./examples/sensor_loop/sensor_initial_state/ 2>/dev/null && echo "  ✓ Backup mesh copied" || echo "  ⚠ Backup mesh not found (will be generated if needed)"
cp $SLURM_SUBMIT_DIR/examples/sensor_loop/sensor_initial_state/backup_sensor_mesh_h${MESH_H}_hstep${HSTEP_INI}.0002.state.npz ./examples/sensor_loop/sensor_initial_state/ 2>/dev/null && echo "  ✓ Backup state copied" || echo "  ⚠ Backup state not found (will be computed if needed)"

# Also copy any existing sensor.* files from initial state dir (config and mesh files)
echo "Copying existing sensor.* files from initial state directory..."
for file in $SLURM_SUBMIT_DIR/examples/sensor_loop/sensor_initial_state/sensor.*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Skip old state files and .dat files (these will be regenerated)
        if [[ ! "$filename" =~ \.state\.npz$ ]] && [[ "$filename" != "sensor.dat" ]]; then
            cp "$file" ./examples/sensor_loop/sensor_initial_state/
            echo "  ✓ Copied $filename"
        fi
    fi
done

# Copy case folders (only sensor.* files: .p2, .krn, .npz, excluding old output states and .dat)
echo "Preparing case directories..."
for CASE_DIR in sensor_case-a_down sensor_case-a_up sensor_case-b_down sensor_case-b_up sensor_case-c_down sensor_case-c_up; do
    mkdir -p examples/sensor_loop/$CASE_DIR
    # Copy all sensor.* files (configuration and mesh), but skip old output files
    # Include: sensor.p2, sensor.krn, sensor.npz
    # Exclude: sensor.*.state.npz (old outputs), sensor.dat (old outputs)
    for file in $SLURM_SUBMIT_DIR/examples/sensor_loop/$CASE_DIR/sensor.*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            # Skip state files and .dat files (these are old outputs to be regenerated)
            if [[ ! "$filename" =~ \.state\.npz$ ]] && [[ "$filename" != "sensor.dat" ]]; then
                cp "$file" ./examples/sensor_loop/$CASE_DIR/
            fi
        fi
    done
done

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

echo "Before RUNNING SIMULATION : sensor loop step by step.py"
echo "SIMULATION PARAMETERS: MESH_H=$MESH_H, HSTEP=$HSTEP, CASES=$CASES"
if [ "$USE_PIXI" = false ]; then
    $MAMBA_EXE run -n $ENV_NAME python -u ./examples/sensor_loop/sensor_loop_step_by_step.py --hstep $HSTEP --cases $CASES --initial-state-file backup_sensor_mesh_h${MESH_H}_hstep${HSTEP_INI}.0002.state.npz --initial-mesh-file backup_mesh_sensor_mesh_h${MESH_H}_hstep${HSTEP_INI}.npz
else
    find examples/sensor_loop/ -type f -name '*.p2' -exec grep -Hn -i -E 'h[_ ]?step' {} \;
    find examples/sensor_loop/sensor_initial_state -type f -name '*.npz' 
    $PIXI_EXE run --manifest-path $SLURM_SUBMIT_DIR/pixi.toml python -u ./examples/sensor_loop/sensor_loop_step_by_step.py --hstep $HSTEP --cases $CASES --initial-state-file backup_sensor_mesh_h${MESH_H}_hstep${HSTEP_INI}.0002.state.npz --initial-mesh-file backup_mesh_sensor_mesh_h${MESH_H}_hstep${HSTEP_INI}.npz
    find examples/sensor_loop/ -type f -name '*.p2' -exec grep -Hn -i -E 'h[_ ]?step' {} \;
fi
echo "After RUNNING SIMULATION : sensor loop step by step.py"

echo "Before RUNNING EVALUATION : sensor loop evaluation.py"
if [ "$USE_PIXI" = false ]; then
    $MAMBA_EXE run -n $ENV_NAME python -u ./examples/sensor_loop/sensor_loop_evaluation.py --cases $CASES --include-mesh-h $MESH_H --include-params --info "Sensor loop evaluation for SLURM job $SLURM_JOB_ID with resulttag $RESULT_TAG"
else
    $PIXI_EXE run --manifest-path $SLURM_SUBMIT_DIR/pixi.toml python -u ./examples/sensor_loop/sensor_loop_evaluation.py --cases $CASES --include-mesh-h $MESH_H --include-params --info "Sensor loop evaluation for SLURM job $SLURM_JOB_ID with resulttag $RESULT_TAG"
fi
echo "After RUNNING EVALUATION : sensor loop evaluation.py"

# Copy files back into timestamped results folder
mkdir -p $RESULT_BASE

copy_back_dir() {
    src_dir=$1
    dst_dir=$2
    mkdir -p "$dst_dir"
    cp -r "$src_dir"/* "$dst_dir" 2>/dev/null || true
}

echo "Copying results to $RESULT_BASE"
copy_back_dir ./examples/sensor_loop/sensor_initial_state $RESULT_BASE/sensor_initial_state
copy_back_dir ./examples/sensor_loop/sensor_case-a_down $RESULT_BASE/sensor_case-a_down
copy_back_dir ./examples/sensor_loop/sensor_case-a_up $RESULT_BASE/sensor_case-a_up
copy_back_dir ./examples/sensor_loop/sensor_case-b_down $RESULT_BASE/sensor_case-b_down
copy_back_dir ./examples/sensor_loop/sensor_case-b_up $RESULT_BASE/sensor_case-b_up
copy_back_dir ./examples/sensor_loop/sensor_case-c_down $RESULT_BASE/sensor_case-c_down
copy_back_dir ./examples/sensor_loop/sensor_case-c_up $RESULT_BASE/sensor_case-c_up
cp ./examples/sensor_loop/*.dat $RESULT_BASE/ 2>/dev/null || true
cp ./examples/sensor_loop/*.png $RESULT_BASE/ 2>/dev/null || true
cp ./examples/sensor_loop/*.log $RESULT_BASE/ 2>/dev/null || true
cp ./examples/sensor_loop/*.py $RESULT_BASE/ 2>/dev/null || true
echo "Results copied to $RESULT_BASE"

# Clean up, remove working directory
cd ..
rm -r $SIMDIR

# Show end time
END_TIME=$(date +%s)
echo "END TIME:" $(date -u +"%Y-%m-%dT%H:%M:%SZ") "($END_TIME)"

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))
echo "ELAPSED TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s"
