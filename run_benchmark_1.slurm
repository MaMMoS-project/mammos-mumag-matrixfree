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
SIMDIR=$HOME/slurm_$SLURM_JOB_ID
echo "JOBNAME:" $SLURM_JOB_NAME
echo "PARTITION:" $SLURM_JOB_PARTITION
echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
echo "ALLOC. NODES:" $SLURMD_NODENAME
echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR
echo "SIMDIR:" $SIMDIR

# ============================================================================
# BENCHMARK 1 PARAMETERS
# ============================================================================
# Mesh configuration
# GRAINS="8"              # Number of grains (default: 8)
# EXTENT="80,80,80"               # Custom extent "Lx,Ly,Lz" or empty for default
# USE_MINIMAL=false       # Set to true for minimal extent (20x20x20), false for full (80x80x80)
# TOL="0.01"              # Tolerance for make_krn.py (default: 0.01)

GRAINS="4"              # Number of grains (default: 8)
EXTENT="80,80,80"               # Custom extent "Lx,Ly,Lz" or empty for default
USE_MINIMAL=true       # Set to true for minimal extent (20x20x20), false for full (80x80x80)
TOL="0.05"              # Tolerance for make_krn.py (default: 0.01)

                        # With <=5 grains use >=0.05; avoid >0.2

# Workflow parameters
NUM_REPEATS="3"        # Number of workflow iterations for averaging (default: 1)
# NUM_REPEATS="10"        # Number of workflow iterations for averaging (default: 1)

# Tag for result collection directory (UTC timestamp + slurm id + parameters)
RESULT_TAG=$(date -u +"%Y%m%dT%H%M%SZ")_slurm_${SLURM_JOB_ID}_grains${GRAINS}_repeats${NUM_REPEATS}
RESULT_BASE=$SLURM_SUBMIT_DIR/examples/benchmark_1/$RESULT_TAG
# ============================================================================

# Set environment variables
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
fi

# Create working directory
mkdir -p $SIMDIR
cd $SIMDIR
echo "WORKING DIR:" $PWD

# Copy required files
echo "Copying project files to working directory..."
cp -r $SLURM_SUBMIT_DIR/src .
mkdir -p examples/benchmark_1/isotrop_down
cp $SLURM_SUBMIT_DIR/examples/benchmark_1/benchmark1_workflow.py ./examples/benchmark_1/
cp $SLURM_SUBMIT_DIR/examples/benchmark_1/isotrop_down/isotrop.p2 ./examples/benchmark_1/isotrop_down/ 2>/dev/null || echo "  ⚠ isotrop.p2 not found in isotrop_down/ - will need to be created"
mkdir -p examples/benchmark_1/isotrop_up
cp $SLURM_SUBMIT_DIR/examples/benchmark_1/isotrop_up/isotrop.p2 ./examples/benchmark_1/isotrop_up/ 2>/dev/null || echo "  ⚠ isotrop.p2 not found in isotrop_down/ - will need to be created"
echo "  ✓ Project files copied"

# ============================================================================
# RUN BENCHMARK 1 WORKFLOW
# ============================================================================

# Build command line arguments
WORKFLOW_ARGS="--repeats $NUM_REPEATS --tol $TOL"

if [ -n "$GRAINS" ]; then
    WORKFLOW_ARGS="$WORKFLOW_ARGS --grains $GRAINS"
fi

if [ -n "$EXTENT" ]; then
    WORKFLOW_ARGS="$WORKFLOW_ARGS --extent $EXTENT"
elif [ "$USE_MINIMAL" = true ]; then
    WORKFLOW_ARGS="$WORKFLOW_ARGS --minimal"
fi

echo "================================================================================"
echo "RUNNING BENCHMARK 1 WORKFLOW"
echo "================================================================================"
echo "PARAMETERS:"
echo "  Grains:      $GRAINS"
echo "  Extent:      ${EXTENT:-$([ "$USE_MINIMAL" = true ] && echo "minimal (20x20x20)" || echo "full (80x80x80)")}"
echo "  Tolerance:   $TOL"
echo "  Repeats:     $NUM_REPEATS"
echo "  Arguments:   $WORKFLOW_ARGS"
echo "================================================================================"

if [ "$USE_PIXI" = false ]; then
    $MAMBA_EXE run -n $ENV_NAME python -u ./examples/benchmark_1/benchmark1_workflow.py $WORKFLOW_ARGS
else
    $PIXI_EXE run --manifest-path $SLURM_SUBMIT_DIR/pixi.toml python -u ./examples/benchmark_1/benchmark1_workflow.py $WORKFLOW_ARGS
fi

echo "================================================================================"
echo "BENCHMARK 1 WORKFLOW COMPLETED"
echo "================================================================================"

# Copy results back to timestamped results folder
mkdir -p $RESULT_BASE

echo "Copying results to $RESULT_BASE..."
cp -r ./examples/benchmark_1/results/* $RESULT_BASE/ 2>/dev/null || echo "  ⚠ No results directory found"
cp -r ./examples/benchmark_1/isotrop_down $RESULT_BASE/ 2>/dev/null || echo "  ⚠ No isotrop_down directory found"
cp -r ./examples/benchmark_1/isotrop_up $RESULT_BASE/ 2>/dev/null || echo "  ⚠ No isotrop_up directory found"
cp ./examples/benchmark_1/benchmark1_workflow.py $RESULT_BASE/ 2>/dev/null || true

echo "  ✓ Results copied to $RESULT_BASE"

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
