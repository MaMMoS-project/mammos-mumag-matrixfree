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
export MAMBA_ROOT_PREFIX='/scandium/home/programs/micromamba/micromamba'
export MAMBA_EXE='/scandium/home/programs/micromamba/bin/micromamba'

# create working directory

mkdir -p $SIMDIR
cd $SIMDIR
echo "WORKING DIR:" $PWD

cp -r $SLURM_SUBMIT_DIR/src .
mkdir examples/
cp -r $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/ ./examples/
cp -r $SLURM_SUBMIT_DIR/*.py .
cp -r $SLURM_SUBMIT_DIR/*.npz .
cp -r $SLURM_SUBMIT_DIR/*.yml .

# compute demagnetization curve
# # $MAMBA_EXE run -n mfree-mumag-gpu python loop.py --mesh sensor

echo "Before RUNNING SIMULATION : sensor loop step by step.py"
$MAMBA_EXE run -n mfree-mumag-gpu python -u ./examples/sensor_loop_step_by_step/sensor_loop_step_by_step.py
echo "After RUNNING SIMULATION : sensor loop step by step.py"

# directly run the loop with initial state working:
# $MAMBA_EXE run -n mfree-mumag-gpu python ./src/loop.py --mesh ./examples/sensor_loop_step_by_step/sensor_loop_initial_state/sensor

$MAMBA_EXE run -n mfree-mumag-gpu python -u ./examples/sensor_loop_step_by_step/sensor_loop_evaluation.py

# copy files back if necessary
cp ./examples/sensor_loop_step_by_step/sensor_loop_initial_state/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_initial_state/
cp ./examples/sensor_loop_step_by_step/sensor_loop_initial_state/sensor.npz $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_initial_state/
cp ./examples/sensor_loop_step_by_step/sensor_loop_initial_state/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_initial_state/
cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/
cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/

cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-b/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-b/
cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-b/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-b/

cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-c/sensor.dat $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-c/
cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-c/sensor.p2 $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-c/

cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/*.vtu $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/
cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-b/*.vtu $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-b/
cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-c/*.vtu $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-c/
# Number changes with settings in .p2 cp ./examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/sensor.0002.state.npz $SLURM_SUBMIT_DIR/examples/sensor_loop_step_by_step/sensor_loop_only_down_case-a/

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
