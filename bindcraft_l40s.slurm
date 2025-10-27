#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem 42gb
#SBATCH --time 72:00:00
#SBATCH --output=binder_design_%A.log
#SBATCH --mail-user=martin.pacesa@epfl.ch
#SBATCH --mail-type=ALL

# Initialise environment and modules
#source /work/lpdi/users/mpacesa/Pipelines/miniforge3/bin/activate /work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/FastBC
source /work/lpdi/users/mpacesa/Pipelines/miniforge3/bin/activate /work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma
module load gcc/13.2
module load cuda/12.4.1
module load cudnn/8.9.7.29-12

# alternatively you can source the environment directly
#source /path/to/mambaforge/bin/activate /path/to/mambaforge/envs/BindCraft

# Get the directory where the bindcraft script is located
SCRIPT_DIR=$(dirname "$0")

# Parsing command line options
SETTINGS=""
FILTERS=""
ADVANCED=""
TEMP=$(getopt -o s:f:a: --long settings:,filters:,advanced: -n 'bindcraft.slurm' -- "$@")
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -s|--settings) SETTINGS="$2" ; shift 2 ;;
        -f|--filters) FILTERS="$2" ; shift 2 ;;
        -a|--advanced) ADVANCED="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

echo "Running the BindCraft pipeline"
echo "Running binder design for target ${SETTINGS}"
echo "Design settings used: ${ADVANCED}"
echo "Filtering designs based on ${FILTERS}"
python -u "/work/lpdi/users/mpacesa/Pipelines/FastBC/bindcraft.py" --settings "${SETTINGS}" --filters "${FILTERS}" --advanced "${ADVANCED}"
