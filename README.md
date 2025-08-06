# Data augmentation using LLM and subsequent finetuning

If you're in a slurm environment you need to load these modules. Make sure you have lmod or any other hierarchical module naming scheme

```bash
module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load DeepSpeed/0.14.5-CUDA-12.1.1
```

Create a python virtual environment with the given `requirements.txt` file and source it

Put the input data in `dataset` directory

steps to run

```bash
sbatch scripts/finetune_<model_name>_<dataset_type>.sh
```