#!/bin/bash

if [ "$1" == "" ]; then
    echo "No model named passed as a parameter to the script. Exiting"
    exit
else
    echo "Fetching model $1 and deploying to triton server, stand by..."
fi
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)
echo "Found $NUM_GPUS gpu(s) via nvidia-smi"

index_slash=`expr index "$1" "/"`
if [ $index_slash > 0 ]; then
  model_name=$1
  len=${#model_name}
  model_name=${model_name:index_slash:len}
else
  model_name=$1
fi

docker pull nvcr.io/nvidia/pytorch:22.05-py3
docker run --name torch-env --rm -it -d --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:22.05-py3
docker exec -it torch-env pip install transformers
docker exec -it torch-env python codegen_gptj_convert.py --code_model=${1} --output_dir=/workspace/models/${1}
docker exec -it torch-env python huggingface_gptj_convert.py -in_file=/workspace/models/${1} -saved_dir=/workspace/triton-models/${model_name}/1 -infer_gpu_num ${NUM_GPUS}
docker exec -it torch-env wget -P /workspace/triton-models/${model_name}/ https://huggingface.co/${1}/raw/main/config.pbtxt

docker stop torch-env

docker run --name triton-server --gpus all -it -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/triton-models:/model moyix/triton_with_ft:22.06 bash -c "CUDA_VISIBLE_DEVICES=${GPUS} mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver --model-repository=/model"
