
# 运行容器
```
         srun -N 1  --pty --gres=gpu:4090:1 docker run --gpus all \
           -v /home/dataset/tmp/:/home/chenyidong \
           -v ./cache:/app/cache \
           -v ./results:/app/results \
           -d --name tilus-artifact docker.1ms.run/yyding/tilus-artifacts:latest
```
# 进入容器

```
srun -N 1 --pty --gres=gpu:4090:1 docker exec -it  2555ed0cba20 bash
```

# 加载环境

```
pip uninstall marlin
cd marlin
export TORCH_CUDA_ARCH_LIST='9.0;8.0;8.9;8.6'
python setup.py install
pip uninstall vllm
pip install gemlite
pip install hqq
export PYTHONPATH=/home/chenyidong/tilus-artifacts/artifacts/tilelp
```

# 实验
1. 不同bit
bash runbit.sh

2. 不同shape

bash runfigure9.sh

3. 不同的batch

bash runbatch.sh

4. breakdown

bash runbreakdown.sh

5.  end2end

需要补充
bash runend2end.sh

6. 在AMD GPU上运行
需要补充
