
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
cd /home/chenyidong/tilus-artifacts/artifacts
export GPUID=0
rm -rf cache/mutis && rm -rf cache/bitblas && rm -r /root/.cache/ && rm -rf ./results/figure9.txt
```

# 实验

1. figure1

bash run_figure1.sh

2. 不同shape

bash run_shape.sh


3. 不同 bit

bash run_bit.sh
