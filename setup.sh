export PYTHONPATH=/home/chenyidong/tilus-artifacts/artifacts/tilelp

export GPUID=0


# cd /opt/conda/envs/titus-artifacts/lib
# mv libstdc++.so.6 libstdc++.so.6.conda.backup
# ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

# cd - 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cuda/lib:/cuda/lib64

export PATH=$PATH:/opt/conda/envs/titus-artifacts/bin:/opt/conda/condabin:/opt/conda/bin:/cuda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

