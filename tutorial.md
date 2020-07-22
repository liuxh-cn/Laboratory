
### Anaconda 的安装与使用

### Juypter-Notebook 添加 Anaconda 虚拟环境的 kernel

```
## 创建虚拟环境，并安装 ipykernel

# 创建虚拟环境
conda create -name [env-name] python=3.7
# 查看所有虚拟环境、删除指定虚拟环境
conda info -e                             
conda remove -n [env-name] --all

# 激活(进入)创建的虚拟环境，并安装 ipykernel
conda activate [env-name]
conda install ipykernel

## 将 Anaconda 的虚拟环境配置到 jupyter-notebook 中

# jupyter-notebook 配置虚拟环境的 kernel (需先进入对应的虚拟环境中 conda activate [env-name])
python -m ipykernel install --user --name [env-name]
# 查看jupyter配置的所有kernel
jupyter kernelspec list
```

###  为jupyter notebook安装jupyter_contrib_nbextensions
#### 安装
https://www.jianshu.com/p/6efc8e412397
#### 配置
tab - Hinderland

#### jupyter 引用外部py文件
https://blog.csdn.net/weixin_40999066/article/details/105509245

### Git 的日常使用

git 功能强大，可以满足很多种的需求，但这同时也带来了过于庞大的命令集和繁琐的操作。如果我们只是想满足实验室各人的代码版本控制和代码共享，则只需要记忆很少的命令即可。

#### Git 安装
#### Git 初始化本地仓库
#### Git 日常操作
