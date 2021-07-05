# SDNE(Graph Embedding)
pytorch Based, Structural Deep Network Embedding

## 修改部分
由于别人都是直接生成拉普拉斯矩阵和领结矩阵，但对于单机器而言，不适用于大规模图数据，因此对领结矩阵和拉普拉斯矩阵部分进行修改。


## 复现代码

```
python3 run.py 
```

## 参数设置

```
  -h, --help            show this help message and exit
  --batch_size N        input batch size for training (default: 1024)
  --epochs N            number of epochs to train (default: 5)
  --data S              which data (default: pid)
  -a N, --alpha N       Parameters that control the 1st order loss(default:
                        1e-5)
  -b N, --beta N        The parameters controlling the second-order loss have
                        higher penalty coefficients for non-zero
                        elements(default: 5)
  --v N                 Controls the parameters of the regularization
                        term(default: 1e-5)
  --lr N                Optimizer parameter(default: 1e-3)
  --method METHOD       Classify_method : node classify(n)/link(lp).(default:
                        n)
  -sf N, --sample_frac N
                        Test size
  --cuda                enables CUDA training(default: False)
  --seed S              random seed (default: 1)
  --mode S              The type of graph(default: Di)
  --save S              Whether or not to save(default: y)
  --train S             Train or predict(default: y)
  --module N            Which model to choose(default: SDNE)
  --log-interval N      how many batches to wait before logging training
                        status(default: 1024)
```

## 原始数据

参数：

- 数据集：wiki
- epoch：5/10/20/50
- 其他参数均一样



|          | base_epoch5 | base_epoch10 | base_epoch20 | base_epoch50 | rec_epoch5 | rec_epoch10 | rec_epoch20 | rec_epoch50 |
| -------- | ----------- | ------------ | ------------ | ------------ | ---------- | ----------- | ----------- | ----------- |
| micro_f1 | 0.4137      | 0.4927       | 0.6216       | 0.5717       | 0.5841     | 0.6267      | **0.7190**  | 0.7072      |
| macro_f2 | 0.2788      | 0.3417       | 0.4479       | 0.4187       | 0.3640     | 0.4208      | **0.5154**  | 0.5510      |

**小点分析**：重构代码效果正常，增加了正则化损失在该数据集更有用

