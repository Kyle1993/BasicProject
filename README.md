# Basic Project  
## File Organization  
```
├── checkpoints/
│
├── data/
│   └──  dataset.py
│
├── models/
│   └── Model.py
│
├── utils/
│   └── visualize.py
│
├── config.py
├── train.py
└── README.md
```  



* checkpoints/： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练  
* data/：数据相关操作，包括数据预处理、dataset实现等  
models/：模型定义，可以有多个模型  
* utils/：可能用到的工具函数  
* config.py：配置文件，所有可配置的变量都集中在此，并提供默认值  
* train.py：训练流程  
