# 虚假招聘信息风险识别系统 (Fake Job Risk Identification System)

本项目为武汉大学国家网络安全学院《人工智能》课程论文配套实验工程。项目基于大规模招聘数据，利用深度学习（MLP）与传统机器学习算法（逻辑回归、随机森林等）构建了一套针对求职场景的虚假职位识别与风险预警系统。

## 🚀 项目亮点
- **深度非线性融合**：通过多层感知机（MLP）有效融合了高维 TF-IDF 文本特征与结构化标签。
- **详尽的实验分析**：包含完整的超参数敏感性分析（学习率、Dropout）、训练曲线监控及过拟合判定。
- **工业级错误分析**：针对误报（FP）和漏报（FN）样本进行了多维统计，揭示了欺诈职位的“背书盗用”和“信息稀疏”模式。
- **高工程规范性**：项目结构清晰，实验数据全记录，支持一键复现。

## 📂 目录结构
```text
FakeJobRisk_Project/
├── data/
│   ├── processed/          # 最终的 train.csv, val.csv, test.csv
│   └── raw/                # 原始数据集 (fake_job_postings.csv)
├── results/               
│   ├── figures/            # 论文引用的核心图表 (最优模型曲线、混淆矩阵)
│   ├── tables/             # 论文引用的核心指标汇总与错误分析案例
│   ├── reports/            # 自动生成的预测错误诊断报告
│   └── _experiments_log/   # 实验存档：包含几十次调参的原始记录 (CSV & PNG)
├── src/                    # 核心代码
│   ├── archive/            # 版本迭代过程中的旧版脚本
│   ├── utils/              # 数据检查与自动汇总工具
│   ├── train_mlp_ce.py     # 主训练程序 (支持命令行调参)
│   ├── baseline_compare.py # 自动化 Baseline 对比工具
│   └── error_analysis.py   # 错误样本深度挖掘脚本
├── requirements.txt        # 环境依赖清单
└── README.md

## 📊 实验结果预览
1. 模型性能对比 (Test Set)
模型	Accuracy	Precision (Fraud)	Recall (Fraud)	Macro-F1
MLP (最优)	0.9883	0.9517	0.7977	0.9309
逻辑回归	0.9678	0.6115	0.9191	0.8586
多数类投票	0.9516	0.0000	0.0000	0.4876
2. 核心结论
MLP 优势：在类别极度不平衡（1:19）的情况下，MLP 通过深度特征提取，在保证高召回率的同时，显著降低了误报率（由逻辑回归的 0.4 降至 0.05 以下）。
参数敏感性：学习率 lr=0.001 是模型收敛的临界点，过大（0.1）会导致梯度崩溃。
🛠️ 环境复现
安装依赖：
pip install -r requirements.txt
数据预处理：
python src/split_data.py
运行完整实验流：
# 训练所有对比模型
python src/train_baseline_new.py
# 训练最优 MLP 模型
python src/train_mlp_ce.py --lr 0.001 --dropout 0.3 --model_name mlp_ce_lr0001
生成分析报告：
python src/error_analysis.py --model_name mlp_ce_lr0001
python src/baseline_compare.py
🎓 致谢
感谢武汉大学国家网络安全学院提供的实验指导。本项目遵循学术诚信原则，代码中引用的第三方库均已在 requirements.txt 中标注。
