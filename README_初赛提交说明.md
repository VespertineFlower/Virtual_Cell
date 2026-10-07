# GOAI 赛道三：虚拟酵母初赛提交包

## 项目方法

本项目采用面向数据充足度的分层预测框架：

1. **Condition MLP 主干**：预测 4,232 维绝对 log2 蛋白丰度。
2. **门控 ES-IFM 支路**：分解化合物、菌株及低秩交互响应，并在官方训练集内部构造伪冷启动任务学习场景门控。
3. **小样本结构核分支**：使用 Morgan、MACCS 和理化描述符，通过逐药留一选择的强正则 RBF 核预测未见化合物响应。
4. **六模块代理与提交审计**：按官方场景检查 FC、绝对保真度、残差相关性、双重未知/时间插值和 DEP，并验证提交合同。

## 目录

- `01_方案文档/`：初赛方案文档。
- `02_源代码/src/vc_stage01/`：公共数据、特征、模型、损失和评估实现。
- `02_源代码/workflows/01_数据合同与MLP主干/`：数据合同、MLP 训练和测试推理。
- `02_源代码/workflows/02_门控ES_IFM/`：ES-IFM-v4、融合诊断和训练内门控。
- `02_源代码/workflows/03_小样本结构核/`：公开结构特征、小样本核回归和官方新化合物验证。
- `02_源代码/workflows/04_评估与提交/`：六模块代理、Matched Control 诊断和提交校验。
- `03_结果证据/`：精简的指标表、门控权重和审计记录；不包含大型预测矩阵。

## 数据放置

赛事原始数据不随源码重复分发。请将四个官方 CSV 放到 `02_源代码/data/`：

- `WAYB_WAYC_metadata_train_val(1).csv`
- `WAYB_WAYC_proteome_raw_train_val.csv`
- `WAYB_WAYC_metadata_test(1).csv`
- `WAYB_WAYC_proteome_raw_test.csv`（仅允许用于公开自评或合同核验，不得参与训练与统计量估计）

## 安装

在 `02_源代码` 目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 主干训练与推理

以下命令均从 `02_源代码` 目录运行：

```powershell
python ".\workflows\01_数据合同与MLP主干\run_stage0.py" --data-dir ".\data"
python ".\workflows\01_数据合同与MLP主干\run_stage1.py"
python ".\workflows\01_数据合同与MLP主干\generate_submission.py" --data-dir ".\data"
```

Stage 0 仅以 `split_final=train` 计算蛋白缺失率，采用严格阈值 `<0.697` 并冻结 4,232 维 feature contract。Stage 1 的随机种子、网络宽度、早停和损失参数均写入检查点与训练日志。

## 门控 ES-IFM

ES-IFM-v4 的完整实验脚本位于 `workflows/02_门控ES_IFM/`。主结果使用非 episodic 版本，并以训练内伪新化合物、伪新菌株和伪双未知任务估计门控：

```powershell
python ".\workflows\02_门控ES_IFM\baseline10_es_ifm_v4.py" --no-episodic
python ".\workflows\02_门控ES_IFM\train_internal_fusion_gate.py" --project-dir "."
```

冻结门控为：新化合物 `0.373497`、新菌株 `0.120257`、双重未知 `0.186644`、时间插值使用全局门控 `0.154011`。官方验证标签不参与门控估计。

## 小样本结构核

首次使用公开资源时可运行特征阶段；提交包同时保留了 PubChem 结构快照和来源字段：

```powershell
python ".\workflows\03_小样本结构核\run_feature_stage.py" --data-dir ".\data" --resolve-pubchem
python ".\workflows\03_小样本结构核\chemical_cold_start_kernel.py" `
  --data-dir ".\data" `
  --strain-features ".\outputs\features\strain_sequence_embeddings.npz" `
  --chemical-features ".\outputs\features\chemical_features.npz"
python ".\workflows\03_小样本结构核\chemical_structure_official.py" `
  --project-dir "." `
  --chemical-features ".\outputs\features\chemical_features.npz"
```

小样本评估采用完全未见化合物、20 次重复和训练化合物留一选择，不使用评估化合物标签选参。

## 评估与提交校验

```powershell
python ".\workflows\04_评估与提交\run_official_proxy.py"
python ".\workflows\04_评估与提交\validate_submission.py" `
  ".\outputs\stage1\prediction.csv" `
  --metadata ".\data\WAYB_WAYC_metadata_test(1).csv" `
  --contract ".\resources\feature_contract.json"
python -m unittest discover -s tests -v
```

最终文件必须为绝对 `log2` 丰度，包含 4,454 个测试 `sample_ID`，蛋白列严格遵循冻结的 4,232 维 feature contract，且不得出现 NA 或 Inf。

## 外部资源披露

- 化合物结构：PubChem PUG REST，本地快照日期 2026-08-13。
- 结构处理：RDKit 2026.03 系列，生成 Morgan、MACCS 与理化描述符。
- 名称别名和固定 CID 映射：见 `workflows/03_小样本结构核/fetch_pubchem_smiles.py`。
- 结构查询结果及状态：见 `resources/chemical_smiles.csv`。
- 未使用商业 API、闭源预测模型或验证/测试蛋白质组标签进行训练。

## 结果边界

训练内门控融合在本地六模块代理上由 Condition MLP 的 `0.71690` 提升到 `0.71955`，四个官方验证场景 RMSE 均改善。该分数是本地代理，不等同于组委会最终成绩。严格 Matched Control 子集上的新菌株与双重未知绝对 RMSE 仍是短板，因此不宣称所有子指标均已超过强基线。
