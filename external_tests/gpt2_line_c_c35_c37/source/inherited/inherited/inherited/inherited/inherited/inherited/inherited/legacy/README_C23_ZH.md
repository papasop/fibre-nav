# C23：冻结起点的步长与精度诊断（A100）

本版使用 C22 完成后的真实 `heads/calibrated/checkpoint_000015.pt`，保持其最终写入器为唯一不可变起点。C22 的记录为训练 54/60、原始训练 32/36，16 个已检查新状态 8/16；本版不增加新状态，也不把这些记录升级为独立确认。

## 怎么运行

1. Colab 选择 NVIDIA A100，上传 `GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1.py`。
2. 保持已完成的 C22 **结果 ZIP** 在 Drive 的原 C22 恢复目录；本版自动寻找。也可修改 PY 顶部 `INPUT_ZIP` 为完整路径，或在提示时上传一个完整 C22 结果 ZIP。
3. 新单元运行：

```python
%run /content/GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1.py
```

源码 ZIP 用于审查和解压，不作为结果输入。自己的 C23 备份可续跑。Drive 已挂载时复用；如果授权失败，先单独 `drive.mount('/content/drive')`。依赖在隔离目录安装/复用，保留 Colab CUDA PyTorch。工作进程没有总时长上限，但 Colab 平台的会话限制仍可能中断。

## 本版检验什么

- 固定 C22 最终头 SHA-256：`3909cd5126c5995dba5ae170980260294f55c0295096ff2390f03ba2083c318c`。
- 复用 C22 round14 的 15 个同头 `xor_10` Jacobian、隐藏特征、残差和三组方向系数。运行时逐项核对头、参数状态、输入、隐藏特征和来源，不重新求导或求解。
- 3 个增量半径 `0.05/0.1/0.2` × 3 个 ridge `1e-6/1e-8/1e-10` × 10 个固定步长，合计 **90 个请求项**。步长为 `1, .5, .25, .125, .03125, .0078125, .001953125, .00048828125, .000244140625, .0001220703125`。
- 每项从同一真实头开始，不继承 C22 已缩小的 trust，不提前停止，不以线性预测淘汰候选。相同实际 float32 头按字节哈希复用测量，全部 90 项保留在分母；别名不是独立实验。
- 每个不同头实测路由全部 15 个状态的内容、flip、未选内容、读取裕度、响应和 KL。候选结束或异常时恢复原头。
- 路由合格候选按训练结果排序；每个不同头最多审计一次全 60 态，到第一个全 60 态通过的候选即停止审计。其余只是“路由合格”，不能算“全 60 态通过”。
- 扫描前后各 3 轮 × 15 态，合计 **90 个零位移重复前向**。
- 对同一次 float32 前向的 logits 和参考 logits，另用 float64 计算响应均值和差值；原 float32 验收不变。这只测试归约误差，不是整个模型 float64 前向，也不能消除 float32 前向或参数量化误差。

## 不变的验收与输出

主门仍为响应最大坐标漂移 `0.002`、平均 KL `0.005`、读取 margin `0.05`。保留 C22 原 guard：先前完整通过项和非响应门不得丢失，全 60 态及原始 36 态最大漂移不得增加；完整通过数增加或平方预算超额改善。

本版不训练、不新增反向/Jacobian/SVD/线性求解、不在线响应修正、不生成准备态、不测试控制器。不提交任何新写入器检查点；若有诊断候选，保存的 `best_candidate/CANDIDATE_LAYER.pt` 标明 `diagnostic_only_not_committed`，原头保持不变。

结果协议：`GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1`。
结果包：`gpt2_c23_frozen_step_diagnosis_results.zip`。
Drive：`GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1_RECOVERY`，只保留本目录最近 3 个已校验备份。C22 及旧版本存档不修改。每 5 个请求项备份，零位移控制和最终阶段也备份。

最终 decision 为 `C23_FULL60_ACCEPTABLE_CANDIDATE_FOUND` 或 `C23_NO_FULL60_ACCEPTABLE_CANDIDATE_FOUND`。它只回答该固定网格是否找到原 guard 可接受候选；不证明不可行，不建立 correction-free 新状态确认、LC1、native 更新或完整 Line C。

解压源码后可只读打印结果：

```bash
python print_summary_c23.py /完整路径/gpt2_c23_frozen_step_diagnosis_results.zip
```

请回传 SUMMARY、FIXED_ORIGIN_SCAN、FULL60_AUDITS 和零位移统计。无 SUMMARY 代表尚未完成，不能判失败。

## 可复现性边界

104 个 C22 来源成员按已交付源包逐字节固定，103 个继承文件不修改；新增文件有独立 SOURCE_MANIFEST。运行时完整检查结果 CRC、成员哈希、来源、嵌套训练胶囊和真实最终检查点。使用紧凑父胶囊，避免递归打包完整历史结果链。

本地已通过 50 项协议/归档检查、10 项方向/精度检查、19 项 Torch 隔离和实际扫描/审计块检查，共 79 项；另检查最终打包和单文件启动器。真实 GPT-2/A100 科学诊断尚未在本工作区运行。C22 结果数字来自用户提供的运行记录，不能把本地合成检查当作真实模型结果。
