# GPT-2 C7 A100 V1：功能门槛与匹配精修

## Colab 运行
选择 NVIDIA A100；上传 `GPT2_C7_FUNCTIONAL_GATE_A100_V1.py`，新单元运行：
```python
%run /content/GPT2_C7_FUNCTIONAL_GATE_A100_V1.py
```
也可完整粘贴 PY 到单元。输入已完成的 `gpt2_c6_staged_writer_results.zip`，或本版C7结果续跑。源码ZIP不是结果输入。

默认先寻找最高实际进度的C7备份，再在本地/已挂载Drive寻找指定C6结果。`INPUT_ZIP` 可填完整路径。默认Drive备份；`USE_GOOGLE_DRIVE=False` 可关闭。保留CUDA PyTorch；隔离固定HF依赖。无程序分钟限制，Colab仍可能回收。

## 已核验的C6父结果
- SHA-256：`98b1429a5a33e5dabf59cd23862773b6ab71f39b0fb7611df9fb3aeb9c6b5ebd`。
- 大小120,714,095字节；CRC与341项成员哈希通过。
- 四个真实warm头均为1200步，检查点SHA和进度一致。
- 原结论为 `C6_WARM_FIT_INCOMPLETE`，4800新增头训练步；未执行精修或排除输入评估。
- 三折训练端点9/9、另一折0/9，但所有折都没有满足C6预声明的MSE<=0.02联合门槛。

C7是观察C6后的新开发协议，**不改判C6通过**。C6完整原包和摘要在C7结果内保留。

## 本版流程
1. 复用C6真实1200步头、原C5训练池、教师、特征和训练归一化；不重训warm，不生成新的教师或扰动。
2. 实际重检12个准备/教师端点及缓存特征一致性。
3. 对每个头，先评估其全部9个训练状态。**实际训练端点9/9通过才具备精修资格**；MSE记录为诊断，不参与资格判断。四个头的资格全部固定后，才读取排除输入的结果。
4. 对四个冻结头都测量三项排除开发状态，包括不具备精修资格的头。这样冻结基线的原输入分母始终为4，不把失败折藏起来。
5. 具备资格的头进入两臂精修：MSE与功能损失，各200步，从同一C6头复制，优化器历史都重置。资格不通过的头仅保留基线，不强行训练精修。
6. 保存每个原始预测及直接响应校正后的实际门控；不使用教师替代失败预测，不选最好排除输入检查点。

## 不变设置
Pinned GPT-2、memory seed82931、固定LoRA-A、rank8、36,864个memory LoRA-B坐标、四地址访问规则均继承。

两臂全批9项MSE，每臂200步，AdamW lr0.0001、weight_decay0、梯度裁剪5。功能臂每步额外实际前向检查按固定顺序循环的一项训练状态。目标沿用C6预声明设置：
`MSE + 0.1*read + 0.01*response + 0.01*KL`。
`read = mean(softplus(1-target_relative_margin))`。
`response = mean(SmoothL1(mean_anchor_response_displacement/0.1,0))`。
`KL = relu(mean_KL/0.005-1)`。

0.1是训练损失尺度，**不是验收响应预算**。功能损失通过真实临时LoRA-B覆盖的模型前向传回头；基础参数和LoRA-A冻结。校正仍使用外部Jacobian/SVD，不是原生前向写入。

## 最终验收阈值
所有exact及开发读表达目标相对margin>=0.05；四个内容正确；所选输出真正翻转；前三个未选内容保留；平均锚响应漂移<=0.002；平均final-token KL<=0.005。

直接校正最多8步，每步最多6次预声明回溯，ridge1e-6。校正步按实测响应下降接受；最终仍须满足全部门控。没有核预投影或放宽阈值。平均锚点检查不代表逐锚点或全能力保证。

## 结果与分母
`SUMMARY.json` 分别给出：
- `frozen_baseline`：四个原C6头的训练36项和排除开发12项。
- `frozen_baseline_original_four_inputs`：四个原输入的冻结基线，分母4。
- `training_gates`：实际训练门控通过数和MSE诊断。
- `eligible_folds`：实际进入精修的折，不根据排除结果筛选。
- `paired_cohorts`：仅资格通过折的两臂配对比较。
- `paired_original_denominator`：精修比较原输入的实际分母；预计3，但以实际复检为准，不写成4/4。
- `paired_refinement_not_run_on_folds`：没有精修的折。

`C7_DEVELOPMENT_FUNCTIONAL_GATE_COMPARISON_COMPLETE` 表示协议结束，**不代表门控全过、外推可靠或C线完成**。当前四种XOR输入、教师、开发表达均已检查；没有全新留出或独立模型种子。C7不测试B5控制器，不证明参数更新内化。

## 保存与恢复
每25步原子保存实际头、AdamW状态、归一化、阶段与步数，进度指针绑定SHA。训练终点、各折基线检查、每4个精修评估及校正每2步封装结果并备份Drive。中断/子进程错误后尝试保存最后已提交检查点；未保存步会重算。

Drive仅保留 `GPT2_C7_FUNCTIONAL_GATE_A100_V1_RECOVERY` 最近3个已校验备份，C6及更早版本不修改。源码、设置、运行时身份和父包不匹配时拒绝猜测续跑；不靠旧日志恢复权重。

唯一结果包 `gpt2_c7_functional_gate_results.zip` 含完整C6父包，其父C5胶囊只能复核包含的成员，不能重建或重新完整核验未包含的原C5字节。

## 验证
本地27项协议、门控、归档、NumPy与封装检查通过；当前环境无Torch/CUDA。Colab运行前强制12项Torch梯度、损失尺度、匹配复制和优化器恢复测试，并重检实际GPT-2功能调用。A100实际结果仍未知。
