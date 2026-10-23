# C153：C152 固定学习率对照只读审计（CPU）

## 使用

1. 把 `C153_CPU_LAUNCHER.py` 全文粘贴到 Colab 单元格并运行。
2. 提示时只上传一份 `C153_CPU_LAUNCHER_BUNDLE.zip`；无需解压。启动器不内嵌科学源码，不要上传旧源码或科学结果包代替 BUNDLE。
3. 挂载保存原结果的 Drive；程序按固定路径和 SHA256 定位 **C152、C150、C149、C148、C147、C146、C145、C143 八个原包**。无需重新上传这八包，也无需 C151 结果 ZIP。
4. 路径变更时，可填写启动器顶部相应 `Cxxx_ZIP` 绝对路径；文件内容必须仍匹配已固定哈希。不自动选择“最新”的其它实验。
5. 只需 CPU 和 NumPy，不需要 Torch、HF、GPU、模型下载或训练。子进程屏蔽GPU，调用者的GPU环境变量保持不变。
6. 科学审计进程上限 **20分钟**，合成自测上限40分钟另计，每30秒有心跳。成功或运行中断均封装**一个结果总包**，包含日志、报告、逐态/逐步诊断及可信源码，并尝试备份到 Drive。

## 固定科学输入

C152 原包：

`/content/drive/MyDrive/GPT2_LC2_C152_FIXED_SMALLSTEP_COUPLING_CPU_V1_RECOVERY/c152_fixed_smallstep_coupling_1792728391669017839_results.zip`

SHA256：`81c0172b142ffbd9ac10b62e5e2aff826b86e039eaea39881e71748ec8e29a06`

其余七包路径与哈希见 `INPUT_PINS.json`。C152 R1 可信交付源码逐文件钉扎，原归档源码只检查字节，**不执行归档中的 Python**。

## 核对范围

- 八个原 ZIP 字节重哈希。随包的可信 C152 读取器重新执行 C151 的 C150 算术审计，并重绑实际初始耦合头、全部84态坐标/完整写前特征、路由权重、V、writer和control。C149作为既存封存审计凭据保留，不声称重跑其更早递归审计链。
- 独立核对 C152 两臂共同初始完整头、零 Adam 动量及216参数；固定学习率分别为5e-4与1e-4，每臂固定12步。
- 独立重算24次手工 Adam 更新、2016条训练记录的损失/梯度线性拉回/精确最大值聚合、336个封存提案，以及四臂两轮672条原生端点门槛。
- 复核全部336次原生前检、84次零头及168次初始头功能观测的归档一致性；不产生新模型观测。
- 逐态报告两臂全部12次写前训练观测、两轮最终原生端点、六个响应坐标的有符号值和裕度；保留全部84态，再单独列出C150与两臂失败态的并集。
- 报告相对冻结C150的新增失败、保留失败和获救，以及两臂的对角/交叉参数位移分解、响应/guard聚合夹角、归档梯度与实际更新的点积、已记录损失变化和盒约束投影。

C141等历史状态名中的 `confirmation` 不改变其当前身份：本轮全部84态均为**已暴露训练数据**，不计入新确认或排除输入。

## 主要输出

- `REPORT_ZH.md`：中文审计报告。
- `RECOMPUTED_NATIVE_SUMMARIES.json` 与 `RECOMPUTED_PAIRED_RESULTS.json`：所有原生门槛、分组结果和获救/破坏对照。
- `ALL84_TWO_RATE_RESPONSE_TRAJECTORIES.json`：全部84态的两臂响应轨迹。
- `FAILURE_UNION_TRAJECTORIES.json`：C150和两臂最终失败态并集的完整轨迹。
- `ALL24_UPDATE_DIAGNOSTICS.json`、`ALL2016_STATE_UPDATE_DIAGNOSTICS.json`、`ALL2016_RECORDED_TRANSITIONS.json`：完整更新和逐态算术诊断。
- `ALL168_BOTH_ARM_BLOCK_DIAGNOSTICS.json`、`ALL_RECORDED_RESPONSE_CROSSINGS.json`、`RATE_AND_FAILURE_DIAGNOSTIC_SUMMARY.json`：两臂参数块分解、已记录阈值跨越及汇总。

## 判定与边界

只有全部必要绑定、来源、算术、门槛、计数及原判定重算通过，审计才为 `complete=true`；任一不完整或冲突都保留 `complete=false` 并封装失败结果。审计通过不把C152的79/84改成84/84，不选择其它 checkpoint 或更优学习率，也不部署写入器。

归档梯度只作为有限数值数组检查，**不独立认证真实导数**。梯度/更新点积、夹角及跨步变化是描述性代数诊断，不证明优化器失败原因、系统误差、曲率界、收敛或表达能力不足。最后一条转移连接功能路径的step12写前观测与最终原生端点，包含潜在路径差异，不能当作独立优化效果。参数块分解不等于消融后的模型端点。

无模型加载、前向、反传、新训练、拟合、求解、新Jacobian/SVD、新确认态、排除输入评估或候选选择。`LC2_closed=false` 和 `LC2_single_fold_passed=false` 始终保留。归档一致性不能证明测量准确性、时序或未访问性。

## 交付验证

见 `LOCAL_VALIDATION.json` 与测试日志：本地采用完整84态合成C152归档验证读取、更新、封存、门槛和失败阻断，并检查实际worker默认审计入口及启动器封装。**本地未取得真实C152结果包，尚未在八个真实原包上完成C153验证**；Colab运行会首先校验八个固定输入，随后执行完整生产审计。合成测试不属于真实模型证据。
