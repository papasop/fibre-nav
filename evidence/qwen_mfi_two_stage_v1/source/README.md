# Qwen self-read then compute · CPU v1

## 运行
上传启动器，Colab运行：

```python
%run run_qwen_self_read_compute_cpu_v1.py --threads 4
```

提示时上传 `qwen_self_read_compute_cpu_v1.zip`。ZIP自带旧adapter/快照，不需上传旧结果。
终端可把两个文件放同一工作目录，使用：

```bash
python run_qwen_self_read_compute_cpu_v1.py --threads 4
```

Linux x86_64 / Colab、Python3.11–3.13、CPU float32，固定Qwen3-0.6B revision与现有权重。
首次需联网下载基座，Linux x86_64安装CPU PyTorch2.8.0+cpu，Transformers4.56.2。
建议8GB可用内存。**没有总时长上限**，仍保留128新token/题上限，截断不算通过。
按你最近CPU实测速度，本轮先预计3–15分钟，安装下载另计，非本包完整实测保证。
每30秒心跳，逐题保存，最后自动下载结果ZIP。平台自身会话上限仍可能生效；本版不支持断点续算。

## 本轮问题与边界
之前“单独读取”能读对，“显式给值后计算”能算对，但“同一次提示中自行读取并计算”失败。
本轮测试两阶段能否接通：
1. 第一阶段，用已通过的固定两行读取提示，由模型生成两个值。
2. 第二阶段，把**第一阶段原始解码文本**加一个换行，再加固定protected规则正文，交给同一模型生成三行结果。

两次生成调用分别启动，上下文缓存不跨调用传递；模型参数与记忆快照相同。
外部程序负责搬运文本与评分，主流程不从参数程序化读出数值，不改模型读错的值，不替模型算答案。
评分当然需要外部真值计算，但真值只用于审核，不用于生成。
**第二阶段提示词确实含有值——来自模型第一阶段输出。** 因而这不是无正文数值的单次前向计算证据。

## 固定规模
读取器与规则adapter一直开启，全部参数冻结，仅装载已有zero/10/11快照。
第一阶段的3条联合读取直接使用本轮开始审核中实际生成的原文，XOR/相等共享同一条，不重复挑选答案。

30道第二阶段题 = 3快照 × 2任务 × 5条件：
- raw：原始模型生成文本，逐字传递，不重排/修复/规范化空格。
- flip0：控制组只翻转中间文本的第一位。
- flip1：控制组只翻转第二位。
- flip_both：控制组翻转两位。XOR/相等的结果应与原值相同，但报告值应变化。
- no_intermediate：不给中间文本，保持上一轮失败的直接计算提示。

后三种翻转是显式的**文本干预**，不是对参数地址写入。它们不能证明两个参数地址的独立因果作用。
控制组在原读取格式有效且未截断时才构造；格式无效则记skipped，不编造数值。raw条件始终原样传递，即使原读错或格式无效。

另有18次前后读取检查：3快照 ×（联合读 + 两个单地址读）× 前后两轮。
第一阶段3条已包含其中，正常最多48次生成，不是另加3条。
同一快照下五条件使用相同权重，不训练、不写新数据。保留响应/KL/规则floors/单地址margin审核。

## 判定
`complete=true` 是全计划处理完、冻结与前后重复性通过、原状态恢复；不等于全部正确，也不等于所有条件都实际生成，需查看generated_rows/skipped_rows。

`known_two_stage_gate=true` 需要：
- 完成与冻结审核通过；
- 已知10/11第一阶段确实读对；
- raw第二阶段4题都报告正确的两个值和正确结果；
- CPU有限响应、KL及原审核门槛通过。

`known_text_control_gate=true` 还要求：
- 8道已知状态单位置文本翻转题正确跟随干预文本；
- 4道双位置文本翻转题正确报告翻转值并计算。

各项都分别报告，不用主条件通过替代控制通过，也不根据结果改门槛。
`follows_intermediate`只表示第二阶段正确使用所提供的文本，不等于该文本是真实记忆。
`end_to_end_success`还要求原始读取吻合实际已知快照。zero没有被赋予存储状态，不计端到端真值成功；zero也能读出默认值并计算，不能据此当作有记忆。

**formal_l6_pass与single_pass_l6_pass固定false。** 即使本轮通过，正确结论也是“同一模型通过自生成文本完成已知状态的分步读取与计算”。
不能宣称单次隐式计算、两参数地址联合因果、未见组合泛化、语义机制定位或普遍推理提升。
两种参数状态仅10/11，第一位恒1；文本翻转测试不增加新的参数状态证据。

## 保留的审核
源文件SHA256；CPU固定原零记忆、无reader、rule-on参照，绝不围绕成功reader重设参照。
响应≤0.02、KL≤0.01、原规则floors、single margin≥1不变。
CPU KL相对CPU重算参照，不是原GPU证书。完整CPU log概率及GPU响应差保留。
每次生成检查参数版本、reader/记忆张量内容。结束复查读取与数值审核完全一致，退出恢复reader-on+zero。
无中间文本组与上一轮直接隐式6题逐条比较；GPU原读取对照也保留。

## 结果文件
- stage1_rows.json：第一阶段原文与生成记录。
- logic_rows.json：30条计划条件、实际prompt、中间值来源哈希、原文、控制标签、token、评分及耗时。
- summary.json：完成状态、两个分步门槛、各条件统计和文本干预响应。
- replay_rows.json：18条前后读取。
- audits.json、cpu_constraints.json、cpu_reference.json、cpu_ref_logp.safetensors。
- parent_direct_replay.json、gpu_read_comparison.json、protocol.json、runtime.json。

## 交付验证
本地16项测试：15通过，1项实际PyTorch序列化测试因未安装PyTorch跳过；启动器安装后重跑全部测试。
另完成语法、全局名称与ZIP完整性检查；未在交付环境运行完整Qwen CPU模型。
