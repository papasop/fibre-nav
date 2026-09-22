# Qwen 声明语法编译接口 CPU V8

V7日志显示96点完成，但fresh_a的00-OR及其重复均错误输出1。V7完整结果ZIP尚未收到并审计；本版保留这一证据边界。
V8把问题拆为：输入字段解析 → 冻结V6提示编译 → Qwen计算 → 受限输出。
目标是检验一个外部软件接口是否稳定，不声称模型自然语言泛化改善。

## Colab 使用

1. 将 `COLAB_LAUNCHER_QWEN_COMPILER_CPU_V8.py` 全文粘贴到Colab单元格运行。
2. 弹窗上传 `qwen_compiler_cpu_v8.zip`。首次不需要旧结果；V6基线已经内置。
3. CPU float32、4线程，GPU不参与；首次安装固定依赖并下载Qwen3-1.7B。建议至少12GB可用内存。
4. 每点原子提交；16/32/48/64/80/96/112/128点请求下载阶段ZIP，结束或暂停时下载主包和日志。
5. 默认不挂载Drive；可选USE_GOOGLE_DRIVE=True，挂载失败继续本地保存。

主包：`/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip`
日志：`/content/qwen_compiler_cpu_v8/QWEN_COMPILER_V8_SESSION_OUTPUT.txt`

```python
from pathlib import Path
from google.colab import files
p = Path('/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip')
if p.is_file():
    files.download(str(p))
else:
    print('结果尚未生成，请检查本次日志。')
```

请确认浏览器已下载文件；运行时销毁后，未下载、未备份的结果无法保证恢复。
启动器不需要venv或ensurepip；独立python -I -S进程避免notebook的-f参数干扰。

## 明确支持的输入范围

解析器只接受以下三种文本语法与JSON。文本大小写、标点和内部换行按下例；外部首尾空白可忽略。
文本可带一行本版已声明的运算规则前缀，也可不带；规则必须与运算匹配。不会接受任意自然语言或额外指令。

```text
A = 0
B = 1
Operation: XOR
Return the Boolean result as one digit (0 or 1).
```

```text
The input bits are A: 0; B: 1.
Apply XOR to these two bits.
Return the Boolean result as one digit (0 or 1).
```

```text
Evaluate XOR on this record:
B = 1
A = 0
Return the Boolean result as one digit (0 or 1).
```

```json
{"op":"XOR","a":0,"b":1}
```

文本运算标签仅OR、AND、XOR、A == B；JSON的op仅OR、AND、XOR、EQUAL。
JSON必须恰好有op/a/b三个字段，不允许重复字段；键顺序可变。a/b必须是整数0或1，布尔值、字符串和浮点数均拒绝。
不支持的格式返回拒绝，不猜测、不纠正、不回退到直接询问模型。

## 编译与计算边界

interface_compiler.py只抽取op/a/b并拼接通用规则及固定模板，不包含真值函数或答案表。
编译后的提示必须逐字等于对应V6 selected_rule提示，包括chat模板、Answer前缀。
expected_fields及真值仅供独立评估器核对，不传入运行时解析器或模型。
每次请求从source_text重新解析、编译、调用模型；没有读取案例标签修复解析，没有查表答案，没有模型结果缓存。

受限输出保持原方式：第一步仅允许0/1两个token、保留各自原logit，由模型选择；第二步由外部控制器强制EOS。
控制器保证格式，不保证答案。如果模型算错，评估器判失败，不替换答案。

## 冻结实验

- 四组：canonical、sentence、record、json。每组完整覆盖四种输入位组合×四种运算，共16题。
- 首轮64个有效输入，各自独立调用模型；随后完整重复64点，总计128模型检查点。
- **四种格式最终只对应16个不同的模型提示**。128次调用不是128个不同推理问题，也不是多seed证据。
- 24个负例在模型加载前检查拒绝：缺字段、重复字段、多个操作、非法位值、额外答案/指令、规则冲突等。
- 主运行只做受限生成，不再执行长篇自由生成。共128次受限生成、384次候选评分前向、256次额外数值前向。
- 每点保留输入原文、解析字段、编译内容与完整提示、模型输出token、候选分数、数值路径logit及有效配置。

FROZEN_QUERIES.json、FROZEN_NEGATIVES.json在运行前冻结；运行时使用实际tokenizer核验标准提示。
sentence/record组复用V7输入形式，已经根据失败结果选择了规范化路线，因此不是新的未知表达确认。

## 验收门槛

1. 64个支持输入及全部重复均解析为预声明字段，编译为对应冻结提示；24个负例全部拒绝。
2. 每组独立达到主分数16/16、受限答案16/16、运算对照14/14、输入对照12/12；主分数gap绝对值≤1e-5时弃权。
3. 同次掩码前后候选logit完全相同；生成的两个原始logit分别与独立末位置投影的cache开/关路径核对，逐值最大绝对差≤1e-5。
4. 原跨路径gap阈值1e-5继续作为legacy指标完整报告，不替代上述投影对齐审计。
5. canonical首轮重放V6的候选分数、受限输出、配置及数值路径记录，容差1e-5；本轮不重放历史自由生成。
6. 四种格式的对应模型结果保持一致；各组完整重复的输出/token/config一致，分数和logit差≤1e-5。
7. 源码、依赖、Python、CPU环境与冻结基线一致；模型参数与结构无变更。

输出overall_pass / compiled_interface_pass只表示“声明语法下的外部编译接口通过”。
native_free_pass为null、free_generation_evaluated=false，表示本轮未测自由输出。
unseen_expression_confirmation、formal_l6_pass、native_implicit_closure、deployment_ready始终false。
V6失败结论保持；V7只记录日志报告的失败及raw_archive_audited=false。

如果通过，支持的是规范化接口能消除已测试格式差异对模型输入的影响。解析与编译能力来自外部程序，不能计作Qwen新增的原生能力。
没有训练、Adapter、参数记忆写入、读写闭环或自动Hugging Face更新。

## 固定环境与耗时

Qwen/Qwen3-1.7B，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。
CPU FP32、4线程、SDPA、seed84917，关闭thinking，greedy。受限生成由EOS第二步结束。
目标torch2.8.0+cpu、transformers4.56.2，其余固定版本见requirements_cpu.txt。
历史基线记录Python3.13.15、Intel Xeon CPU @2.20GHz；环境字段不匹配会阻止通过。

取消长篇自由生成但调用数增加到128，CPU粗估25–60分钟，安装下载另计。这不是V8实测，慢机器可能更久。
默认90分钟会话预算（含加载），在检查点之间暂停；正在执行的点完成后才检查预算。

同一运行时重跑启动器并上传源ZIP，自动续用已提交记录。
新运行时上传同一源ZIP及一份最新V8主包或阶段包；旧版results不能混入。
源码、环境或已有记录冲突会停止而不覆盖；需要隔离目录时使用--workdir。

```bash
python COLAB_LAUNCHER_QWEN_COMPILER_CPU_V8.py --source qwen_compiler_cpu_v8.zip --minutes 90
```

## 工程检查及来源

```bash
cd qwen_compiler_cpu_v8
python -m unittest discover -p 'test_*.py' -v
```

启动器位于源目录父目录，或设置QWEN_COMPILER_V8_LAUNCHER为其绝对路径。
check_constraint_tiny.py用随机小Qwen3检查真实CPU生成、掩码和数值路径；不是预训练模型结果。
V6_PROVENANCE.json来自已审计的V6原始结果；V7_LOG_PROVENANCE.json明确只使用V7日志。
工程验证及目标环境差异见VALIDATION.md。哈希只用于一致性检查，不能单独证明执行真实性。
