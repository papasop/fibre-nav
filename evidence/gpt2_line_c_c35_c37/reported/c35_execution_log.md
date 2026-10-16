```
START C35 calibration/split/bootstrap self-tests
.DRIVE_BACKUP_OK /tmp/tmpgwnk3qzi/drive/gpt2_c35_prewrite_coefficient_results_1792137614413515655.zip
RESULT_ARCHIVE=/tmp/tmpgwnk3qzi/s/gpt2_c35_prewrite_coefficient_results.zip
DRIVE_BACKUP_OK /tmp/tmpgwnk3qzi/drive/gpt2_c35_prewrite_coefficient_results_1792137614415402707.zip
RESULT_ARCHIVE=/tmp/tmpgwnk3qzi/s/gpt2_c35_prewrite_coefficient_results.zip
DRIVE_BACKUP_OK /tmp/tmpgwnk3qzi/drive/gpt2_c35_prewrite_coefficient_results_1792137614417416160.zip
RESULT_ARCHIVE=/tmp/tmpgwnk3qzi/s/gpt2_c35_prewrite_coefficient_results.zip
DRIVE_BACKUP_OK /tmp/tmpgwnk3qzi/drive/gpt2_c35_prewrite_coefficient_results_1792137614419806292.zip
RESULT_ARCHIVE=/tmp/tmpgwnk3qzi/s/gpt2_c35_prewrite_coefficient_results.zip
.................
----------------------------------------------------------------------
Ran 18 tests in 0.126s

OK
END C35 calibration/split/bootstrap self-tests exit= 0
START A100 device
A100_READY NVIDIA A100-SXM4-40GB 2.11.0+cu130
END A100 device exit= 0
C35 compact actual C34 bases/writer/states imported; no old16 target used for fitting.
明确输入： /content/drive/MyDrive/GPT2_C34_LOW_DIMENSIONAL_CONTROL_A100_V1_RECOVERY/gpt2_c34_low_dimensional_control_results_1792135579783794370.zip
C35：冻结C34基；12态离线几何先验+36态校准；原写入/固定几何预测/学习系数三组，写前生成所有候选。
运行时不测中间写后响应，无Jacobian/SVD/校正/回溯/教师兜底；每臂一次组合提案+外部提交/前向验收。
离线几何与校准已编译为冻结仿射权重，不在线求导。冻结后生成新16态，种子/扰动幅度固定，不重抽。
旧16态仅开发诊断；新态仍同模型/参数图；准备态允许外部几何。不测试内部原生更新或B5控制器。
阈值0.002/0.005/0.05不变；无总会话时限；每路由/4态备份，仅清理本C35最近3份外已校验备份。
START 复用隔离HF依赖
DEPENDENCIES_OK
END 复用隔离HF依赖 exit= 0
START HF依赖检查
DEPENDENCIES_OK
END HF依赖检查 exit= 0
START C35 frozen-forward/no-leak/resume self-tests
C35 frozen-forward/no-leak/resume self-tests：子进程尚未退出，等待输出；已用 20 秒。
C35 frozen-forward/no-leak/resume self-tests：子进程尚未退出，等待输出；已用 40 秒。
C35 frozen-forward/no-leak/resume self-tests：子进程尚未退出，等待输出；已用 60 秒。
.........
----------------------------------------------------------------------
Ran 9 tests in 71.922s

OK
END C35 frozen-forward/no-leak/resume self-tests exit= 0
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137715388533355.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
START C35 A100 frozen prewrite coefficient experiment
C35 TRAIN-ONLY COEFFICIENT HEAD xor_00 {'kind': 'ridge', 'ridge': 1.0} fit_prediction_RMSE 1.952002270131705e-05
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137725824231480.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 TRAIN-ONLY COEFFICIENT HEAD xor_01 {'kind': 'ridge', 'ridge': 100.0} fit_prediction_RMSE 2.9994302214091837e-05
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137735968297532.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 TRAIN-ONLY COEFFICIENT HEAD xor_10 {'kind': 'ridge', 'ridge': 1.0} fit_prediction_RMSE 3.175916571295243e-05
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137747820923253.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 TRAIN-ONLY COEFFICIENT HEAD xor_11 {'kind': 'ridge', 'ridge': 1.0} fit_prediction_RMSE 2.622603942313697e-05
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137761272122241.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137775862242776.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C31 Loading pinned GPT-2 once; C25+C27 writer immutable; prepare-only geometry, raw write endpoints.
`torch_dtype` is deprecated! Use `dtype` instead!
/content/gpt2_c31_residual_coverage_a100_v1/deps/peft/tuners/lora/layer.py:2174: UserWarning: fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.
  warnings.warn(
C35 ENDPOINT training xor_00__c31_train00 R {'raw': 0.00214385986328125, 'geometry6': 7.62939453125e-06, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train01 R {'raw': 0.005035400390625, 'geometry6': 3.0517578125e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train02 R {'raw': 0.00128936767578125, 'geometry6': 3.0517578125e-05, 'selected6': 4.57763671875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train03 R {'raw': 0.00585174560546875, 'geometry6': 0.0001220703125, 'selected6': 3.814697265625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137799497928285.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_00__c31_train04 R {'raw': 0.0019989013671875, 'geometry6': 4.57763671875e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train05 R {'raw': 0.01099395751953125, 'geometry6': 9.1552734375e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train06 R {'raw': 0.0020599365234375, 'geometry6': 6.103515625e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train07 R {'raw': 0.00348663330078125, 'geometry6': 2.288818359375e-05, 'selected6': 4.57763671875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137816841925482.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_00__c31_train08 R {'raw': 0.001068115234375, 'geometry6': 2.288818359375e-05, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train09 R {'raw': 0.00274658203125, 'geometry6': 3.0517578125e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train10 R {'raw': 0.0029754638671875, 'geometry6': 3.814697265625e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_00__c31_train11 R {'raw': 0.00266265869140625, 'geometry6': 0.00014495849609375, 'selected6': 3.814697265625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137834291008424.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_01__c31_train00 R {'raw': 0.00351715087890625, 'geometry6': 2.288818359375e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train01 R {'raw': 0.005126953125, 'geometry6': 3.0517578125e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train02 R {'raw': 0.00115203857421875, 'geometry6': 1.52587890625e-05, 'selected6': 4.57763671875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train03 R {'raw': 0.0018463134765625, 'geometry6': 0.00011444091796875, 'selected6': 6.866455078125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137851820510048.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_01__c31_train04 R {'raw': 0.0013580322265625, 'geometry6': 3.0517578125e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train05 R {'raw': 0.00213623046875, 'geometry6': 4.57763671875e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train06 R {'raw': 0.00138092041015625, 'geometry6': 4.57763671875e-05, 'selected6': 2.288818359375e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train07 R {'raw': 0.0018310546875, 'geometry6': 6.103515625e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137869574793682.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_01__c31_train08 R {'raw': 0.00118255615234375, 'geometry6': 3.814697265625e-05, 'selected6': 2.288818359375e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train09 R {'raw': 0.00261688232421875, 'geometry6': 6.103515625e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train10 R {'raw': 0.002838134765625, 'geometry6': 1.52587890625e-05, 'selected6': 4.57763671875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_01__c31_train11 R {'raw': 0.005645751953125, 'geometry6': 1.52587890625e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137887482428120.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_10__c31_train00 R {'raw': 0.00289154052734375, 'geometry6': 3.814697265625e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train01 R {'raw': 0.0017852783203125, 'geometry6': 2.288818359375e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train02 R {'raw': 0.0064697265625, 'geometry6': 2.288818359375e-05, 'selected6': 7.62939453125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train03 R {'raw': 0.00646209716796875, 'geometry6': 7.62939453125e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137905688809004.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_10__c31_train04 R {'raw': 0.001373291015625, 'geometry6': 7.62939453125e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train05 R {'raw': 0.0015869140625, 'geometry6': 0.000244140625, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train06 R {'raw': 0.0005340576171875, 'geometry6': 2.288818359375e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train07 R {'raw': 0.0016632080078125, 'geometry6': 3.0517578125e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137924047536852.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_10__c31_train08 R {'raw': 0.0031585693359375, 'geometry6': 6.103515625e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train09 R {'raw': 0.0014190673828125, 'geometry6': 7.62939453125e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train10 R {'raw': 0.00112152099609375, 'geometry6': 6.866455078125e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_10__c31_train11 R {'raw': 0.0046844482421875, 'geometry6': 7.62939453125e-05, 'selected6': 5.340576171875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137942758137728.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_11__c31_train00 R {'raw': 0.001922607421875, 'geometry6': 4.57763671875e-05, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train01 R {'raw': 0.0012054443359375, 'geometry6': 6.103515625e-05, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train02 R {'raw': 0.00240325927734375, 'geometry6': 7.62939453125e-06, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train03 R {'raw': 0.00301361083984375, 'geometry6': 2.288818359375e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137961439578403.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_11__c31_train04 R {'raw': 0.00040435791015625, 'geometry6': 2.288818359375e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train05 R {'raw': 0.00202178955078125, 'geometry6': 2.288818359375e-05, 'selected6': 3.814697265625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train06 R {'raw': 0.00180816650390625, 'geometry6': 4.57763671875e-05, 'selected6': 4.57763671875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train07 R {'raw': 0.00696563720703125, 'geometry6': 6.103515625e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137980371293673.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT training xor_11__c31_train08 R {'raw': 0.00048828125, 'geometry6': 8.392333984375e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train09 R {'raw': 0.0032196044921875, 'geometry6': 3.0517578125e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train10 R {'raw': 0.0016326904296875, 'geometry6': 2.288818359375e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT training xor_11__c31_train11 R {'raw': 0.000701904296875, 'geometry6': 9.918212890625e-05, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792137999697146118.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT inspected_development xor_00__c31_new00 R {'raw': 0.002288818359375, 'geometry6': 4.57763671875e-05, 'selected6': 4.57763671875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_00__c31_new01 R {'raw': 0.00354766845703125, 'geometry6': 0.000152587890625, 'selected6': 8.392333984375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_00__c31_new02 R {'raw': 0.001251220703125, 'geometry6': 2.288818359375e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_00__c31_new03 R {'raw': 0.00732421875, 'geometry6': 4.57763671875e-05, 'selected6': 4.57763671875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138019030506849.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT inspected_development xor_01__c31_new00 R {'raw': 0.0010528564453125, 'geometry6': 3.814697265625e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_01__c31_new01 R {'raw': 0.00322723388671875, 'geometry6': 4.57763671875e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_01__c31_new02 R {'raw': 0.002532958984375, 'geometry6': 3.0517578125e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_01__c31_new03 R {'raw': 0.00167083740234375, 'geometry6': 9.918212890625e-05, 'selected6': 6.866455078125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138038621842316.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT inspected_development xor_10__c31_new00 R {'raw': 0.00234222412109375, 'geometry6': 3.814697265625e-05, 'selected6': 3.814697265625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_10__c31_new01 R {'raw': 0.00112152099609375, 'geometry6': 4.57763671875e-05, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_10__c31_new02 R {'raw': 0.0008087158203125, 'geometry6': 8.392333984375e-05, 'selected6': 6.866455078125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_10__c31_new03 R {'raw': 0.00045013427734375, 'geometry6': 7.62939453125e-05, 'selected6': 0.0001220703125} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138058206621012.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 ENDPOINT inspected_development xor_11__c31_new00 R {'raw': 0.00140380859375, 'geometry6': 2.288818359375e-05, 'selected6': 2.288818359375e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_11__c31_new01 R {'raw': 0.00150299072265625, 'geometry6': 6.866455078125e-05, 'selected6': 3.814697265625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_11__c31_new02 R {'raw': 0.0017852783203125, 'geometry6': 3.814697265625e-05, 'selected6': 4.57763671875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 ENDPOINT inspected_development xor_11__c31_new03 R {'raw': 0.00189971923828125, 'geometry6': 3.0517578125e-05, 'selected6': 5.340576171875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138077770436354.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 NEW PREPARATION xor_00__c35_new00 valid True
C35 ENDPOINT new_local xor_00__c35_new00 R {'raw': 0.001007080078125, 'geometry6': 3.0517578125e-05, 'selected6': 5.340576171875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_00__c35_new01 valid True
C35 ENDPOINT new_local xor_00__c35_new01 R {'raw': 0.00177001953125, 'geometry6': 3.814697265625e-05, 'selected6': 7.62939453125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_00__c35_new02 valid True
C35 ENDPOINT new_local xor_00__c35_new02 R {'raw': 0.0037994384765625, 'geometry6': 3.0517578125e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_00__c35_new03 valid True
C35 ENDPOINT new_local xor_00__c35_new03 R {'raw': 0.0024261474609375, 'geometry6': 3.814697265625e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138099995124067.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 NEW PREPARATION xor_01__c35_new00 valid True
C35 ENDPOINT new_local xor_01__c35_new00 R {'raw': 0.00323486328125, 'geometry6': 3.0517578125e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_01__c35_new01 valid True
C35 ENDPOINT new_local xor_01__c35_new01 R {'raw': 0.0024261474609375, 'geometry6': 0.0001220703125, 'selected6': 0.00017547607421875} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_01__c35_new02 valid True
C35 ENDPOINT new_local xor_01__c35_new02 R {'raw': 0.00130462646484375, 'geometry6': 6.866455078125e-05, 'selected6': 3.0517578125e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_01__c35_new03 valid True
C35 ENDPOINT new_local xor_01__c35_new03 R {'raw': 0.00736236572265625, 'geometry6': 6.103515625e-05, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138122775028968.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 NEW PREPARATION xor_10__c35_new00 valid True
C35 ENDPOINT new_local xor_10__c35_new00 R {'raw': 0.00669097900390625, 'geometry6': 9.1552734375e-05, 'selected6': 6.866455078125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_10__c35_new01 valid True
C35 ENDPOINT new_local xor_10__c35_new01 R {'raw': 0.00331878662109375, 'geometry6': 8.392333984375e-05, 'selected6': 0.00011444091796875} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_10__c35_new02 valid True
C35 ENDPOINT new_local xor_10__c35_new02 R {'raw': 0.00363922119140625, 'geometry6': 2.288818359375e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_10__c35_new03 valid True
C35 ENDPOINT new_local xor_10__c35_new03 R {'raw': 0.00676727294921875, 'geometry6': 0.0001220703125, 'selected6': 3.0517578125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 A100 frozen prewrite coefficient experiment：子进程尚未退出，等待输出；已用 437 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138146456569221.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 NEW PREPARATION xor_11__c35_new00 valid True
C35 ENDPOINT new_local xor_11__c35_new00 R {'raw': 0.002899169921875, 'geometry6': 7.62939453125e-05, 'selected6': 7.62939453125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_11__c35_new01 valid True
C35 ENDPOINT new_local xor_11__c35_new01 R {'raw': 0.0046844482421875, 'geometry6': 5.340576171875e-05, 'selected6': 5.340576171875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_11__c35_new02 valid True
C35 ENDPOINT new_local xor_11__c35_new02 R {'raw': 0.00215911865234375, 'geometry6': 3.0517578125e-05, 'selected6': 7.62939453125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 NEW PREPARATION xor_11__c35_new03 valid True
C35 ENDPOINT new_local xor_11__c35_new03 R {'raw': 0.0031890869140625, 'geometry6': 5.340576171875e-05, 'selected6': 5.340576171875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C35 A100 frozen prewrite coefficient experiment：子进程尚未退出，等待输出；已用 461 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138170899781985.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
C35 A100 frozen prewrite coefficient experiment：子进程尚未退出，等待输出；已用 482 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138192027687604.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
{
  "protocol": "GPT2_C35_PREWRITE_COEFFICIENT_A100_V1",
  "complete": true,
  "decision": "C35_FROZEN_SINGLE_PROPOSAL_LOCAL_DEVELOPMENT_PASS",
  "cohorts": {
    "training": {
      "raw": {
        "n_requested": 48,
        "valid_preparations": 48,
        "complete_passed": 23,
        "gate_counts": {
          "bits_match": 48,
          "output_flip": 48,
          "unselected_preserved": 48,
          "response": 23,
          "KL": 48,
          "exact_read_margin": 48,
          "development_read_margin": 48
        },
        "maximum_response_drift": 0.01099395751953125,
        "median_response_drift": 0.002040863037109375,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 48,
        "valid_preparations": 48,
        "complete_passed": 48,
        "gate_counts": {
          "bits_match": 48,
          "output_flip": 48,
          "unselected_preserved": 48,
          "response": 48,
          "KL": 48,
          "exact_read_margin": 48,
          "development_read_margin": 48
        },
        "maximum_response_drift": 0.000244140625,
        "median_response_drift": 3.814697265625e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 48,
        "valid_preparations": 48,
        "complete_passed": 48,
        "gate_counts": {
          "bits_match": 48,
          "output_flip": 48,
          "unselected_preserved": 48,
          "response": 48,
          "KL": 48,
          "exact_read_margin": 48,
          "development_read_margin": 48
        },
        "maximum_response_drift": 7.62939453125e-05,
        "median_response_drift": 3.814697265625e-05,
        "clipped_compensations": 0
      }
    },
    "inspected_development": {
      "raw": {
        "n_requested": 16,
        "valid_preparations": 16,
        "complete_passed": 10,
        "gate_counts": {
          "bits_match": 16,
          "output_flip": 16,
          "unselected_preserved": 16,
          "response": 10,
          "KL": 16,
          "exact_read_margin": 16,
          "development_read_margin": 16
        },
        "maximum_response_drift": 0.00732421875,
        "median_response_drift": 0.001728057861328125,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 16,
        "valid_preparations": 16,
        "complete_passed": 16,
        "gate_counts": {
          "bits_match": 16,
          "output_flip": 16,
          "unselected_preserved": 16,
          "response": 16,
          "KL": 16,
          "exact_read_margin": 16,
          "development_read_margin": 16
        },
        "maximum_response_drift": 0.000152587890625,
        "median_response_drift": 4.57763671875e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 16,
        "valid_preparations": 16,
        "complete_passed": 16,
        "gate_counts": {
          "bits_match": 16,
          "output_flip": 16,
          "unselected_preserved": 16,
          "response": 16,
          "KL": 16,
          "exact_read_margin": 16,
          "development_read_margin": 16
        },
        "maximum_response_drift": 0.0001220703125,
        "median_response_drift": 4.57763671875e-05,
        "clipped_compensations": 0
      }
    },
    "new_local": {
      "raw": {
        "n_requested": 16,
        "valid_preparations": 16,
        "complete_passed": 3,
        "gate_counts": {
          "bits_match": 16,
          "output_flip": 16,
          "unselected_preserved": 16,
          "response": 3,
          "KL": 16,
          "exact_read_margin": 16,
          "development_read_margin": 16
        },
        "maximum_response_drift": 0.00736236572265625,
        "median_response_drift": 0.00321197509765625,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 16,
        "valid_preparations": 16,
        "complete_passed": 16,
        "gate_counts": {
          "bits_match": 16,
          "output_flip": 16,
          "unselected_preserved": 16,
          "response": 16,
          "KL": 16,
          "exact_read_margin": 16,
          "development_read_margin": 16
        },
        "maximum_response_drift": 0.0001220703125,
        "median_response_drift": 5.340576171875e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 16,
        "valid_preparations": 16,
        "complete_passed": 16,
        "gate_counts": {
          "bits_match": 16,
          "output_flip": 16,
          "unselected_preserved": 16,
          "response": 16,
          "KL": 16,
          "exact_read_margin": 16,
          "development_read_margin": 16
        },
        "maximum_response_drift": 0.00017547607421875,
        "median_response_drift": 5.340576171875e-05,
        "clipped_compensations": 0
      }
    }
  },
  "new_local_by_case": {
    "xor_00": {
      "raw": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 2,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 2,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.0037994384765625,
        "median_response_drift": 0.00209808349609375,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 3.814697265625e-05,
        "median_response_drift": 3.4332275390625e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 7.62939453125e-05,
        "median_response_drift": 4.1961669921875e-05,
        "clipped_compensations": 0
      }
    },
    "xor_01": {
      "raw": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 1,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 1,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.00736236572265625,
        "median_response_drift": 0.00283050537109375,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.0001220703125,
        "median_response_drift": 6.4849853515625e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.00017547607421875,
        "median_response_drift": 3.0517578125e-05,
        "clipped_compensations": 0
      }
    },
    "xor_10": {
      "raw": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 0,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 0,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.00676727294921875,
        "median_response_drift": 0.00516510009765625,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.0001220703125,
        "median_response_drift": 8.7738037109375e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.00011444091796875,
        "median_response_drift": 6.4849853515625e-05,
        "clipped_compensations": 0
      }
    },
    "xor_11": {
      "raw": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 0,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 0,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 0.0046844482421875,
        "median_response_drift": 0.00304412841796875,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 7.62939453125e-05,
        "median_response_drift": 5.340576171875e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 4,
        "valid_preparations": 4,
        "complete_passed": 4,
        "gate_counts": {
          "bits_match": 4,
          "output_flip": 4,
          "unselected_preserved": 4,
          "response": 4,
          "KL": 4,
          "exact_read_margin": 4,
          "development_read_margin": 4
        },
        "maximum_response_drift": 7.62939453125e-05,
        "median_response_drift": 6.4849853515625e-05,
        "clipped_compensations": 0
      }
    }
  },
  "selected_heads": {
    "xor_00": {
      "kind": "ridge",
      "ridge": 1.0
    },
    "xor_01": {
      "kind": "ridge",
      "ridge": 100.0
    },
    "xor_10": {
      "kind": "ridge",
      "ridge": 1.0
    },
    "xor_11": {
      "kind": "ridge",
      "ridge": 1.0
    }
  },
  "calibration_training_n": 36,
  "fixed_geometry_prior_n": 12,
  "coefficients_frozen_sha256": "d95a6d84f312261d355bf2d584c878eff2caeb2b7d19d38d92e0bc8f1c1faec8",
  "all80_requested_states_retained": true,
  "new_state_generation_retries": 0,
  "old16_used_for_training_or_selection": false,
  "all_heads_frozen_before_new_state_generation": true,
  "candidate_points_sealed_before_all_endpoint_audits": true,
  "online_write_Jacobians": 0,
  "online_write_SVD_calls": 0,
  "online_write_correction_steps": 0,
  "online_write_backtracking": 0,
  "postwrite_response_used_for_proposal": false,
  "teacher_fallback_used": false,
  "offline_training_uses_archived_response_targets_and_Jacobians": true,
  "offline_new_state_preparation_uses_geometry": true,
  "one_frozen_core_forward_per_state_and_fixed_affine_head_operations": true,
  "preparation_Jacobian_files": 12,
  "preparation_correction_Jacobian_files": 0,
  "original_C25_C27_writer_unchanged": true,
  "fresh_state_seeds": true,
  "fresh_evaluation": false,
  "same_model_chart_access_prompts": true,
  "requested_output_bit_supplied": true,
  "LC1_local_development_protocol_passed": true,
  "LC1_independently_confirmed": false,
  "full_Line_C_complete": false,
  "internal_parameter_update": false,
  "controller_dispatch_tested": false,
  "scope": "Same-model/chart development of a frozen composed writer using prewrite geometric features and learned affine residual calibration. C34 bases fixed; final proposal committed once per arm. New16 seeded states generated after freeze, no resampling or test checkpoint selection. External runtime and offline geometry remain.",
  "interpretation": "All three frozen proposals are produced from prewrite data before any written-endpoint audit. Fixed offline Jacobians are features, not rebuilt online. This is externally committed same-model/chart development; native updates, independent model replication and broad generalisation are not established."
}
END C35 A100 frozen prewrite coefficient experiment exit= 0
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY/gpt2_c35_prewrite_coefficient_results_1792138215085983986.zip
RESULT_ARCHIVE=/content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip
唯一结果总包： /content/gpt2_c35_prewrite_coefficient_a100_v1/session_1792137616903923884/gpt2_c35_prewrite_coefficient_results.zip

```