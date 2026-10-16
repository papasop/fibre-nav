```
START C37 source/archive/prospective-split tests
.............
----------------------------------------------------------------------
Ran 13 tests in 0.110s

OK
END C37 source/archive/prospective-split tests exit= 0
START A100 device
A100_READY NVIDIA A100-SXM4-40GB 2.11.0+cu130
END A100 device exit= 0
HASHES_OK 1000 / 1632
明确输入： /content/drive/MyDrive/GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1_RECOVERY/gpt2_c36_frozen_local_replication_results_1792142124359043303.zip
C37：核对已完成C36归档、实际权重与32态记录；不训练、不重选头。
每类固定C36 new00..03准备态作中心；16个已检查中心，每个生成2个新扰动态，共32态。
中心曾检查但未用于写入器训练；不是未见输入类或独立模型测试。新种子固定，不重抽。
准备态允许外部Jacobian/SVD和纠正；在线写入每臂一个冻结组合提案，无求导/SVD/校正/回溯/教师。
预算0.002/0.005/0.05不变；无总会话时限；每4态备份，仅保留本C37最近3份已校验备份。
START 复用隔离HF依赖
DEPENDENCIES_OK
END 复用隔离HF依赖 exit= 0
START HF依赖检查
DEPENDENCIES_OK
END HF依赖检查 exit= 0
C37 合成自测有阶段/逐项日志，不设300秒限时。
START C37 archive/tensor/seal/resume tests
C37 SELFTEST SETUP 1/3: synthetic C35 archive
C36 SELFTEST SETUP 1/3: build synthetic C35 states; no GPT-2 loading
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 21 秒。
C36 SELFTEST SETUP 2/3: create and verify one synthetic archive
C36 SELFTEST SETUP 3/3: isolated per-test copies ready; elapsed 25.7s
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 47 秒。
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 67 秒。
C37 SELFTEST SETUP 2/3: verify synthetic C36 and copy frozen context
C37 SELFTEST SETUP 3/3: independent test copies ready
test_centre_is_prewrite_preparation_not_written_point (__main__.Tests.test_centre_is_prewrite_preparation_not_written_point) ... C37 SELFTEST CASE test_centre_is_prewrite_preparation_not_written_point
ok
test_collapsed_perturbation_is_invalid_without_resampling (__main__.Tests.test_collapsed_perturbation_is_invalid_without_resampling) ... C37 SELFTEST CASE test_collapsed_perturbation_is_invalid_without_resampling
ok
test_context_weight_tamper_rejected (__main__.Tests.test_context_weight_tamper_rejected) ... C37 SELFTEST CASE test_context_weight_tamper_rejected
ok
test_freeze_required_before_generation (__main__.Tests.test_freeze_required_before_generation) ... C37 SELFTEST CASE test_freeze_required_before_generation
ok
test_full32_changed_centres_and_completed_resume (__main__.Tests.test_full32_changed_centres_and_completed_resume) ... C37 SELFTEST CASE test_full32_changed_centres_and_completed_resume
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 104 秒。
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 124 秒。
ok
test_interrupt_resumes_same_sealed_candidate (__main__.Tests.test_interrupt_resumes_same_sealed_candidate) ... C37 SELFTEST CASE test_interrupt_resumes_same_sealed_candidate
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 144 秒。
ok
test_invalid_state_and_partial_branch_remain_visible (__main__.Tests.test_invalid_state_and_partial_branch_remain_visible) ... C37 SELFTEST CASE test_invalid_state_and_partial_branch_remain_visible
C37 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 179 秒。
ok
test_original_parent_tamper_rejected (__main__.Tests.test_original_parent_tamper_rejected) ... C37 SELFTEST CASE test_original_parent_tamper_rejected
ok

----------------------------------------------------------------------
Ran 8 tests in 192.116s

OK
END C37 archive/tensor/seal/resume tests exit= 0
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143362975001457.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
START C37 A100 fixed-centre transfer
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 20 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 40 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143404423457799.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C31 Loading pinned GPT-2 once; C25+C27 writer immutable; prepare-only geometry, raw write endpoints.
`torch_dtype` is deprecated! Use `dtype` instead!
/content/gpt2_c36_frozen_local_replication_a100_v1/deps/peft/tuners/lora/layer.py:2174: UserWarning: fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.
  warnings.warn(
C37 C36 PHYSICAL REPLAY xor_00__c36_new00 OK
C37 C36 PHYSICAL REPLAY xor_01__c36_new00 OK
C37 C36 PHYSICAL REPLAY xor_10__c36_new00 OK
C37 C36 PHYSICAL REPLAY xor_11__c36_new00 OK
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 78 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 98 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143455199755093.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_00__c37_new00 centre xor_00__c36_new00 distance 0.3750000007466571 valid True
C35 ENDPOINT new_local xor_00__c37_new00 R {'raw': 0.00667572021484375, 'geometry6': 7.62939453125e-05, 'selected6': 8.392333984375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 1 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_00__c37_new01 centre xor_00__c36_new00 distance 0.7499999990880185 valid True
C35 ENDPOINT new_local xor_00__c37_new01 R {'raw': 0.009124755859375, 'geometry6': 9.1552734375e-05, 'selected6': 0.00014495849609375} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 2 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_00__c37_new02 centre xor_00__c36_new01 distance 0.37500000034130504 valid True
C35 ENDPOINT new_local xor_00__c37_new02 R {'raw': 0.00260162353515625, 'geometry6': 5.340576171875e-05, 'selected6': 8.392333984375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 3 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_00__c37_new03 centre xor_00__c36_new01 distance 0.7500000008718793 valid True
C35 ENDPOINT new_local xor_00__c37_new03 R {'raw': 0.00849151611328125, 'geometry6': 7.62939453125e-05, 'selected6': 7.62939453125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 4 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 126 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 146 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143503685948737.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_00__c37_new04 centre xor_00__c36_new02 distance 0.37499999919206234 valid True
C35 ENDPOINT new_local xor_00__c37_new04 R {'raw': 0.0007171630859375, 'geometry6': 8.392333984375e-05, 'selected6': 9.1552734375e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 5 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_00__c37_new05 centre xor_00__c36_new02 distance 0.750000000485911 valid True
C35 ENDPOINT new_local xor_00__c37_new05 R {'raw': 0.005950927734375, 'geometry6': 0.0001373291015625, 'selected6': 8.392333984375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 6 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_00__c37_new06 centre xor_00__c36_new03 distance 0.3750000009794183 valid True
C35 ENDPOINT new_local xor_00__c37_new06 R {'raw': 0.007598876953125, 'geometry6': 4.57763671875e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 7 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_00__c37_new07 centre xor_00__c36_new03 distance 0.7499999979220063 valid True
C35 ENDPOINT new_local xor_00__c37_new07 R {'raw': 0.00726318359375, 'geometry6': 0.00016021728515625, 'selected6': 0.00017547607421875} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 8 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 179 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 199 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143557258560727.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_01__c37_new00 centre xor_01__c36_new00 distance 0.37499999956114016 valid True
C35 ENDPOINT new_local xor_01__c37_new00 R {'raw': 0.00388336181640625, 'geometry6': 6.103515625e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 9 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_01__c37_new01 centre xor_01__c36_new00 distance 0.7500000004353831 valid True
C35 ENDPOINT new_local xor_01__c37_new01 R {'raw': 0.00213623046875, 'geometry6': 8.392333984375e-05, 'selected6': 5.340576171875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 10 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_01__c37_new02 centre xor_01__c36_new01 distance 0.37500000100887154 valid True
C35 ENDPOINT new_local xor_01__c37_new02 R {'raw': 0.00180816650390625, 'geometry6': 0.00016021728515625, 'selected6': 0.0001983642578125} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 11 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_01__c37_new03 centre xor_01__c36_new01 distance 0.7500000010435717 valid True
C35 ENDPOINT new_local xor_01__c37_new03 R {'raw': 0.00433349609375, 'geometry6': 0.00012969970703125, 'selected6': 0.00016021728515625} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 12 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 234 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 254 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143612186945333.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_01__c37_new04 centre xor_01__c36_new02 distance 0.3750000008898139 valid True
C35 ENDPOINT new_local xor_01__c37_new04 R {'raw': 0.0032196044921875, 'geometry6': 3.0517578125e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 13 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_01__c37_new05 centre xor_01__c36_new02 distance 0.7499999995118247 valid True
C35 ENDPOINT new_local xor_01__c37_new05 R {'raw': 0.0014495849609375, 'geometry6': 6.103515625e-05, 'selected6': 4.57763671875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 14 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_01__c37_new06 centre xor_01__c36_new03 distance 0.37500000161848795 valid True
C35 ENDPOINT new_local xor_01__c37_new06 R {'raw': 0.002838134765625, 'geometry6': 2.288818359375e-05, 'selected6': 3.814697265625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 15 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_01__c37_new07 centre xor_01__c36_new03 distance 0.7499999981589185 valid True
C35 ENDPOINT new_local xor_01__c37_new07 R {'raw': 0.00199127197265625, 'geometry6': 0.00012969970703125, 'selected6': 9.918212890625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 16 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 290 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 310 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143668465389181.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_10__c37_new00 centre xor_10__c36_new00 distance 0.37500000739746897 valid True
C35 ENDPOINT new_local xor_10__c37_new00 R {'raw': 0.00399017333984375, 'geometry6': 3.0517578125e-05, 'selected6': 6.866455078125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 17 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_10__c37_new01 centre xor_10__c36_new00 distance 0.7500000009543613 valid True
C35 ENDPOINT new_local xor_10__c37_new01 R {'raw': 0.0062103271484375, 'geometry6': 0.000213623046875, 'selected6': 0.00016021728515625} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 18 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_10__c37_new02 centre xor_10__c36_new01 distance 0.37499999918955046 valid True
C35 ENDPOINT new_local xor_10__c37_new02 R {'raw': 0.001220703125, 'geometry6': 7.62939453125e-05, 'selected6': 6.103515625e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 19 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_10__c37_new03 centre xor_10__c36_new01 distance 0.7499999965578524 valid True
C35 ENDPOINT new_local xor_10__c37_new03 R {'raw': 0.00189971923828125, 'geometry6': 0.0001983642578125, 'selected6': 0.0001373291015625} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 20 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 346 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 366 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143724995089967.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_10__c37_new04 centre xor_10__c36_new02 distance 0.37500000016302754 valid True
C35 ENDPOINT new_local xor_10__c37_new04 R {'raw': 0.0021514892578125, 'geometry6': 3.0517578125e-05, 'selected6': 6.866455078125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 21 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_10__c37_new05 centre xor_10__c36_new02 distance 0.7499999998472825 valid True
C35 ENDPOINT new_local xor_10__c37_new05 R {'raw': 0.001220703125, 'geometry6': 6.103515625e-05, 'selected6': 5.340576171875e-05} pass {'raw': True, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 22 /32 {'raw': True, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_10__c37_new06 centre xor_10__c36_new03 distance 0.37500000055189153 valid True
C35 ENDPOINT new_local xor_10__c37_new06 R {'raw': 0.00251007080078125, 'geometry6': 6.103515625e-05, 'selected6': 4.57763671875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 23 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_10__c37_new07 centre xor_10__c36_new03 distance 0.750000000466922 valid True
C35 ENDPOINT new_local xor_10__c37_new07 R {'raw': 0.0032196044921875, 'geometry6': 0.000244140625, 'selected6': 0.0002899169921875} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 24 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 402 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 422 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143781542437395.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_11__c37_new00 centre xor_11__c36_new00 distance 0.3749999986963702 valid True
C35 ENDPOINT new_local xor_11__c37_new00 R {'raw': 0.00820159912109375, 'geometry6': 6.866455078125e-05, 'selected6': 4.57763671875e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 25 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_11__c37_new01 centre xor_11__c36_new00 distance 0.7500000004714589 valid True
C35 ENDPOINT new_local xor_11__c37_new01 R {'raw': 0.003753662109375, 'geometry6': 0.000152587890625, 'selected6': 0.0001373291015625} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 26 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_11__c37_new02 centre xor_11__c36_new01 distance 0.3749999999342501 valid True
C35 ENDPOINT new_local xor_11__c37_new02 R {'raw': 0.0096282958984375, 'geometry6': 5.340576171875e-05, 'selected6': 2.288818359375e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 27 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_11__c37_new03 centre xor_11__c36_new01 distance 0.7499999987451949 valid True
C35 ENDPOINT new_local xor_11__c37_new03 R {'raw': 0.0051116943359375, 'geometry6': 3.814697265625e-05, 'selected6': 3.814697265625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 28 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 460 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 480 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143839334309159.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 PREPARATION xor_11__c37_new04 centre xor_11__c36_new02 distance 0.3750000005864203 valid True
C35 ENDPOINT new_local xor_11__c37_new04 R {'raw': 0.00234222412109375, 'geometry6': 5.340576171875e-05, 'selected6': 6.103515625e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 29 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_11__c37_new05 centre xor_11__c36_new02 distance 0.7500000007563318 valid True
C35 ENDPOINT new_local xor_11__c37_new05 R {'raw': 0.00738525390625, 'geometry6': 7.62939453125e-05, 'selected6': 6.866455078125e-05} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 30 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_11__c37_new06 centre xor_11__c36_new03 distance 0.3749999999862259 valid True
C35 ENDPOINT new_local xor_11__c37_new06 R {'raw': 0.0034332275390625, 'geometry6': 0.0001068115234375, 'selected6': 0.0001068115234375} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 31 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 PREPARATION xor_11__c37_new07 centre xor_11__c36_new03 distance 0.7500000013373251 valid True
C35 ENDPOINT new_local xor_11__c37_new07 R {'raw': 0.0027618408203125, 'geometry6': 0.0001983642578125, 'selected6': 0.00018310546875} pass {'raw': False, 'geometry6': True, 'selected6': True}
C37 ENDPOINT 32 /32 {'raw': False, 'geometry6': True, 'selected6': True}
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 517 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 537 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143897450529343.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 571 秒。
C37 A100 fixed-centre transfer：子进程尚未退出，等待输出；已用 591 秒。
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792143952884659231.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
{
  "protocol": "GPT2_C37_CENTRE_TRANSFER_A100_V1",
  "complete": true,
  "decision": "C37_BOTH_FROZEN_WRITERS_CENTRE_TRANSFER_PASS",
  "new_local": {
    "raw": {
      "n_requested": 32,
      "valid_preparations": 32,
      "complete_passed": 7,
      "gate_counts": {
        "bits_match": 32,
        "output_flip": 32,
        "unselected_preserved": 32,
        "response": 7,
        "KL": 32,
        "exact_read_margin": 32,
        "development_read_margin": 32
      },
      "maximum_response_drift": 0.0096282958984375,
      "median_response_drift": 0.003326416015625,
      "clipped_compensations": 0
    },
    "geometry6": {
      "n_requested": 32,
      "valid_preparations": 32,
      "complete_passed": 32,
      "gate_counts": {
        "bits_match": 32,
        "output_flip": 32,
        "unselected_preserved": 32,
        "response": 32,
        "KL": 32,
        "exact_read_margin": 32,
        "development_read_margin": 32
      },
      "maximum_response_drift": 0.000244140625,
      "median_response_drift": 7.62939453125e-05,
      "clipped_compensations": 0
    },
    "selected6": {
      "n_requested": 32,
      "valid_preparations": 32,
      "complete_passed": 32,
      "gate_counts": {
        "bits_match": 32,
        "output_flip": 32,
        "unselected_preserved": 32,
        "response": 32,
        "KL": 32,
        "exact_read_margin": 32,
        "development_read_margin": 32
      },
      "maximum_response_drift": 0.0002899169921875,
      "median_response_drift": 7.2479248046875e-05,
      "clipped_compensations": 0
    }
  },
  "by_case": {
    "xor_00": {
      "raw": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 1,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 1,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.009124755859375,
        "median_response_drift": 0.006969451904296875,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.00016021728515625,
        "median_response_drift": 8.0108642578125e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.00017547607421875,
        "median_response_drift": 8.392333984375e-05,
        "clipped_compensations": 0
      }
    },
    "xor_01": {
      "raw": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 3,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 3,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.00433349609375,
        "median_response_drift": 0.0024871826171875,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.00016021728515625,
        "median_response_drift": 7.2479248046875e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.0001983642578125,
        "median_response_drift": 5.7220458984375e-05,
        "clipped_compensations": 0
      }
    },
    "xor_10": {
      "raw": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 3,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 3,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.0062103271484375,
        "median_response_drift": 0.002330780029296875,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.000244140625,
        "median_response_drift": 6.866455078125e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.0002899169921875,
        "median_response_drift": 6.866455078125e-05,
        "clipped_compensations": 0
      }
    },
    "xor_11": {
      "raw": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 0,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 0,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.0096282958984375,
        "median_response_drift": 0.00443267822265625,
        "clipped_compensations": 0
      },
      "geometry6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.0001983642578125,
        "median_response_drift": 7.2479248046875e-05,
        "clipped_compensations": 0
      },
      "selected6": {
        "n_requested": 8,
        "valid_preparations": 8,
        "complete_passed": 8,
        "gate_counts": {
          "bits_match": 8,
          "output_flip": 8,
          "unselected_preserved": 8,
          "response": 8,
          "KL": 8,
          "exact_read_margin": 8,
          "development_read_margin": 8
        },
        "maximum_response_drift": 0.00018310546875,
        "median_response_drift": 6.4849853515625e-05,
        "clipped_compensations": 0
      }
    }
  },
  "branch_complete_acceptance": {
    "geometry6": true,
    "selected6": true
  },
  "centres": [
    {
      "id": "xor_00__c36_new00",
      "case": "xor_00"
    },
    {
      "id": "xor_00__c36_new01",
      "case": "xor_00"
    },
    {
      "id": "xor_00__c36_new02",
      "case": "xor_00"
    },
    {
      "id": "xor_00__c36_new03",
      "case": "xor_00"
    },
    {
      "id": "xor_01__c36_new00",
      "case": "xor_01"
    },
    {
      "id": "xor_01__c36_new01",
      "case": "xor_01"
    },
    {
      "id": "xor_01__c36_new02",
      "case": "xor_01"
    },
    {
      "id": "xor_01__c36_new03",
      "case": "xor_01"
    },
    {
      "id": "xor_10__c36_new00",
      "case": "xor_10"
    },
    {
      "id": "xor_10__c36_new01",
      "case": "xor_10"
    },
    {
      "id": "xor_10__c36_new02",
      "case": "xor_10"
    },
    {
      "id": "xor_10__c36_new03",
      "case": "xor_10"
    },
    {
      "id": "xor_11__c36_new00",
      "case": "xor_11"
    },
    {
      "id": "xor_11__c36_new01",
      "case": "xor_11"
    },
    {
      "id": "xor_11__c36_new02",
      "case": "xor_11"
    },
    {
      "id": "xor_11__c36_new03",
      "case": "xor_11"
    }
  ],
  "centres_previously_inspected": true,
  "centres_selected_by_index_not_outcome": true,
  "all32_requested_states_retained": true,
  "invalid_preparations": 0,
  "new_training_steps": 0,
  "state_generation_retries": 0,
  "test_based_branch_or_checkpoint_selection": false,
  "online_write_Jacobians": 0,
  "online_write_SVD_calls": 0,
  "online_write_correction_steps": 0,
  "online_write_backtracking": 0,
  "postwrite_response_used_for_proposal": false,
  "teacher_fallback_used": false,
  "offline_state_preparation_uses_geometry": true,
  "centre_preparation_Jacobian_files": 16,
  "prepared_distances": {
    "xor_00__c37_new00": 0.3750000007466571,
    "xor_00__c37_new01": 0.7499999990880185,
    "xor_00__c37_new02": 0.37500000034130504,
    "xor_00__c37_new03": 0.7500000008718793,
    "xor_00__c37_new04": 0.37499999919206234,
    "xor_00__c37_new05": 0.750000000485911,
    "xor_00__c37_new06": 0.3750000009794183,
    "xor_00__c37_new07": 0.7499999979220063,
    "xor_01__c37_new00": 0.37499999956114016,
    "xor_01__c37_new01": 0.7500000004353831,
    "xor_01__c37_new02": 0.37500000100887154,
    "xor_01__c37_new03": 0.7500000010435717,
    "xor_01__c37_new04": 0.3750000008898139,
    "xor_01__c37_new05": 0.7499999995118247,
    "xor_01__c37_new06": 0.37500000161848795,
    "xor_01__c37_new07": 0.7499999981589185,
    "xor_10__c37_new00": 0.37500000739746897,
    "xor_10__c37_new01": 0.7500000009543613,
    "xor_10__c37_new02": 0.37499999918955046,
    "xor_10__c37_new03": 0.7499999965578524,
    "xor_10__c37_new04": 0.37500000016302754,
    "xor_10__c37_new05": 0.7499999998472825,
    "xor_10__c37_new06": 0.37500000055189153,
    "xor_10__c37_new07": 0.750000000466922,
    "xor_11__c37_new00": 0.3749999986963702,
    "xor_11__c37_new01": 0.7500000004714589,
    "xor_11__c37_new02": 0.3749999999342501,
    "xor_11__c37_new03": 0.7499999987451949,
    "xor_11__c37_new04": 0.3750000005864203,
    "xor_11__c37_new05": 0.7500000007563318,
    "xor_11__c37_new06": 0.3749999999862259,
    "xor_11__c37_new07": 0.7500000013373251
  },
  "C35_frozen_sha256": "d95a6d84f312261d355bf2d584c878eff2caeb2b7d19d38d92e0bc8f1c1faec8",
  "C37_frozen_sha256": "7afef2c48ba50c88590b85b337f32250999fd395ea7e4d894ee15e5531f836da",
  "deployed_writer_identity_sha256": "08986913bedc7c0656458b6e91b0998c052f67756bae14a40815d7d99eca0226",
  "same_model_chart_access_prompts": true,
  "requested_output_bit_supplied": true,
  "fresh_evaluation": false,
  "fresh_state_seeds": true,
  "independent_model_or_chart_replication": false,
  "internal_parameter_update": false,
  "controller_dispatch_tested": false,
  "LC1_same_model_centre_transfer_passed": true,
  "LC1_independently_confirmed": false,
  "full_Line_C_complete": false,
  "scope": "Development-only transfer of frozen composed writers to 32 fixed-seed perturbations around16 inspected C36 preparation centres, with no refitting or branch selection. Same model/chart and four trained input classes; external preparation geometry and runtime commitment remain."
}
END C37 A100 fixed-centre transfer exit= 0
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C37_CENTRE_TRANSFER_A100_V1_RECOVERY/gpt2_c37_centre_transfer_results_1792144008241680356.zip
RESULT_ARCHIVE=/content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip
唯一结果总包： /content/gpt2_c37_centre_transfer_a100_v1/session_1792143121287659850/gpt2_c37_centre_transfer_results.zip

```