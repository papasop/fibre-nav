```
START C36 source/archive/prospective-split tests
.................
----------------------------------------------------------------------
Ran 17 tests in 0.113s

OK
END C36 source/archive/prospective-split tests exit= 0
START A100 device
A100_READY NVIDIA A100-SXM4-40GB 2.11.0+cu130
END A100 device exit= 0
Mounted at /content/drive
恢复C36真实冻结权重、已封存提案与验收记录： (1, 1, 1, 8, 32)
明确输入： /content/drive/MyDrive/GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1_RECOVERY/gpt2_c36_frozen_local_replication_results_1792141289339567703.zip
C36：不训练、不重选头；核对C35全部80态存档、复现80个写前提案、实际重放8态端点。
固定几何/学习校准平行报告；冻结后按固定种子生成新32态，每类8态；不重抽、不删失败态。
每臂写前一次组合提案并封存，外部提交/前向验收；在线无Jacobian/SVD/校正/回溯/教师。
离线几何、准备态计算、外部运行时仍存在；同模型/图的额外局部复验，不是独立模型确认或原生更新。
阈值0.002/0.005/0.05保持；无总会话时限；每4新态备份，仅保留本C36最近3份已校验备份。
START HF依赖检查
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)
                                                                                                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
END HF依赖检查 exit= 1
START 隔离HF依赖；保留CUDA PyTorch
Collecting transformers==4.56.2
  Downloading transformers-4.56.2-py3-none-any.whl.metadata (40 kB)
Collecting peft==0.17.1
  Downloading peft-0.17.1-py3-none-any.whl.metadata (14 kB)
Collecting tokenizers==0.22.0
  Downloading tokenizers-0.22.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting huggingface_hub==0.34.4
  Downloading huggingface_hub-0.34.4-py3-none-any.whl.metadata (14 kB)
Collecting accelerate==1.10.1
  Downloading accelerate-1.10.1-py3-none-any.whl.metadata (19 kB)
Collecting safetensors==0.6.2
  Downloading safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Downloading transformers-4.56.2-py3-none-any.whl (11.6 MB)
Downloading peft-0.17.1-py3-none-any.whl (504 kB)
Downloading tokenizers-0.22.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Downloading huggingface_hub-0.34.4-py3-none-any.whl (561 kB)
Downloading accelerate-1.10.1-py3-none-any.whl (374 kB)
Downloading safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (485 kB)
Installing collected packages: transformers, tokenizers, safetensors, peft, huggingface_hub, accelerate
Successfully installed accelerate-1.10.1 huggingface_hub-0.34.4 peft-0.17.1 safetensors-0.6.2 tokenizers-0.22.0 transformers-4.56.2
END 隔离HF依赖；保留CUDA PyTorch exit= 0
START 安装后依赖检查
DEPENDENCIES_OK
END 安装后依赖检查 exit= 0
C36 FIX1：合成流水线自测显示分阶段进度，不设300秒超时；实际模型权重、种子和验收规则不变。
START C36 archive/tensor/seal/resume tests
C36 SELFTEST SETUP 1/3: build synthetic C35 states; no GPT-2 loading
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 21 秒。
C36 SELFTEST SETUP 2/3: create and verify one synthetic archive
C36 SELFTEST SETUP 3/3: isolated per-test copies ready; elapsed 25.7s
test_actual_core_identity_checker_rejects_synthetic (__main__.Tests.test_actual_core_identity_checker_rejects_synthetic) ... C36 SELFTEST CASE test_actual_core_identity_checker_rejects_synthetic
ok
test_full_pipeline_32_and_completed_resume (__main__.Tests.test_full_pipeline_32_and_completed_resume) ... C36 SELFTEST CASE test_full_pipeline_32_and_completed_resume
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 47 秒。
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 67 秒。
ok
test_interrupted_endpoint_resumes_same_frozen_proposals (__main__.Tests.test_interrupted_endpoint_resumes_same_frozen_proposals) ... C36 SELFTEST CASE test_interrupted_endpoint_resumes_same_frozen_proposals
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 88 秒。
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 108 秒。
ok
test_invalid_state_retained_and_branch_fails (__main__.Tests.test_invalid_state_retained_and_branch_fails) ... C36 SELFTEST CASE test_invalid_state_retained_and_branch_fails
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 130 秒。
ok
test_no_freeze_after_new_generation (__main__.Tests.test_no_freeze_after_new_generation) ... C36 SELFTEST CASE test_no_freeze_after_new_generation
ok
test_nonfinite_tensor_rejected (__main__.Tests.test_nonfinite_tensor_rejected) ... C36 SELFTEST CASE test_nonfinite_tensor_rejected
ok
test_parent_numeric_false_gate_rejected (__main__.Tests.test_parent_numeric_false_gate_rejected) ... C36 SELFTEST CASE test_parent_numeric_false_gate_rejected
ok
test_parent_tensor_corruption_rejected (__main__.Tests.test_parent_tensor_corruption_rejected) ... C36 SELFTEST CASE test_parent_tensor_corruption_rejected
ok
test_partial_branch_results_not_hidden (__main__.Tests.test_partial_branch_results_not_hidden) ... C36 SELFTEST CASE test_partial_branch_results_not_hidden
C36 archive/tensor/seal/resume tests：子进程尚未退出，等待输出；已用 171 秒。
ok
test_proposal_point_tamper_caught_even_with_updated_local_record (__main__.Tests.test_proposal_point_tamper_caught_even_with_updated_local_record) ... C36 SELFTEST CASE test_proposal_point_tamper_caught_even_with_updated_local_record
ok
test_replay_gate_failure_never_hidden_by_tolerance (__main__.Tests.test_replay_gate_failure_never_hidden_by_tolerance) ... C36 SELFTEST CASE test_replay_gate_failure_never_hidden_by_tolerance
ok

----------------------------------------------------------------------
Ran 11 tests in 191.027s

OK
END C36 archive/tensor/seal/resume tests exit= 0
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1_RECOVERY/gpt2_c36_frozen_local_replication_results_1792142090378055561.zip
RESULT_ARCHIVE=/content/gpt2_c36_frozen_local_replication_a100_v1/session_1792141839856215886/gpt2_c36_frozen_local_replication_results.zip
START C36 A100 frozen local replication
{
  "protocol": "GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1",
  "complete": true,
  "decision": "C36_BOTH_FROZEN_WRITERS_LOCAL_REPLICATION_PASS",
  "parent_archival_states_audited": 80,
  "parent_proposals_exactly_reproduced": 80,
  "parent_physical_replay_states": 8,
  "new_local": {
    "raw": {
      "n_requested": 32,
      "valid_preparations": 32,
      "complete_passed": 5,
      "gate_counts": {
        "bits_match": 32,
        "output_flip": 32,
        "unselected_preserved": 32,
        "response": 5,
        "KL": 32,
        "exact_read_margin": 32,
        "development_read_margin": 32
      },
      "maximum_response_drift": 0.00923919677734375,
      "median_response_drift": 0.003025054931640625,
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
      "maximum_response_drift": 0.000152587890625,
      "median_response_drift": 4.57763671875e-05,
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
      "maximum_response_drift": 0.000152587890625,
      "median_response_drift": 6.103515625e-05,
      "clipped_compensations": 0
    }
  },
  "new_local_by_case": {
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
        "maximum_response_drift": 0.00742340087890625,
        "median_response_drift": 0.005397796630859375,
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
        "maximum_response_drift": 0.000152587890625,
        "median_response_drift": 6.103515625e-05,
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
        "maximum_response_drift": 0.000152587890625,
        "median_response_drift": 8.7738037109375e-05,
        "clipped_compensations": 0
      }
    },
    "xor_01": {
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
        "maximum_response_drift": 0.006561279296875,
        "median_response_drift": 0.003528594970703125,
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
        "maximum_response_drift": 9.918212890625e-05,
        "median_response_drift": 3.4332275390625e-05,
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
        "maximum_response_drift": 0.0001220703125,
        "median_response_drift": 5.340576171875e-05,
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
        "maximum_response_drift": 0.00455474853515625,
        "median_response_drift": 0.002582550048828125,
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
        "maximum_response_drift": 0.0001068115234375,
        "median_response_drift": 5.7220458984375e-05,
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
        "maximum_response_drift": 6.866455078125e-05,
        "median_response_drift": 4.9591064453125e-05,
        "clipped_compensations": 0
      }
    },
    "xor_11": {
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
        "maximum_response_drift": 0.00923919677734375,
        "median_response_drift": 0.003192901611328125,
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
        "maximum_response_drift": 0.0001068115234375,
        "median_response_drift": 3.4332275390625e-05,
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
        "maximum_response_drift": 9.1552734375e-05,
        "median_response_drift": 4.9591064453125e-05,
        "clipped_compensations": 0
      }
    }
  },
  "branch_complete_acceptance": {
    "geometry6": true,
    "selected6": true
  },
  "all32_requested_states_retained": true,
  "invalid_preparations": 0,
  "state_generation_retries": 0,
  "C35_coefficients_frozen_sha256": "d95a6d84f312261d355bf2d584c878eff2caeb2b7d19d38d92e0bc8f1c1faec8",
  "C36_frozen_sha256": "6a694cf429590e007420a642cfed951a62c3253336f41df3faa9815ab4da8131",
  "deployed_writer_identity_sha256": "bfaa45ea93c2c9421220fb20873695a72a1d382d7105bd7eeee17e1be7631a36",
  "new_training_steps": 0,
  "test_based_branch_or_checkpoint_selection": false,
  "online_write_Jacobians": 0,
  "online_write_SVD_calls": 0,
  "online_write_correction_steps": 0,
  "online_write_backtracking": 0,
  "postwrite_response_used_for_proposal": false,
  "teacher_fallback_used": false,
  "offline_preparation_uses_geometry": true,
  "requested_output_bit_supplied": true,
  "fresh_state_seeds": true,
  "fresh_evaluation": false,
  "same_model_chart_access_prompts": true,
  "same_pipeline_local_replication_only": true,
  "independent_model_or_chart_replication": false,
  "LC1_same_model_local_replication_passed": true,
  "LC1_independently_confirmed": false,
  "internal_parameter_update": false,
  "controller_dispatch_tested": false,
  "full_Line_C_complete": false,
  "scope": "Prospectively fixed-seed same-model/chart local replication of both C35 frozen composed writers; no refitting or test-selected replacement. Offline geometry and external commitment remain. Not independent cross-model replication or full Line C.",
  "interpretation": "Both frozen branches are reported separately; no test-selected switch or refitting. This is an additional predeclared local replication using the original implementation and same model/chart, not independent replication. A passing affine geometry branch alone does not establish a need for learned calibration."
}

END C36 A100 frozen local replication exit= 0
DRIVE_BACKUP_OK /content/drive/MyDrive/GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1_RECOVERY/gpt2_c36_frozen_local_replication_results_1792142124359043303.zip
RESULT_ARCHIVE=/content/gpt2_c36_frozen_local_replication_a100_v1/session_1792141839856215886/gpt2_c36_frozen_local_replication_results.zip
唯一结果总包： /content/gpt2_c36_frozen_local_replication_a100_v1/session_1792141839856215886/gpt2_c36_frozen_local_replication_results.zip

```