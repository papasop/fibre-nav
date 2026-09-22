```
CPU 模式，GPU 不参与计算。Profile: expression_confirm
Qwen confirmation V7：冻结受限接口＋两组新表达，96个检查点、192次生成；纯CPU，无训练。
请上传 qwen_confirm_cpu_v7.zip；续跑可同时上传一份 V7 *_results.zip。

```

1. **qwen\_confirm\_cpu\_v7.zip**(application/zip) - 43029 bytes, last modified: 2026/9/22 - 100% done

```
Saving qwen_confirm_cpu_v7.zip to qwen_confirm_cpu_v7.zip
Loading pinned Qwen3-1.7B: CPU float32; no Adapter or training.
`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:13<00:13, 13.11s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:15<00:00,  6.76s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:15<00:00,  7.71s/it]
The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
QWEN_CONFIRM_V7_POINT {"n": 1, "id": "historical-00-OR", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 2, "id": "historical-00-AND", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 3, "id": "historical-00-XOR", "raw": "0\n\nExplanation:  \nSince both inputs `A` and `B` are 0, they are the same. According to the XOR operation, the result is 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 4, "id": "historical-00-EQUAL", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 5, "id": "historical-01-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 6, "id": "historical-01-AND", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 7, "id": "historical-01-XOR", "raw": "1\n\nExplanation:  \n- A = 0  \n- B = 1  \n- XOR of 0 and 1 is 1 (since the bits are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 8, "id": "historical-01-EQUAL", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 9, "id": "historical-10-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 10, "id": "historical-10-AND", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 11, "id": "historical-10-XOR", "raw": "1\n\nExplanation:  \n- A = 1  \n- B = 0  \n- XOR of 1 and 0 is 1 (since the bits are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 12, "id": "historical-10-EQUAL", "raw": "0\n\nExplanation:  \nGiven:\n- A = 1  \n- B = 0  \n\nThe operation `A == B` checks if the two bits are equal. Since 1 ≠ 0, the result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 13, "id": "historical-11-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 14, "id": "historical-11-AND", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 15, "id": "historical-11-XOR", "raw": "0\n\nExplanation:  \nSince both inputs A and B are 1, they are the same. According to the XOR operation, the output is 0 when the inputs are the same.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 16, "id": "historical-11-EQUAL", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
STAGE_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_016_results.zip
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_016_results.zip

```

```
QWEN_CONFIRM_V7_POINT {"n": 17, "id": "fresh_a-00-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 18, "id": "fresh_a-00-AND", "raw": "0\n\nExplanation:  \nThe AND operation returns 1 only when both inputs are 1.  \nGiven:  \nA = 0  \nB = 0  \n\nSo, 0 AND 0 = 0  \n\nFinal result: **0**", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 19, "id": "fresh_a-00-XOR", "raw": "0\n\nExplanation:  \n- Input bits: A = 0, B = 0  \n- XOR: 0 ⊕ 0 = 0  \n- Result: 0 (since the two bits are the same)", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 20, "id": "fresh_a-00-EQUAL", "raw": "1\n\nExplanation:  \nGiven:\n- A = 0  \n- B = 0  \n\nSince A == B is true (both are 0), the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 21, "id": "fresh_a-01-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 22, "id": "fresh_a-01-AND", "raw": "0\n\nExplanation:  \n- Input A = 0  \n- Input B = 1  \n- AND operation: 0 AND 1 = 0  \n- Result: 0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 23, "id": "fresh_a-01-XOR", "raw": "1\n\nExplanation:  \n- A = 0  \n- B = 1  \n- XOR of 0 and 1 is 1 (since the bits are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 24, "id": "fresh_a-01-EQUAL", "raw": "0\n\nExplanation:  \n- A = 0  \n- B = 1  \n- A == B is false (0 ≠ 1)  \n- So the result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 25, "id": "fresh_a-10-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 26, "id": "fresh_a-10-AND", "raw": "0\n\nExplanation:  \n- Input A = 1  \n- Input B = 0  \n- AND operation: 1 AND 0 = 0  \n\nSo the Boolean result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 27, "id": "fresh_a-10-XOR", "raw": "1\n\nExplanation:  \n- Input bits: A = 1, B = 0  \n- XOR: 1 ⊕ 0 = 1 (since the bits are different)  \n- Result: **1**", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 28, "id": "fresh_a-10-EQUAL", "raw": "0\n\nExplanation:  \nGiven:\n- A = 1  \n- B = 0  \n\nThe expression `A == B` evaluates to `False`, which is represented as **0** in Boolean terms.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 29, "id": "fresh_a-11-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 30, "id": "fresh_a-11-AND", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 31, "id": "fresh_a-11-XOR", "raw": "0\n\nExplanation:  \nXOR returns 1 when the two input bits are different.  \nSince both A and B are 1, they are the same, so XOR returns 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 32, "id": "fresh_a-11-EQUAL", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
STAGE_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_032_results.zip
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_032_results.zip

```

```
QWEN_CONFIRM_V7_POINT {"n": 33, "id": "fresh_b-00-OR", "raw": "0\n\nExplanation: The OR operation returns 1 if at least one of the inputs is 1. In this case, both A and B are 0, so the result is 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 34, "id": "fresh_b-00-AND", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 35, "id": "fresh_b-00-XOR", "raw": "0\n\nExplanation:  \nSince both `A` and `B` are 0, their XOR is 0 (because they are the same).  \nSo the Boolean result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 36, "id": "fresh_b-00-EQUAL", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 37, "id": "fresh_b-01-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 38, "id": "fresh_b-01-AND", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 39, "id": "fresh_b-01-XOR", "raw": "1\n\nExplanation:  \n- `A = 0`  \n- `B = 1`  \n- XOR of 0 and 1 is 1 (since they are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 40, "id": "fresh_b-01-EQUAL", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 41, "id": "fresh_b-10-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 42, "id": "fresh_b-10-AND", "raw": "0\n\nExplanation: The AND operation returns 1 only when both inputs are 1. Since A = 1 and B = 0, the result is 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 43, "id": "fresh_b-10-XOR", "raw": "1\n\nExplanation:  \n- `A = 1`  \n- `B = 0`  \n- XOR of 1 and 0 is 1 (since they are different).  \n- So the Boolean result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 44, "id": "fresh_b-10-EQUAL", "raw": "0\n\nExplanation: Since $ A = 1 $ and $ B = 0 $, the values are different. Therefore, the result of $ A == B $ is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 45, "id": "fresh_b-11-OR", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 46, "id": "fresh_b-11-AND", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 47, "id": "fresh_b-11-XOR", "raw": "0\n\nExplanation:  \nSince both `A` and `B` are 1, the XOR operation returns 0 because the two input bits are the same.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 48, "id": "fresh_b-11-EQUAL", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
STAGE_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_048_results.zip
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_048_results.zip

```

```
QWEN_CONFIRM_V7_POINT {"n": 49, "id": "historical-00-OR-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 50, "id": "historical-00-AND-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 51, "id": "historical-00-XOR-repeat", "raw": "0\n\nExplanation:  \nSince both inputs `A` and `B` are 0, they are the same. According to the XOR operation, the result is 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 52, "id": "historical-00-EQUAL-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 53, "id": "historical-01-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 54, "id": "historical-01-AND-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 55, "id": "historical-01-XOR-repeat", "raw": "1\n\nExplanation:  \n- A = 0  \n- B = 1  \n- XOR of 0 and 1 is 1 (since the bits are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 56, "id": "historical-01-EQUAL-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 57, "id": "historical-10-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 58, "id": "historical-10-AND-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 59, "id": "historical-10-XOR-repeat", "raw": "1\n\nExplanation:  \n- A = 1  \n- B = 0  \n- XOR of 1 and 0 is 1 (since the bits are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 60, "id": "historical-10-EQUAL-repeat", "raw": "0\n\nExplanation:  \nGiven:\n- A = 1  \n- B = 0  \n\nThe operation `A == B` checks if the two bits are equal. Since 1 ≠ 0, the result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 61, "id": "historical-11-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 62, "id": "historical-11-AND-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 63, "id": "historical-11-XOR-repeat", "raw": "0\n\nExplanation:  \nSince both inputs A and B are 1, they are the same. According to the XOR operation, the output is 0 when the inputs are the same.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 64, "id": "historical-11-EQUAL-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
STAGE_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_064_results.zip
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_064_results.zip

```

```
QWEN_CONFIRM_V7_POINT {"n": 65, "id": "fresh_a-00-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 66, "id": "fresh_a-00-AND-repeat", "raw": "0\n\nExplanation:  \nThe AND operation returns 1 only when both inputs are 1.  \nGiven:  \nA = 0  \nB = 0  \n\nSo, 0 AND 0 = 0  \n\nFinal result: **0**", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 67, "id": "fresh_a-00-XOR-repeat", "raw": "0\n\nExplanation:  \n- Input bits: A = 0, B = 0  \n- XOR: 0 ⊕ 0 = 0  \n- Result: 0 (since the two bits are the same)", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 68, "id": "fresh_a-00-EQUAL-repeat", "raw": "1\n\nExplanation:  \nGiven:\n- A = 0  \n- B = 0  \n\nSince A == B is true (both are 0), the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 69, "id": "fresh_a-01-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 70, "id": "fresh_a-01-AND-repeat", "raw": "0\n\nExplanation:  \n- Input A = 0  \n- Input B = 1  \n- AND operation: 0 AND 1 = 0  \n- Result: 0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 71, "id": "fresh_a-01-XOR-repeat", "raw": "1\n\nExplanation:  \n- A = 0  \n- B = 1  \n- XOR of 0 and 1 is 1 (since the bits are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 72, "id": "fresh_a-01-EQUAL-repeat", "raw": "0\n\nExplanation:  \n- A = 0  \n- B = 1  \n- A == B is false (0 ≠ 1)  \n- So the result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 73, "id": "fresh_a-10-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 74, "id": "fresh_a-10-AND-repeat", "raw": "0\n\nExplanation:  \n- Input A = 1  \n- Input B = 0  \n- AND operation: 1 AND 0 = 0  \n\nSo the Boolean result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 75, "id": "fresh_a-10-XOR-repeat", "raw": "1\n\nExplanation:  \n- Input bits: A = 1, B = 0  \n- XOR: 1 ⊕ 0 = 1 (since the bits are different)  \n- Result: **1**", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 76, "id": "fresh_a-10-EQUAL-repeat", "raw": "0\n\nExplanation:  \nGiven:\n- A = 1  \n- B = 0  \n\nThe expression `A == B` evaluates to `False`, which is represented as **0** in Boolean terms.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 77, "id": "fresh_a-11-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 78, "id": "fresh_a-11-AND-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 79, "id": "fresh_a-11-XOR-repeat", "raw": "0\n\nExplanation:  \nXOR returns 1 when the two input bits are different.  \nSince both A and B are 1, they are the same, so XOR returns 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 80, "id": "fresh_a-11-EQUAL-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
STAGE_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_080_results.zip
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_080_results.zip

```

```
QWEN_CONFIRM_V7_POINT {"n": 81, "id": "fresh_b-00-OR-repeat", "raw": "0\n\nExplanation: The OR operation returns 1 if at least one of the inputs is 1. In this case, both A and B are 0, so the result is 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 82, "id": "fresh_b-00-AND-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 83, "id": "fresh_b-00-XOR-repeat", "raw": "0\n\nExplanation:  \nSince both `A` and `B` are 0, their XOR is 0 (because they are the same).  \nSo the Boolean result is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 84, "id": "fresh_b-00-EQUAL-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 85, "id": "fresh_b-01-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 86, "id": "fresh_b-01-AND-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 87, "id": "fresh_b-01-XOR-repeat", "raw": "1\n\nExplanation:  \n- `A = 0`  \n- `B = 1`  \n- XOR of 0 and 1 is 1 (since they are different).  \n- So the result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 88, "id": "fresh_b-01-EQUAL-repeat", "raw": "0", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 89, "id": "fresh_b-10-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 90, "id": "fresh_b-10-AND-repeat", "raw": "0\n\nExplanation: The AND operation returns 1 only when both inputs are 1. Since A = 1 and B = 0, the result is 0.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 91, "id": "fresh_b-10-XOR-repeat", "raw": "1\n\nExplanation:  \n- `A = 1`  \n- `B = 0`  \n- XOR of 1 and 0 is 1 (since they are different).  \n- So the Boolean result is **1**.", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 92, "id": "fresh_b-10-EQUAL-repeat", "raw": "0\n\nExplanation: Since $ A = 1 $ and $ B = 0 $, the values are different. Therefore, the result of $ A == B $ is **0**.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 93, "id": "fresh_b-11-OR-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 94, "id": "fresh_b-11-AND-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 95, "id": "fresh_b-11-XOR-repeat", "raw": "0\n\nExplanation:  \nSince both `A` and `B` are 1, the XOR operation returns 0 because the two input bits are the same.", "bounded": "0", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
QWEN_CONFIRM_V7_POINT {"n": 96, "id": "fresh_b-11-EQUAL-repeat", "raw": "1", "bounded": "1", "same_forward_mask_exact": true}
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
STAGE_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_096_results.zip
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_096_results.zip

```

```
RESULT_ARCHIVE=/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip
SUMMARY {"complete": true, "completed": 96, "decision": "FRESH_EXPRESSION_OR_REGRESSION_FAILURE", "overall_pass": false, "formal_l6_pass": false}
已保存：/content/qwen_confirm_cpu_v7/run/cpu_expression_confirm_results.zip

```

```
已保存：/content/qwen_confirm_cpu_v7/QWEN_CONFIRM_V7_SESSION_OUTPUT.txt

```

```
请确认浏览器下载完成。完整原始数据以 results.zip 为准。

```