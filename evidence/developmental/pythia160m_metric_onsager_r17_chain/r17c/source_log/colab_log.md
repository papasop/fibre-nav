1. **pythia160m\_metric\_budget\_scan\_r17c.zip**(application/zip) - 20704 bytes, last modified: 2026/9/2 - 100% done

```
Saving pythia160m_metric_budget_scan_r17c.zip to pythia160m_metric_budget_scan_r17c.zip
Running: /usr/bin/python3 /content/pythia_r17c_run/output/pythia160m_metric_budget_scan_r17c/pythia160m_metric_budget_scan_r17c.py --device cuda --outdir /content/pythia_r17c_results
{
  "protocol": "PYTHIA160M_SST2_AGNEWS_METRIC_ONSAGER_BUDGET_SCAN_R17C_DIAGNOSTIC",
  "mode": "same_seed_budget_calibration_diagnostic",
  "model": "EleutherAI/pythia-160m",
  "pretrained": true,
  "seed": 54217,
  "learning_target_L": "GLUE/SST-2 prompted binary sentiment loss",
  "response_map_R": "four AG News topic-margin coordinates on frozen disjoint inputs",
  "r_l_separation": "different datasets, prompts, labels, verbalizers and declared functionals",
  "r17b_seed_reuse_intentional": true,
  "frozen_step_multipliers": [
    1.0,
    1.15,
    1.3,
    1.45
  ],
  "frozen_calibration_eligibility": "response-budget utilization in [0.80,1.00] and zero-step fraction <=0.05",
  "frozen_selection_rule": "minimum final validation loss among eligible multipliers; ties choose smaller multiplier",
  "frozen_incremental_principle": "min <mhat,delta>+delta^T M delta/(2 eta) subject to DR(theta)delta=0; M=diag(sqrt(vhat)+eps)",
  "analytic_solution": "delta*=-eta[M^-1-M^-1 J^T(J M^-1 J^T)^-1 J M^-1]mhat",
  "frozen_global_response_budget": 0.004543482202852718,
  "arms": [
    "current_metric_m100",
    "current_metric_m115",
    "current_metric_m130",
    "current_metric_m145",
    "current_projected_adamw",
    "source_frozen_metric_m100"
  ],
  "data": {
    "learning_dataset": "glue/sst2",
    "response_dataset": "ag_news",
    "learning_train_sha256": "287a28a6a8c0966c1c13138e4f61fd8b1a70c1da2c78346b2dd01d1c4146d30c",
    "learning_validation_sha256": "f2669241cd7ccec00f54650078a9fbca273132145d872a85cc1070565138e4bc",
    "response_anchors_sha256": "8e94ec110e9faf8551f6b337c2c2f8cb9d777c9c3eea8aa10f7166d90e61ca58",
    "learning_train_examples": 2048,
    "learning_validation_examples": 256,
    "response_anchor_examples": 16,
    "response_anchor_examples_per_topic": 4,
    "r_l_datasets_distinct": true
  },
  "learning_verbalizers": [
    " negative",
    " positive"
  ],
  "response_verbalizers": [
    " World",
    " Sports",
    " Business",
    " Technology"
  ],
  "chart": {
    "dimension": 32,
    "lora_rank": 4,
    "layers": [
      10,
      11
    ]
  },
  "records": [
    {
      "arm": "current_metric_m100",
      "metric_step_multiplier": 1.0,
      "initial_validation_loss": 0.6441060900688171,
      "initial_validation_accuracy": 0.5703125,
      "final_validation_loss": 0.6330770254135132,
      "final_validation_accuracy": 0.58203125,
      "validation_loss_gain": 0.011029064655303955,
      "median_current_metric_onsager_regret": 0.0,
      "maximum_negative_regret_numerical": 0.0,
      "median_kkt_residual": 1.7907378831902735e-18,
      "maximum_linearized_response_leak": 1.1345420000681942e-12,
      "maximum_global_response_drift": 0.003370534912414239,
      "median_backtracks": 0.0,
      "zero_step_fraction": 0.0,
      "response_ranks": [
        4
      ],
      "maximum_projector_leakage": 4.876213228464686e-16,
      "maximum_projector_idempotence": 4.541708812094152e-16,
      "timed_seconds": 11.262768857000083,
      "trace": [
        {
          "step": 10,
          "validation_loss": 0.6434506177902222,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0005682746719218772,
          "onsager_regret": 0.0,
          "kkt_residual": 1.4624034116007647e-18
        },
        {
          "step": 20,
          "validation_loss": 0.6424948573112488,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0010847700553696423,
          "onsager_regret": 0.0,
          "kkt_residual": 1.8998511418536415e-18
        },
        {
          "step": 30,
          "validation_loss": 0.6419207453727722,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.001474112749155086,
          "onsager_regret": 0.0,
          "kkt_residual": 4.763218062587228e-19
        },
        {
          "step": 40,
          "validation_loss": 0.6392921209335327,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.001772254322294264,
          "onsager_regret": 0.0,
          "kkt_residual": 1.4811688079275457e-18
        },
        {
          "step": 50,
          "validation_loss": 0.6370963454246521,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.0020881138531473843,
          "onsager_regret": 0.0,
          "kkt_residual": 2.1898552322877385e-18
        },
        {
          "step": 60,
          "validation_loss": 0.6345717906951904,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.0026092975456483408,
          "onsager_regret": 0.0,
          "kkt_residual": 2.9955147431373266e-18
        },
        {
          "step": 70,
          "validation_loss": 0.6330863833427429,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.0030890286934352907,
          "onsager_regret": 0.0,
          "kkt_residual": 1.85377045361125e-18
        },
        {
          "step": 80,
          "validation_loss": 0.6330770254135132,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.0020209688644252243,
          "onsager_regret": 0.0,
          "kkt_residual": 2.2757049934228117e-18
        }
      ],
      "response_budget_utilization": 0.7418395763271571,
      "calibration_eligible": false
    },
    {
      "arm": "current_metric_m115",
      "metric_step_multiplier": 1.15,
      "initial_validation_loss": 0.6441060900688171,
      "initial_validation_accuracy": 0.5703125,
      "final_validation_loss": 0.6320953369140625,
      "final_validation_accuracy": 0.58203125,
      "validation_loss_gain": 0.012010753154754639,
      "median_current_metric_onsager_regret": 0.0,
      "maximum_negative_regret_numerical": 0.0,
      "median_kkt_residual": 1.794346302420446e-18,
      "maximum_linearized_response_leak": 1.2321434637849057e-12,
      "maximum_global_response_drift": 0.004539406399001793,
      "median_backtracks": 0.0,
      "zero_step_fraction": 0.0,
      "response_ranks": [
        4
      ],
      "maximum_projector_leakage": 4.501059415845425e-16,
      "maximum_projector_idempotence": 4.3068788137303113e-16,
      "timed_seconds": 13.515803368999968,
      "trace": [
        {
          "step": 10,
          "validation_loss": 0.6433209776878357,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0009341569741829814,
          "onsager_regret": 0.0,
          "kkt_residual": 2.4065778700554728e-18
        },
        {
          "step": 20,
          "validation_loss": 0.6418050527572632,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.0007090002536309169,
          "onsager_regret": 0.0,
          "kkt_residual": 1.4228550111174966e-18
        },
        {
          "step": 30,
          "validation_loss": 0.6410587430000305,
          "validation_accuracy": 0.578125,
          "response_drift": 0.0011202487086778353,
          "onsager_regret": 0.0,
          "kkt_residual": 1.1605600220077668e-18
        },
        {
          "step": 40,
          "validation_loss": 0.6381087303161621,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.002114485008243203,
          "onsager_regret": 0.0,
          "kkt_residual": 2.386053173929388e-18
        },
        {
          "step": 50,
          "validation_loss": 0.6352366805076599,
          "validation_accuracy": 0.578125,
          "response_drift": 0.0034858235226425545,
          "onsager_regret": 0.0,
          "kkt_residual": 2.6901601051509028e-18
        },
        {
          "step": 60,
          "validation_loss": 0.6326236724853516,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.004398926550202117,
          "onsager_regret": 0.0,
          "kkt_residual": 2.823800145216158e-18
        },
        {
          "step": 70,
          "validation_loss": 0.6321179866790771,
          "validation_accuracy": 0.5859375,
          "response_drift": 0.004412508328027404,
          "onsager_regret": 0.0,
          "kkt_residual": 1.429069095959402e-18
        },
        {
          "step": 80,
          "validation_loss": 0.6320953369140625,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.004512011283209191,
          "onsager_regret": 0.0,
          "kkt_residual": 2.3781209898046594e-18
        }
      ],
      "response_budget_utilization": 0.9991029339020265,
      "calibration_eligible": true
    },
    {
      "arm": "current_metric_m130",
      "metric_step_multiplier": 1.3,
      "initial_validation_loss": 0.6441060900688171,
      "initial_validation_accuracy": 0.5703125,
      "final_validation_loss": 0.6333790421485901,
      "final_validation_accuracy": 0.58203125,
      "validation_loss_gain": 0.01072704792022705,
      "median_current_metric_onsager_regret": 0.0,
      "maximum_negative_regret_numerical": 0.0,
      "median_kkt_residual": 1.893970492432135e-18,
      "maximum_linearized_response_leak": 1.3967283529026536e-12,
      "maximum_global_response_drift": 0.004539483334833314,
      "median_backtracks": 0.0,
      "zero_step_fraction": 0.175,
      "response_ranks": [
        4
      ],
      "maximum_projector_leakage": 4.26484450443342e-16,
      "maximum_projector_idempotence": 4.496285982784158e-16,
      "timed_seconds": 17.750384452000162,
      "trace": [
        {
          "step": 10,
          "validation_loss": 0.6431572437286377,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0008580270946966277,
          "onsager_regret": 0.0,
          "kkt_residual": 1.3856765219669903e-18
        },
        {
          "step": 20,
          "validation_loss": 0.6414684653282166,
          "validation_accuracy": 0.578125,
          "response_drift": 0.0021701302793120094,
          "onsager_regret": 0.0,
          "kkt_residual": 1.0364264129909266e-18
        },
        {
          "step": 30,
          "validation_loss": 0.6402521729469299,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.0033764359368819766,
          "onsager_regret": 0.0,
          "kkt_residual": 1.3705663158063116e-18
        },
        {
          "step": 40,
          "validation_loss": 0.6366614699363708,
          "validation_accuracy": 0.578125,
          "response_drift": 0.0038819750874848563,
          "onsager_regret": 0.0,
          "kkt_residual": 1.5413111691093472e-18
        },
        {
          "step": 50,
          "validation_loss": 0.6341983079910278,
          "validation_accuracy": 0.578125,
          "response_drift": 0.004539483334833314,
          "onsager_regret": 0.0,
          "kkt_residual": 4.163505648986863e-18
        },
        {
          "step": 60,
          "validation_loss": 0.6341665387153625,
          "validation_accuracy": 0.578125,
          "response_drift": 0.00449896280766599,
          "onsager_regret": 0.0,
          "kkt_residual": 2.0141581474102925e-18
        },
        {
          "step": 70,
          "validation_loss": 0.6341792345046997,
          "validation_accuracy": 0.578125,
          "response_drift": 0.004433301924457407,
          "onsager_regret": 0.0,
          "kkt_residual": 1.698491789370516e-18
        },
        {
          "step": 80,
          "validation_loss": 0.6333790421485901,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.0042904068207193285,
          "onsager_regret": 0.0,
          "kkt_residual": 2.91734674038362e-18
        }
      ],
      "response_budget_utilization": 0.9991198671325503,
      "calibration_eligible": false
    },
    {
      "arm": "current_metric_m145",
      "metric_step_multiplier": 1.45,
      "initial_validation_loss": 0.6441060900688171,
      "initial_validation_accuracy": 0.5703125,
      "final_validation_loss": 0.6318311095237732,
      "final_validation_accuracy": 0.5859375,
      "validation_loss_gain": 0.012274980545043945,
      "median_current_metric_onsager_regret": 0.0,
      "maximum_negative_regret_numerical": 0.0,
      "median_kkt_residual": 1.9533323364695493e-18,
      "maximum_linearized_response_leak": 1.9013172899722287e-12,
      "maximum_global_response_drift": 0.004536173915779469,
      "median_backtracks": 0.0,
      "zero_step_fraction": 0.1375,
      "response_ranks": [
        4
      ],
      "maximum_projector_leakage": 4.478879268199605e-16,
      "maximum_projector_idempotence": 4.001914076240266e-16,
      "timed_seconds": 16.8735332680003,
      "trace": [
        {
          "step": 10,
          "validation_loss": 0.6430938243865967,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0007963860077331475,
          "onsager_regret": 0.0,
          "kkt_residual": 1.5925120481767246e-18
        },
        {
          "step": 20,
          "validation_loss": 0.6409651637077332,
          "validation_accuracy": 0.578125,
          "response_drift": 0.0010621037936581331,
          "onsager_regret": 0.0,
          "kkt_residual": 1.36266331603804e-18
        },
        {
          "step": 30,
          "validation_loss": 0.6391260027885437,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.0020381767205206575,
          "onsager_regret": 0.0,
          "kkt_residual": 1.1579491845712366e-18
        },
        {
          "step": 40,
          "validation_loss": 0.635357141494751,
          "validation_accuracy": 0.578125,
          "response_drift": 0.0023070057831058436,
          "onsager_regret": 0.0,
          "kkt_residual": 1.869512052055517e-18
        },
        {
          "step": 50,
          "validation_loss": 0.6324911117553711,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.004517194348174703,
          "onsager_regret": 0.0,
          "kkt_residual": 2.9620198383263386e-18
        },
        {
          "step": 60,
          "validation_loss": 0.6320466995239258,
          "validation_accuracy": 0.5859375,
          "response_drift": 0.004390555808855922,
          "onsager_regret": 0.0,
          "kkt_residual": 2.0021078233461854e-18
        },
        {
          "step": 70,
          "validation_loss": 0.6319867372512817,
          "validation_accuracy": 0.5859375,
          "response_drift": 0.004531654841326174,
          "onsager_regret": 0.0,
          "kkt_residual": 1.7662075019584316e-18
        },
        {
          "step": 80,
          "validation_loss": 0.6318311095237732,
          "validation_accuracy": 0.5859375,
          "response_drift": 0.004471137651064313,
          "onsager_regret": 0.0,
          "kkt_residual": 2.3441569098438817e-18
        }
      ],
      "response_budget_utilization": 0.9983914788818452,
      "calibration_eligible": false
    },
    {
      "arm": "current_projected_adamw",
      "metric_step_multiplier": null,
      "initial_validation_loss": 0.6441060900688171,
      "initial_validation_accuracy": 0.5703125,
      "final_validation_loss": 0.6325869560241699,
      "final_validation_accuracy": 0.5859375,
      "validation_loss_gain": 0.011519134044647217,
      "median_current_metric_onsager_regret": 1.041916026625696e-06,
      "maximum_negative_regret_numerical": 0.0,
      "median_kkt_residual": 0.0008967030874353333,
      "maximum_linearized_response_leak": 1.7914472497780004e-18,
      "maximum_global_response_drift": 0.004526668361614986,
      "median_backtracks": 0.0,
      "zero_step_fraction": 0.0,
      "response_ranks": [
        4
      ],
      "maximum_projector_leakage": 4.3701943098341166e-16,
      "maximum_projector_idempotence": 4.054138278864961e-16,
      "timed_seconds": 11.22632749999957,
      "trace": [
        {
          "step": 10,
          "validation_loss": 0.6435093879699707,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0007121131524400656,
          "onsager_regret": 1.4165566730323109e-06,
          "kkt_residual": 0.0012009250767031276
        },
        {
          "step": 20,
          "validation_loss": 0.642365574836731,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.0005770163086300365,
          "onsager_regret": 1.5309781482192352e-06,
          "kkt_residual": 0.0010117517344099861
        },
        {
          "step": 30,
          "validation_loss": 0.641636848449707,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.002256392626468002,
          "onsager_regret": 9.110400583153128e-08,
          "kkt_residual": 0.00020628389001244447
        },
        {
          "step": 40,
          "validation_loss": 0.6394275426864624,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.0020634374186047693,
          "onsager_regret": 5.471398059796119e-07,
          "kkt_residual": 0.0005864021258026919
        },
        {
          "step": 50,
          "validation_loss": 0.6374610662460327,
          "validation_accuracy": 0.57421875,
          "response_drift": 0.002827079324091629,
          "onsager_regret": 1.2075333588358063e-06,
          "kkt_residual": 0.0010640924058640466
        },
        {
          "step": 60,
          "validation_loss": 0.6351615190505981,
          "validation_accuracy": 0.58203125,
          "response_drift": 0.003048284461395013,
          "onsager_regret": 7.759711544337859e-07,
          "kkt_residual": 0.0008291831584000076
        },
        {
          "step": 70,
          "validation_loss": 0.6332776546478271,
          "validation_accuracy": 0.5859375,
          "response_drift": 0.003966023171052392,
          "onsager_regret": 2.1671106580997842e-07,
          "kkt_residual": 0.0004147645617833295
        },
        {
          "step": 80,
          "validation_loss": 0.6325869560241699,
          "validation_accuracy": 0.5859375,
          "response_drift": 0.003704493348300511,
          "onsager_regret": 1.78134655807224e-06,
          "kkt_residual": 0.001198486846888136
        }
      ]
    },
    {
      "arm": "source_frozen_metric_m100",
      "metric_step_multiplier": null,
      "initial_validation_loss": 0.6441060900688171,
      "initial_validation_accuracy": 0.5703125,
      "final_validation_loss": 0.6412972211837769,
      "final_validation_accuracy": 0.578125,
      "validation_loss_gain": 0.002808868885040283,
      "median_current_metric_onsager_regret": -2.4346836471976266e-06,
      "maximum_negative_regret_numerical": 3.0826083803897625e-05,
      "median_kkt_residual": 0.00147483810270796,
      "maximum_linearized_response_leak": 0.0015015700193990755,
      "maximum_global_response_drift": 0.0045413294037729155,
      "median_backtracks": 6.0,
      "zero_step_fraction": 0.25,
      "response_ranks": [
        4
      ],
      "maximum_projector_leakage": 4.218479708505917e-16,
      "maximum_projector_idempotence": 4.251057999116375e-16,
      "timed_seconds": 23.165063633000045,
      "trace": [
        {
          "step": 10,
          "validation_loss": 0.6435367465019226,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.004384240711627141,
          "onsager_regret": 4.786385916529965e-06,
          "kkt_residual": 0.00095211039156386
        },
        {
          "step": 20,
          "validation_loss": 0.6433042287826538,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.004486577013391757,
          "onsager_regret": -1.8174525336251015e-06,
          "kkt_residual": 0.0004171637560433761
        },
        {
          "step": 30,
          "validation_loss": 0.6432255506515503,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.004534325747876409,
          "onsager_regret": 9.910525164888429e-07,
          "kkt_residual": 0.000799589012656056
        },
        {
          "step": 40,
          "validation_loss": 0.6432507634162903,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.004469236540817477,
          "onsager_regret": 4.310572543433173e-06,
          "kkt_residual": 0.0009862173884259875
        },
        {
          "step": 50,
          "validation_loss": 0.6431080102920532,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.004262531185595405,
          "onsager_regret": -1.1247296391733784e-05,
          "kkt_residual": 0.0023774083450498794
        },
        {
          "step": 60,
          "validation_loss": 0.6417291164398193,
          "validation_accuracy": 0.5703125,
          "response_drift": 0.004486992153703164,
          "onsager_regret": -1.4325693464656422e-05,
          "kkt_residual": 0.0022606979851504208
        },
        {
          "step": 70,
          "validation_loss": 0.6413166522979736,
          "validation_accuracy": 0.578125,
          "response_drift": 0.004446804890579245,
          "onsager_regret": -4.710491335592357e-06,
          "kkt_residual": 0.0018157875199287649
        },
        {
          "step": 80,
          "validation_loss": 0.6412972211837769,
          "validation_accuracy": 0.578125,
          "response_drift": 0.00442731076011665,
          "onsager_regret": -2.0350949228569533e-05,
          "kkt_residual": 0.0030389807467112624
        }
      ]
    }
  ],
  "selected_multiplier": 1.15,
  "selected_arm": "current_metric_m115",
  "control_minus_selected_final_loss": {
    "current_projected_adamw": 0.0004916191101074219,
    "source_frozen_metric_m100": 0.009201884269714355
  },
  "numerical_gates": {
    "all_runs_finite": true,
    "all_response_balls_respected": true,
    "float64_projector_leakage_at_most_1e_10": true,
    "projector_idempotence_at_most_1e_10": true,
    "response_rank_constant": true,
    "at_least_one_frozen_multiplier_budget_matched": true,
    "all_current_metric_kkt_residuals_at_most_1e_8": true,
    "all_current_metric_linearized_constraint_residuals_at_most_1e_8": true,
    "all_current_metric_onsager_regrets_numerically_zero": true
  },
  "scientific_gates": {
    "selected_multiplier_beats_source_frozen_metric": true,
    "selected_multiplier_beats_projected_adamw": true,
    "selected_multiplier_positive_learning_gain": true
  },
  "scientific_status": "R17C_BUDGET_MATCHED_METRIC_ONSAGER_CANDIDATE_SELECTED",
  "next_step_if_supported": "Freeze the selected multiplier and run at least five untouched R17d seeds without further tuning.",
  "wall_seconds": 108.31756234169006,
  "claim_boundary": "Same-seed post-R17b response-budget calibration on pretrained Pythia-160M. It may select one frozen multiplier for untouched-seed R17d confirmation but cannot confirm superiority itself. No continuous-action theorem, universal optimizer ordering, Principle-R theorem, or physical-law claim."
}


```