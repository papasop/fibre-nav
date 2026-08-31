```
Upload picard_gpt2_pretrained_dynamic_r12b.zip

```

1. **picard\_gpt2\_pretrained\_dynamic\_r12b.zip**(application/zip) - 9892 bytes, last modified: 2026/8/31 - 100% done

```
Saving picard_gpt2_pretrained_dynamic_r12b.zip to picard_gpt2_pretrained_dynamic_r12b.zip
GPU: NVIDIA A100-SXM4-40GB
Running: /usr/bin/python3 /content/r12b_run/picard_gpt2_pretrained_r12b/picard_gpt2_pretrained_r12b.py --device cuda --outdir /content/picard_r12b_results --data-root /content/data
{
  "protocol": "PRETRAINED_GPT2_LORA_DYNAMIC_KERNEL_R12B_MULTISEED_DEVELOPMENT",
  "mode": "multiseed_development",
  "development_reference": "R12A_R2_PARETO_SWEEP_COMPLETE",
  "model": "openai-community/gpt2",
  "pretrained": true,
  "data_sha256": "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
  "seeds": [
    32211,
    32217,
    32229
  ],
  "budgets": [
    2e-05,
    5e-05
  ],
  "primary_arms": [
    "current_identity",
    "source_identity",
    "adamw_budgeted"
  ],
  "diagnostic_arms": [
    "adamw_unconstrained"
  ],
  "determinism": [
    {
      "seed": 32211,
      "disabled_dropout_modules": 37,
      "fixed_state_response_repeat_error": 0.0
    },
    {
      "seed": 32217,
      "disabled_dropout_modules": 37,
      "fixed_state_response_repeat_error": 0.0
    },
    {
      "seed": 32229,
      "disabled_dropout_modules": 37,
      "fixed_state_response_repeat_error": 0.0
    }
  ],
  "frozen_configuration": {
    "steps": 300,
    "warm_steps": 30,
    "chart_dim": 24,
    "lora_rank": 4,
    "layers": 2,
    "global_response_backtracking": true
  },
  "seed_pairs": [
    {
      "seed": 32211,
      "budget": 2e-05,
      "adamw_budgeted_minus_current": 0.0011970599492396872,
      "source_minus_current": 0.0011889934539794922,
      "current_beats_both": true
    },
    {
      "seed": 32217,
      "budget": 2e-05,
      "adamw_budgeted_minus_current": 0.0012100537617998341,
      "source_minus_current": 0.0011931260426836232,
      "current_beats_both": true
    },
    {
      "seed": 32229,
      "budget": 2e-05,
      "adamw_budgeted_minus_current": 0.0009737809499101857,
      "source_minus_current": 0.0009558995564775685,
      "current_beats_both": true
    },
    {
      "seed": 32211,
      "budget": 5e-05,
      "adamw_budgeted_minus_current": 0.001184980074564912,
      "source_minus_current": 0.0011612176895141602,
      "current_beats_both": true
    },
    {
      "seed": 32217,
      "budget": 5e-05,
      "adamw_budgeted_minus_current": 0.0011983315149945994,
      "source_minus_current": 0.0011559724807739258,
      "current_beats_both": true
    },
    {
      "seed": 32229,
      "budget": 5e-05,
      "adamw_budgeted_minus_current": 0.0009594360987348338,
      "source_minus_current": 0.0009315808614092091,
      "current_beats_both": true
    }
  ],
  "budget_results": [
    {
      "budget": 2e-05,
      "median_adamw_budgeted_minus_current": 0.0011970599492396872,
      "median_source_minus_current": 0.0011889934539794922,
      "dual_win_seeds": 3,
      "supports_development_gate": true
    },
    {
      "budget": 5e-05,
      "median_adamw_budgeted_minus_current": 0.001184980074564912,
      "median_source_minus_current": 0.0011559724807739258,
      "dual_win_seeds": 3,
      "supports_development_gate": true
    }
  ],
  "supporting_budget_count": 2,
  "candidate_for_untouched_seed_confirmation": true,
  "records": [
    {
      "seed": 32211,
      "arm": "current_identity",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.372983654340108,
      "best_validation_loss": 4.372983654340108,
      "max_global_response_drift": 1.2839233415447483e-05,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.025,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.756302028091773e-16,
      "max_projector_idempotence": 4.117684118716061e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 29.638834504999977,
      "geometry_seconds": 11.227752410992252,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374190966288249,
          "global_response_drift": 5.741879739185474e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374189178148906,
          "global_response_drift": 1.2006452857984707e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.37418262163798,
          "global_response_drift": 5.56082906231432e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374178369839986,
          "global_response_drift": 4.074098465594068e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.374172210693359,
          "global_response_drift": 6.198883056640625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.37415889898936,
          "global_response_drift": 8.753566623061094e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.374142011006673,
          "global_response_drift": 1.966049969490843e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 80,
          "validation_loss": 4.374123891194661,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 90,
          "validation_loss": 4.374105493227641,
          "global_response_drift": 7.539457464619587e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374091267585754,
          "global_response_drift": 2.86102294921875e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 110,
          "validation_loss": 4.374061028162639,
          "global_response_drift": 7.762346551942685e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 120,
          "validation_loss": 4.374036947886149,
          "global_response_drift": 1.0271386732357986e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 130,
          "validation_loss": 4.373997608820598,
          "global_response_drift": 1.0501249096653575e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 140,
          "validation_loss": 4.373962124188741,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 150,
          "validation_loss": 4.373916347821553,
          "global_response_drift": 6.9428538220219784e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 160,
          "validation_loss": 4.373872439066569,
          "global_response_drift": 1.1523290608973095e-05,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 170,
          "validation_loss": 4.373826662699382,
          "global_response_drift": 6.9428538220219784e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 180,
          "validation_loss": 4.373773415883382,
          "global_response_drift": 5.974752467223009e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 190,
          "validation_loss": 4.373728513717651,
          "global_response_drift": 4.396221378942913e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 200,
          "validation_loss": 4.373680035273234,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 210,
          "validation_loss": 4.373616774876912,
          "global_response_drift": 8.635888231408517e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 220,
          "validation_loss": 4.373560905456543,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 230,
          "validation_loss": 4.3735005458196,
          "global_response_drift": 8.792442757885825e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 240,
          "validation_loss": 4.373451630274455,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 250,
          "validation_loss": 4.373380939165751,
          "global_response_drift": 7.688768146799611e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 260,
          "validation_loss": 4.37330961227417,
          "global_response_drift": 1.2839233415447483e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 270,
          "validation_loss": 4.3732218742370605,
          "global_response_drift": 2.5678466830894965e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 280,
          "validation_loss": 4.373149633407593,
          "global_response_drift": 3.932099938981686e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 290,
          "validation_loss": 4.37307341893514,
          "global_response_drift": 4.264961199760036e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 300,
          "validation_loss": 4.372983654340108,
          "global_response_drift": 8.529922399520072e-06,
          "accepted_step_norm": 0.025
        }
      ]
    },
    {
      "seed": 32211,
      "arm": "source_identity",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374172647794087,
      "best_validation_loss": 4.374172647794087,
      "max_global_response_drift": 1.9970313636180805e-05,
      "zero_step_fraction": 0.5866666666666667,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 124,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 2.6666364473471536e-16,
      "max_projector_idempotence": 2.2188704481125572e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 129.834949185999,
      "geometry_seconds": 0.004811574977793498,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374190092086792,
          "global_response_drift": 1.066240299940009e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374186992645264,
          "global_response_drift": 8.635888231408517e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.374182939529419,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.3741774559021,
          "global_response_drift": 1.9073486328125e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.374176224072774,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 60,
          "validation_loss": 4.3741763432820635,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 70,
          "validation_loss": 4.374176740646362,
          "global_response_drift": 1.9896169705414835e-05,
          "accepted_step_norm": 0.0015625
        },
        {
          "step": 80,
          "validation_loss": 4.3741763432820635,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 6.103515625e-06
        },
        {
          "step": 90,
          "validation_loss": 4.374174435933431,
          "global_response_drift": 1.9660499694908432e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374172767003377,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 9.765625e-05
        },
        {
          "step": 110,
          "validation_loss": 4.374172687530518,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.374173005421956,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.374172846476237,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 6.103515625e-06
        },
        {
          "step": 140,
          "validation_loss": 4.374172846476237,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374172846476237,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374172925949097,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374172727266948,
          "global_response_drift": 1.9637356073354723e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374173045158386,
          "global_response_drift": 1.9833216040692732e-05,
          "accepted_step_norm": 2.4414062499999998e-05
        },
        {
          "step": 250,
          "validation_loss": 4.374173084894816,
          "global_response_drift": 1.9637356073354723e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374173084894816,
          "global_response_drift": 1.987330053727599e-05,
          "accepted_step_norm": 0.000390625
        },
        {
          "step": 270,
          "validation_loss": 4.374173045158386,
          "global_response_drift": 1.9619980441514e-05,
          "accepted_step_norm": 0.00078125
        },
        {
          "step": 280,
          "validation_loss": 4.374172965685527,
          "global_response_drift": 1.9833216040692732e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374172767003377,
          "global_response_drift": 1.9833216040692732e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374172647794087,
          "global_response_drift": 1.9410259663627262e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32211,
      "arm": "adamw_budgeted",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374180714289348,
      "best_validation_loss": 4.374180316925049,
      "max_global_response_drift": 1.9970313636180805e-05,
      "zero_step_fraction": 0.8266666666666667,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 52,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 2.6666364473471536e-16,
      "max_projector_idempotence": 2.2188704481125572e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 158.51841411199712,
      "geometry_seconds": 0.004889893007202772,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374181747436523,
          "global_response_drift": 1.9457059634307065e-05,
          "accepted_step_norm": 0.00022999579684928267
        },
        {
          "step": 20,
          "validation_loss": 4.374181906382243,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 5.285102154520857e-05
        },
        {
          "step": 30,
          "validation_loss": 4.374181389808655,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 40,
          "validation_loss": 4.374181350072225,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 6.534703794937871e-06
        },
        {
          "step": 50,
          "validation_loss": 4.374181191126506,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 3.2307269242448278e-06
        },
        {
          "step": 60,
          "validation_loss": 4.3741811116536455,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 70,
          "validation_loss": 4.374181270599365,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 5.1725684834762206e-05
        },
        {
          "step": 80,
          "validation_loss": 4.374181429545085,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 3.233201089663573e-06
        },
        {
          "step": 90,
          "validation_loss": 4.374180873235066,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.374181191126506,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 9.212313477640126e-05
        },
        {
          "step": 110,
          "validation_loss": 4.374181191126506,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.374181191126506,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.374181191126506,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374181191126506,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374180754025777,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374180912971497,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374181071917216,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374181071917216,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374181071917216,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.3741811116536455,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.3741804758707685,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 9.404065869402992e-05
        },
        {
          "step": 220,
          "validation_loss": 4.374180634816487,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374180316925049,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374180634816487,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374180714289348,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.00010458682791080878
        },
        {
          "step": 260,
          "validation_loss": 4.374180714289348,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.374180714289348,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374180555343628,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374180555343628,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374180714289348,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32211,
      "arm": "adamw_unconstrained",
      "budget": 2e-05,
      "budget_constrained": false,
      "final_validation_loss": 4.3689281940460205,
      "best_validation_loss": 4.3689281940460205,
      "max_global_response_drift": 0.01813813710900082,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.04940289331547332,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 2.6666364473471536e-16,
      "max_projector_idempotence": 2.2188704481125572e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 18.20915598099964,
      "geometry_seconds": 0.0049974629837379325,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374171654383342,
          "global_response_drift": 4.186673696375054e-05,
          "accepted_step_norm": 0.02988527969845707
        },
        {
          "step": 20,
          "validation_loss": 4.374152421951294,
          "global_response_drift": 9.95162690599151e-05,
          "accepted_step_norm": 0.030035574972169816
        },
        {
          "step": 30,
          "validation_loss": 4.3741246064503985,
          "global_response_drift": 0.00016421277595406818,
          "accepted_step_norm": 0.03199179071169356
        },
        {
          "step": 40,
          "validation_loss": 4.374091625213623,
          "global_response_drift": 0.00026683843161372846,
          "accepted_step_norm": 0.03436873631607728
        },
        {
          "step": 50,
          "validation_loss": 4.374045928319295,
          "global_response_drift": 0.00037542107469962184,
          "accepted_step_norm": 0.037138918253032685
        },
        {
          "step": 60,
          "validation_loss": 4.373989741007487,
          "global_response_drift": 0.0005463188893012148,
          "accepted_step_norm": 0.043910451596768006
        },
        {
          "step": 70,
          "validation_loss": 4.373911062876384,
          "global_response_drift": 0.0007461526459680796,
          "accepted_step_norm": 0.04075020682997651
        },
        {
          "step": 80,
          "validation_loss": 4.373825788497925,
          "global_response_drift": 0.0009896594108036606,
          "accepted_step_norm": 0.046086356473932895
        },
        {
          "step": 90,
          "validation_loss": 4.373716990152995,
          "global_response_drift": 0.0013029774937405455,
          "accepted_step_norm": 0.0494977719489136
        },
        {
          "step": 100,
          "validation_loss": 4.373610178629558,
          "global_response_drift": 0.0016427563297704256,
          "accepted_step_norm": 0.046965773585400584
        },
        {
          "step": 110,
          "validation_loss": 4.373472094535828,
          "global_response_drift": 0.002000108792819973,
          "accepted_step_norm": 0.04975538389167167
        },
        {
          "step": 120,
          "validation_loss": 4.37332006295522,
          "global_response_drift": 0.002447075938023699,
          "accepted_step_norm": 0.04844400031917977
        },
        {
          "step": 130,
          "validation_loss": 4.373166958491008,
          "global_response_drift": 0.002946370626882372,
          "accepted_step_norm": 0.053373308717673096
        },
        {
          "step": 140,
          "validation_loss": 4.372979323069255,
          "global_response_drift": 0.003460482950947843,
          "accepted_step_norm": 0.04976791234696481
        },
        {
          "step": 150,
          "validation_loss": 4.372752785682678,
          "global_response_drift": 0.004000134654743279,
          "accepted_step_norm": 0.04756372785914317
        },
        {
          "step": 160,
          "validation_loss": 4.372517466545105,
          "global_response_drift": 0.004647262869925407,
          "accepted_step_norm": 0.05228027538005158
        },
        {
          "step": 170,
          "validation_loss": 4.372235616048177,
          "global_response_drift": 0.00539135396788689,
          "accepted_step_norm": 0.05509479980141837
        },
        {
          "step": 180,
          "validation_loss": 4.371926784515381,
          "global_response_drift": 0.006168831165996877,
          "accepted_step_norm": 0.05130830523250028
        },
        {
          "step": 190,
          "validation_loss": 4.371665000915527,
          "global_response_drift": 0.006979728901473836,
          "accepted_step_norm": 0.05190959553932228
        },
        {
          "step": 200,
          "validation_loss": 4.371447245279948,
          "global_response_drift": 0.0077585673317411095,
          "accepted_step_norm": 0.05293296761845362
        },
        {
          "step": 210,
          "validation_loss": 4.371201515197754,
          "global_response_drift": 0.008622556029436897,
          "accepted_step_norm": 0.046882530579489025
        },
        {
          "step": 220,
          "validation_loss": 4.370961149533589,
          "global_response_drift": 0.0095525627654292,
          "accepted_step_norm": 0.054049280270191576
        },
        {
          "step": 230,
          "validation_loss": 4.370751976966858,
          "global_response_drift": 0.010516055217573154,
          "accepted_step_norm": 0.04972512175907035
        },
        {
          "step": 240,
          "validation_loss": 4.370658000310262,
          "global_response_drift": 0.011443309479040924,
          "accepted_step_norm": 0.04947256227957991
        },
        {
          "step": 250,
          "validation_loss": 4.37043297290802,
          "global_response_drift": 0.012534936180443832,
          "accepted_step_norm": 0.05846015874486669
        },
        {
          "step": 260,
          "validation_loss": 4.370186686515808,
          "global_response_drift": 0.013711255338614963,
          "accepted_step_norm": 0.05467695856491545
        },
        {
          "step": 270,
          "validation_loss": 4.369916756947835,
          "global_response_drift": 0.014858335738344768,
          "accepted_step_norm": 0.05367917228248884
        },
        {
          "step": 280,
          "validation_loss": 4.369643688201904,
          "global_response_drift": 0.016026588544382023,
          "accepted_step_norm": 0.04800559361870737
        },
        {
          "step": 290,
          "validation_loss": 4.369294961293538,
          "global_response_drift": 0.017070636562271358,
          "accepted_step_norm": 0.04342826033528889
        },
        {
          "step": 300,
          "validation_loss": 4.3689281940460205,
          "global_response_drift": 0.01813813710900082,
          "accepted_step_norm": 0.05104832249671376
        }
      ]
    },
    {
      "seed": 32211,
      "arm": "current_identity",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.372983654340108,
      "best_validation_loss": 4.372983654340108,
      "max_global_response_drift": 1.2839233415447483e-05,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.025,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.756302028091773e-16,
      "max_projector_idempotence": 4.117684118716061e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 29.557969514000433,
      "geometry_seconds": 11.31189071200788,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374190966288249,
          "global_response_drift": 5.741879739185474e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374189178148906,
          "global_response_drift": 1.2006452857984707e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.37418262163798,
          "global_response_drift": 5.56082906231432e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374178369839986,
          "global_response_drift": 4.074098465594068e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.374172210693359,
          "global_response_drift": 6.198883056640625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.37415889898936,
          "global_response_drift": 8.753566623061094e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.374142011006673,
          "global_response_drift": 1.966049969490843e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 80,
          "validation_loss": 4.374123891194661,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 90,
          "validation_loss": 4.374105493227641,
          "global_response_drift": 7.539457464619587e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374091267585754,
          "global_response_drift": 2.86102294921875e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 110,
          "validation_loss": 4.374061028162639,
          "global_response_drift": 7.762346551942685e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 120,
          "validation_loss": 4.374036947886149,
          "global_response_drift": 1.0271386732357986e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 130,
          "validation_loss": 4.373997608820598,
          "global_response_drift": 1.0501249096653575e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 140,
          "validation_loss": 4.373962124188741,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 150,
          "validation_loss": 4.373916347821553,
          "global_response_drift": 6.9428538220219784e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 160,
          "validation_loss": 4.373872439066569,
          "global_response_drift": 1.1523290608973095e-05,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 170,
          "validation_loss": 4.373826662699382,
          "global_response_drift": 6.9428538220219784e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 180,
          "validation_loss": 4.373773415883382,
          "global_response_drift": 5.974752467223009e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 190,
          "validation_loss": 4.373728513717651,
          "global_response_drift": 4.396221378942913e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 200,
          "validation_loss": 4.373680035273234,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 210,
          "validation_loss": 4.373616774876912,
          "global_response_drift": 8.635888231408517e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 220,
          "validation_loss": 4.373560905456543,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 230,
          "validation_loss": 4.3735005458196,
          "global_response_drift": 8.792442757885825e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 240,
          "validation_loss": 4.373451630274455,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 250,
          "validation_loss": 4.373380939165751,
          "global_response_drift": 7.688768146799611e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 260,
          "validation_loss": 4.37330961227417,
          "global_response_drift": 1.2839233415447483e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 270,
          "validation_loss": 4.3732218742370605,
          "global_response_drift": 2.5678466830894965e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 280,
          "validation_loss": 4.373149633407593,
          "global_response_drift": 3.932099938981686e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 290,
          "validation_loss": 4.37307341893514,
          "global_response_drift": 4.264961199760036e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 300,
          "validation_loss": 4.372983654340108,
          "global_response_drift": 8.529922399520072e-06,
          "accepted_step_norm": 0.025
        }
      ]
    },
    {
      "seed": 32211,
      "arm": "source_identity",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374144872029622,
      "best_validation_loss": 4.374143997828166,
      "max_global_response_drift": 4.997699295419747e-05,
      "zero_step_fraction": 0.47,
      "median_accepted_step_norm": 6.1035156249999995e-06,
      "accepted_steps": 159,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 12.0,
      "max_projector_leakage": 2.6666364473471536e-16,
      "max_projector_idempotence": 2.2188704481125572e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 117.8966509569982,
      "geometry_seconds": 0.005134062979777809,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374190092086792,
          "global_response_drift": 1.066240299940009e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374186992645264,
          "global_response_drift": 8.635888231408517e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.374182939529419,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.3741774559021,
          "global_response_drift": 1.9073486328125e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.3741640249888105,
          "global_response_drift": 3.411968959808029e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.374149560928345,
          "global_response_drift": 4.9093390183386806e-05,
          "accepted_step_norm": 0.00625
        },
        {
          "step": 70,
          "validation_loss": 4.3741481701533,
          "global_response_drift": 4.9480902752200495e-05,
          "accepted_step_norm": 0.0125
        },
        {
          "step": 80,
          "validation_loss": 4.374147931734721,
          "global_response_drift": 4.970326934609751e-05,
          "accepted_step_norm": 4.8828125e-05
        },
        {
          "step": 90,
          "validation_loss": 4.374147733052571,
          "global_response_drift": 4.9929199648698516e-05,
          "accepted_step_norm": 1.220703125e-05
        },
        {
          "step": 100,
          "validation_loss": 4.374146223068237,
          "global_response_drift": 4.8740128857348014e-05,
          "accepted_step_norm": 0.0062499999999999995
        },
        {
          "step": 110,
          "validation_loss": 4.374144872029622,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 1.220703125e-05
        },
        {
          "step": 120,
          "validation_loss": 4.374144872029622,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.3741447528203325,
          "global_response_drift": 4.968725558895977e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.3741445144017534,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.3741445144017534,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374144474665324,
          "global_response_drift": 4.990414679292414e-05,
          "accepted_step_norm": 2.44140625e-05
        },
        {
          "step": 170,
          "validation_loss": 4.3741445144017534,
          "global_response_drift": 4.990414679292414e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374144355456035,
          "global_response_drift": 4.990414679292414e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.3741441170374555,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0001953125
        },
        {
          "step": 200,
          "validation_loss": 4.374144156773885,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374143997828166,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374144156773885,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 1.52587890625e-06
        },
        {
          "step": 230,
          "validation_loss": 4.374144037564595,
          "global_response_drift": 4.990414679292414e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.3741441170374555,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374144792556763,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.000390625
        },
        {
          "step": 260,
          "validation_loss": 4.374145110448201,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.3741447528203325,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 1.2207031249999999e-05
        },
        {
          "step": 280,
          "validation_loss": 4.3741443157196045,
          "global_response_drift": 4.992236829852192e-05,
          "accepted_step_norm": 0.00078125
        },
        {
          "step": 290,
          "validation_loss": 4.374144872029622,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374144872029622,
          "global_response_drift": 4.990186863689854e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32211,
      "arm": "adamw_budgeted",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374168634414673,
      "best_validation_loss": 4.374168276786804,
      "max_global_response_drift": 4.988819751606203e-05,
      "zero_step_fraction": 0.8533333333333334,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 44,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 2.6666364473471536e-16,
      "max_projector_idempotence": 2.2188704481125572e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 163.4892034359982,
      "geometry_seconds": 0.005505091008672025,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374171654383342,
          "global_response_drift": 4.186673696375054e-05,
          "accepted_step_norm": 0.02988527969845707
        },
        {
          "step": 20,
          "validation_loss": 4.374168395996094,
          "global_response_drift": 4.950157674510694e-05,
          "accepted_step_norm": 2.8063445365296182e-05
        },
        {
          "step": 30,
          "validation_loss": 4.374168594678243,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 1.4129361673671677e-05
        },
        {
          "step": 40,
          "validation_loss": 4.374168395996094,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 50,
          "validation_loss": 4.374168713887532,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 60,
          "validation_loss": 4.3741687933603925,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 70,
          "validation_loss": 4.374168356259664,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 80,
          "validation_loss": 4.374168356259664,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 90,
          "validation_loss": 4.3741685549418134,
          "global_response_drift": 4.9783260864784954e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.374168475468953,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.374168475468953,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.374168475468953,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.374168515205383,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374168515205383,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374168515205383,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374168515205383,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374168475468953,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374168475468953,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374168475468953,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.374168276786804,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374168276786804,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.00011058149072585788
        },
        {
          "step": 230,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374168634414673,
          "global_response_drift": 4.9611692539894135e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "current_identity",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.372974872589111,
      "best_validation_loss": 4.372974872589111,
      "max_global_response_drift": 1.5355341137032144e-05,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.025,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.9873037487079396e-16,
      "max_projector_idempotence": 4.0656571636333047e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 30.41007772300145,
      "geometry_seconds": 11.674044881037844,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374189615249634,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374188264211019,
          "global_response_drift": 7.417845337917349e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.3741798003514605,
          "global_response_drift": 3.4714269110109892e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.3741748332977295,
          "global_response_drift": 6.0691461855687405e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.3741614818573,
          "global_response_drift": 4.1019083342755445e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.374142011006673,
          "global_response_drift": 3.3717478808715227e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.374128262201945,
          "global_response_drift": 8.327602957045068e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 80,
          "validation_loss": 4.374110023180644,
          "global_response_drift": 6.198883056640625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 90,
          "validation_loss": 4.374085982640584,
          "global_response_drift": 4.1019083342755445e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374057412147522,
          "global_response_drift": 1.966049969490843e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 110,
          "validation_loss": 4.374024391174316,
          "global_response_drift": 9.440894066440263e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 120,
          "validation_loss": 4.373991370201111,
          "global_response_drift": 7.215855574808863e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 130,
          "validation_loss": 4.373954971631368,
          "global_response_drift": 5.135693366178993e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 140,
          "validation_loss": 4.373921672503154,
          "global_response_drift": 8.120244200671388e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 150,
          "validation_loss": 4.3738804658253985,
          "global_response_drift": 4.978326086478496e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 160,
          "validation_loss": 4.373841365178426,
          "global_response_drift": 4.696301365755131e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 170,
          "validation_loss": 4.373792767524719,
          "global_response_drift": 3.724217260316207e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 180,
          "validation_loss": 4.373751521110535,
          "global_response_drift": 3.844384073399806e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 190,
          "validation_loss": 4.373713731765747,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 200,
          "validation_loss": 4.3736644585927325,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 210,
          "validation_loss": 4.373611330986023,
          "global_response_drift": 4.792154131470151e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 220,
          "validation_loss": 4.373558680216472,
          "global_response_drift": 4.862803948967728e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 230,
          "validation_loss": 4.373477498690288,
          "global_response_drift": 3.198720899820027e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 240,
          "validation_loss": 4.373421986897786,
          "global_response_drift": 3.814697265625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 250,
          "validation_loss": 4.3733720779418945,
          "global_response_drift": 2.5678466830894965e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 260,
          "validation_loss": 4.373313585917155,
          "global_response_drift": 7.463682099580063e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 270,
          "validation_loss": 4.373230576515198,
          "global_response_drift": 1.966049969490843e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 280,
          "validation_loss": 4.373156984647115,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 290,
          "validation_loss": 4.373070041338603,
          "global_response_drift": 1.066240299940009e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 300,
          "validation_loss": 4.372974872589111,
          "global_response_drift": 2.132480599880018e-06,
          "accepted_step_norm": 0.024999999999999998
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "source_identity",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374167998631795,
      "best_validation_loss": 4.3741676807403564,
      "max_global_response_drift": 1.9970313636180805e-05,
      "zero_step_fraction": 0.5566666666666666,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 133,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 1.880811911581309e-16,
      "max_projector_idempotence": 3.2638964663305215e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 125.89489062199937,
      "geometry_seconds": 0.004878718067629961,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374189615249634,
          "global_response_drift": 2.6973983046972182e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 9.258502883409309e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.3741780916849775,
          "global_response_drift": 1.117265178094862e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374174356460571,
          "global_response_drift": 1.9637356073354723e-05,
          "accepted_step_norm": 0.0015624999999999999
        },
        {
          "step": 50,
          "validation_loss": 4.374174237251282,
          "global_response_drift": 1.9221920366999028e-05,
          "accepted_step_norm": 0.00625
        },
        {
          "step": 60,
          "validation_loss": 4.37417201201121,
          "global_response_drift": 1.9913304345913984e-05,
          "accepted_step_norm": 0.0062499999999999995
        },
        {
          "step": 70,
          "validation_loss": 4.374167799949646,
          "global_response_drift": 1.9221920366999028e-05,
          "accepted_step_norm": 0.0015625000000000003
        },
        {
          "step": 80,
          "validation_loss": 4.3741685549418134,
          "global_response_drift": 1.9833216040692732e-05,
          "accepted_step_norm": 0.003125
        },
        {
          "step": 90,
          "validation_loss": 4.374167998631795,
          "global_response_drift": 1.9660499694908432e-05,
          "accepted_step_norm": 4.8828125e-05
        },
        {
          "step": 100,
          "validation_loss": 4.374168395996094,
          "global_response_drift": 1.985040502214303e-05,
          "accepted_step_norm": 1.5258789062499999e-06
        },
        {
          "step": 110,
          "validation_loss": 4.374168276786804,
          "global_response_drift": 1.9619980441514e-05,
          "accepted_step_norm": 1.52587890625e-06
        },
        {
          "step": 120,
          "validation_loss": 4.374168713887532,
          "global_response_drift": 1.9637356073354723e-05,
          "accepted_step_norm": 0.00625
        },
        {
          "step": 130,
          "validation_loss": 4.3741676807403564,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.000390625
        },
        {
          "step": 140,
          "validation_loss": 4.374167839686076,
          "global_response_drift": 1.987330053727599e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.3741679191589355,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 6.103515625e-06
        },
        {
          "step": 160,
          "validation_loss": 4.374168038368225,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374167998631795,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374167720476787,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374167720476787,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.3741681178410845,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.3741681178410845,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.3741681178410845,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.3741681178410845,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374168038368225,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374167998631795,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374167998631795,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.3741676807403564,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374167760213216,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374167760213216,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374167998631795,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "adamw_budgeted",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374184926350911,
      "best_validation_loss": 4.374184727668762,
      "max_global_response_drift": 1.9970313636180805e-05,
      "zero_step_fraction": 0.89,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 33,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 1.880811911581309e-16,
      "max_projector_idempotence": 3.2638964663305215e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 163.90115614299793,
      "geometry_seconds": 0.004825927986530587,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374185085296631,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 20,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 4.808320334958615e-05
        },
        {
          "step": 30,
          "validation_loss": 4.374185880025228,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 40,
          "validation_loss": 4.374185880025228,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 50,
          "validation_loss": 4.374184926350911,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 60,
          "validation_loss": 4.37418520450592,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 70,
          "validation_loss": 4.37418516476949,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 6.527719858215959e-06
        },
        {
          "step": 80,
          "validation_loss": 4.374185601870219,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 90,
          "validation_loss": 4.374185601870219,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.374185601870219,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.374185601870219,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.37418532371521,
          "global_response_drift": 1.971247349605025e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.37418536345164,
          "global_response_drift": 1.987330053727599e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.37418536345164,
          "global_response_drift": 1.987330053727599e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0004677614996688296
        },
        {
          "step": 220,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374184926350911,
          "global_response_drift": 1.9970313636180805e-05,
          "accepted_step_norm": 3.741832766184076e-06
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "adamw_unconstrained",
      "budget": 2e-05,
      "budget_constrained": false,
      "final_validation_loss": 4.3681589762369795,
      "best_validation_loss": 4.3681589762369795,
      "max_global_response_drift": 0.018480871251112772,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.04975507837399135,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 1.880811911581309e-16,
      "max_projector_idempotence": 3.2638964663305215e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 18.344487656999263,
      "geometry_seconds": 0.00467703296453692,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374177535374959,
          "global_response_drift": 4.491897144729876e-05,
          "accepted_step_norm": 0.02583816756588769
        },
        {
          "step": 20,
          "validation_loss": 4.374165574709575,
          "global_response_drift": 9.219249080995403e-05,
          "accepted_step_norm": 0.026857116576852687
        },
        {
          "step": 30,
          "validation_loss": 4.374136328697205,
          "global_response_drift": 0.0001660222087722176,
          "accepted_step_norm": 0.02930463927884257
        },
        {
          "step": 40,
          "validation_loss": 4.3741029103597,
          "global_response_drift": 0.0002553395084294038,
          "accepted_step_norm": 0.03217889400496249
        },
        {
          "step": 50,
          "validation_loss": 4.374051570892334,
          "global_response_drift": 0.0003733074638251078,
          "accepted_step_norm": 0.03585868923294088
        },
        {
          "step": 60,
          "validation_loss": 4.373982826868693,
          "global_response_drift": 0.0005202190684870172,
          "accepted_step_norm": 0.040259431200407925
        },
        {
          "step": 70,
          "validation_loss": 4.373908519744873,
          "global_response_drift": 0.0007084731538676664,
          "accepted_step_norm": 0.041545991863545746
        },
        {
          "step": 80,
          "validation_loss": 4.373836318651835,
          "global_response_drift": 0.0009296133321152917,
          "accepted_step_norm": 0.037861604285879896
        },
        {
          "step": 90,
          "validation_loss": 4.373742540677388,
          "global_response_drift": 0.0012195131529326283,
          "accepted_step_norm": 0.04600143719512713
        },
        {
          "step": 100,
          "validation_loss": 4.373615980148315,
          "global_response_drift": 0.001556452654895508,
          "accepted_step_norm": 0.047313899794108036
        },
        {
          "step": 110,
          "validation_loss": 4.373480359713237,
          "global_response_drift": 0.0019455587736795237,
          "accepted_step_norm": 0.0507321510105453
        },
        {
          "step": 120,
          "validation_loss": 4.373320817947388,
          "global_response_drift": 0.002405974816147342,
          "accepted_step_norm": 0.05035575376128319
        },
        {
          "step": 130,
          "validation_loss": 4.37316898504893,
          "global_response_drift": 0.00290352629728395,
          "accepted_step_norm": 0.05149689402250703
        },
        {
          "step": 140,
          "validation_loss": 4.373016913731893,
          "global_response_drift": 0.003490393311171159,
          "accepted_step_norm": 0.05539648882433906
        },
        {
          "step": 150,
          "validation_loss": 4.372828205426534,
          "global_response_drift": 0.004145673287952667,
          "accepted_step_norm": 0.048467856684769016
        },
        {
          "step": 160,
          "validation_loss": 4.372622450192769,
          "global_response_drift": 0.00473586909418433,
          "accepted_step_norm": 0.050582200482279766
        },
        {
          "step": 170,
          "validation_loss": 4.372357249259949,
          "global_response_drift": 0.005447324938940211,
          "accepted_step_norm": 0.054526925201261955
        },
        {
          "step": 180,
          "validation_loss": 4.372109293937683,
          "global_response_drift": 0.006199031993103802,
          "accepted_step_norm": 0.049271913276072506
        },
        {
          "step": 190,
          "validation_loss": 4.3718762795130415,
          "global_response_drift": 0.006937778607602193,
          "accepted_step_norm": 0.04540314013803434
        },
        {
          "step": 200,
          "validation_loss": 4.371642589569092,
          "global_response_drift": 0.007703484943671887,
          "accepted_step_norm": 0.04667990590313464
        },
        {
          "step": 210,
          "validation_loss": 4.371372580528259,
          "global_response_drift": 0.008571057087041598,
          "accepted_step_norm": 0.05700795055023202
        },
        {
          "step": 220,
          "validation_loss": 4.37106986840566,
          "global_response_drift": 0.009533867700525207,
          "accepted_step_norm": 0.05059656279926258
        },
        {
          "step": 230,
          "validation_loss": 4.370697498321533,
          "global_response_drift": 0.010529805279604617,
          "accepted_step_norm": 0.056519310292235427
        },
        {
          "step": 240,
          "validation_loss": 4.370337883631389,
          "global_response_drift": 0.011461142126991102,
          "accepted_step_norm": 0.04848912084767308
        },
        {
          "step": 250,
          "validation_loss": 4.370009462038676,
          "global_response_drift": 0.012485446940362957,
          "accepted_step_norm": 0.048722515629951
        },
        {
          "step": 260,
          "validation_loss": 4.369679808616638,
          "global_response_drift": 0.013490661618455873,
          "accepted_step_norm": 0.05290874920726093
        },
        {
          "step": 270,
          "validation_loss": 4.369355758031209,
          "global_response_drift": 0.014698521334847557,
          "accepted_step_norm": 0.056923001137970906
        },
        {
          "step": 280,
          "validation_loss": 4.368968208630879,
          "global_response_drift": 0.015910476483860504,
          "accepted_step_norm": 0.0503757312261105
        },
        {
          "step": 290,
          "validation_loss": 4.368587255477905,
          "global_response_drift": 0.017103103652818474,
          "accepted_step_norm": 0.05289807232297107
        },
        {
          "step": 300,
          "validation_loss": 4.3681589762369795,
          "global_response_drift": 0.018480871251112772,
          "accepted_step_norm": 0.061905739887382176
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "current_identity",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.372974872589111,
      "best_validation_loss": 4.372974872589111,
      "max_global_response_drift": 1.5355341137032144e-05,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.025,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.9873037487079396e-16,
      "max_projector_idempotence": 4.0656571636333047e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 29.80454051199922,
      "geometry_seconds": 11.296943493980507,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374189615249634,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374188264211019,
          "global_response_drift": 7.417845337917349e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.3741798003514605,
          "global_response_drift": 3.4714269110109892e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.3741748332977295,
          "global_response_drift": 6.0691461855687405e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.3741614818573,
          "global_response_drift": 4.1019083342755445e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.374142011006673,
          "global_response_drift": 3.3717478808715227e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.374128262201945,
          "global_response_drift": 8.327602957045068e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 80,
          "validation_loss": 4.374110023180644,
          "global_response_drift": 6.198883056640625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 90,
          "validation_loss": 4.374085982640584,
          "global_response_drift": 4.1019083342755445e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374057412147522,
          "global_response_drift": 1.966049969490843e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 110,
          "validation_loss": 4.374024391174316,
          "global_response_drift": 9.440894066440263e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 120,
          "validation_loss": 4.373991370201111,
          "global_response_drift": 7.215855574808863e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 130,
          "validation_loss": 4.373954971631368,
          "global_response_drift": 5.135693366178993e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 140,
          "validation_loss": 4.373921672503154,
          "global_response_drift": 8.120244200671388e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 150,
          "validation_loss": 4.3738804658253985,
          "global_response_drift": 4.978326086478496e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 160,
          "validation_loss": 4.373841365178426,
          "global_response_drift": 4.696301365755131e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 170,
          "validation_loss": 4.373792767524719,
          "global_response_drift": 3.724217260316207e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 180,
          "validation_loss": 4.373751521110535,
          "global_response_drift": 3.844384073399806e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 190,
          "validation_loss": 4.373713731765747,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 200,
          "validation_loss": 4.3736644585927325,
          "global_response_drift": 2.384185791015625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 210,
          "validation_loss": 4.373611330986023,
          "global_response_drift": 4.792154131470151e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 220,
          "validation_loss": 4.373558680216472,
          "global_response_drift": 4.862803948967728e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 230,
          "validation_loss": 4.373477498690288,
          "global_response_drift": 3.198720899820027e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 240,
          "validation_loss": 4.373421986897786,
          "global_response_drift": 3.814697265625e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 250,
          "validation_loss": 4.3733720779418945,
          "global_response_drift": 2.5678466830894965e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 260,
          "validation_loss": 4.373313585917155,
          "global_response_drift": 7.463682099580063e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 270,
          "validation_loss": 4.373230576515198,
          "global_response_drift": 1.966049969490843e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 280,
          "validation_loss": 4.373156984647115,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 290,
          "validation_loss": 4.373070041338603,
          "global_response_drift": 1.066240299940009e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 300,
          "validation_loss": 4.372974872589111,
          "global_response_drift": 2.132480599880018e-06,
          "accepted_step_norm": 0.024999999999999998
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "source_identity",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374130845069885,
      "best_validation_loss": 4.374130368232727,
      "max_global_response_drift": 4.999518790991239e-05,
      "zero_step_fraction": 0.5366666666666666,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 139,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 1.880811911581309e-16,
      "max_projector_idempotence": 3.2638964663305215e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 121.10043037499781,
      "geometry_seconds": 0.004703708000306506,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374189615249634,
          "global_response_drift": 2.6973983046972182e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374184727668762,
          "global_response_drift": 9.258502883409309e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.3741780916849775,
          "global_response_drift": 1.117265178094862e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374171257019043,
          "global_response_drift": 2.0100821407659522e-05,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 50,
          "validation_loss": 4.374156951904297,
          "global_response_drift": 3.558432237015845e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.374138236045837,
          "global_response_drift": 4.951076241539293e-05,
          "accepted_step_norm": 0.00078125
        },
        {
          "step": 70,
          "validation_loss": 4.374136249224345,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0001953125
        },
        {
          "step": 80,
          "validation_loss": 4.37413477897644,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0031250000000000006
        },
        {
          "step": 90,
          "validation_loss": 4.3741341431935625,
          "global_response_drift": 4.970326934609751e-05,
          "accepted_step_norm": 6.1035156249999995e-06
        },
        {
          "step": 100,
          "validation_loss": 4.374131361643474,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 9.765625e-05
        },
        {
          "step": 110,
          "validation_loss": 4.374131242434184,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.374130845069885,
          "global_response_drift": 4.963918332102479e-05,
          "accepted_step_norm": 6.103515625e-06
        },
        {
          "step": 130,
          "validation_loss": 4.374130646387736,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374130725860596,
          "global_response_drift": 4.963918332102479e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374130566914876,
          "global_response_drift": 4.963918332102479e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374130884806315,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374130884806315,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374131004015605,
          "global_response_drift": 4.963918332102479e-05,
          "accepted_step_norm": 1.220703125e-05
        },
        {
          "step": 190,
          "validation_loss": 4.374130368232727,
          "global_response_drift": 4.9803809346376496e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.374130566914876,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 6.103515625e-06
        },
        {
          "step": 210,
          "validation_loss": 4.374130566914876,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374130646387736,
          "global_response_drift": 4.9393517160715334e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374130725860596,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374130805333455,
          "global_response_drift": 4.9563546982522135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374130805333455,
          "global_response_drift": 4.9563546982522135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374130765597026,
          "global_response_drift": 4.9393517160715334e-05,
          "accepted_step_norm": 1.52587890625e-06
        },
        {
          "step": 270,
          "validation_loss": 4.374130765597026,
          "global_response_drift": 4.9393517160715334e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374130845069885,
          "global_response_drift": 4.9803809346376496e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374130845069885,
          "global_response_drift": 4.9803809346376496e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374130845069885,
          "global_response_drift": 4.9803809346376496e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32217,
      "arm": "adamw_budgeted",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374173204104106,
      "best_validation_loss": 4.374172965685527,
      "max_global_response_drift": 4.997699295419747e-05,
      "zero_step_fraction": 0.8466666666666667,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 46,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 1.880811911581309e-16,
      "max_projector_idempotence": 3.2638964663305215e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 158.3048427040012,
      "geometry_seconds": 0.0047364040328830015,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374177535374959,
          "global_response_drift": 4.491897144729876e-05,
          "accepted_step_norm": 0.02583816756588769
        },
        {
          "step": 20,
          "validation_loss": 4.37417455514272,
          "global_response_drift": 4.9822067548564905e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 30,
          "validation_loss": 4.37417467435201,
          "global_response_drift": 4.9822067548564905e-05,
          "accepted_step_norm": 6.514526268984115e-06
        },
        {
          "step": 40,
          "validation_loss": 4.374172965685527,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 50,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 60,
          "validation_loss": 4.3741733233133955,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 70,
          "validation_loss": 4.374173521995544,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 80,
          "validation_loss": 4.374173482259114,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 90,
          "validation_loss": 4.374173720677693,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.3741733233133955,
          "global_response_drift": 4.989503354471212e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.374173283576965,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 1.893428888542185e-06
        },
        {
          "step": 120,
          "validation_loss": 4.3741739590962725,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.3741739590962725,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374173402786255,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374173402786255,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374173243840535,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374173124631246,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374173204104106,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374173204104106,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "current_identity",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.373203158378601,
      "best_validation_loss": 4.373203158378601,
      "max_global_response_drift": 1.3283146168884391e-05,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.025,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.536713674849702e-16,
      "max_projector_idempotence": 3.6187954514672666e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 29.33741761499914,
      "geometry_seconds": 11.232252966005035,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374186992645264,
          "global_response_drift": 5.741879739185474e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374181389808655,
          "global_response_drift": 1.3486991523486091e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.37417205174764,
          "global_response_drift": 3.198720899820027e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374166011810303,
          "global_response_drift": 2.86102294921875e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.374157190322876,
          "global_response_drift": 5.761645304486548e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.374150991439819,
          "global_response_drift": 4.862803948967728e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.374142607053121,
          "global_response_drift": 4.76837158203125e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 80,
          "validation_loss": 4.374127944310506,
          "global_response_drift": 8.203816668551089e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 90,
          "validation_loss": 4.374114712079366,
          "global_response_drift": 3.337860107421875e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374101916948955,
          "global_response_drift": 8.70146159691556e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 110,
          "validation_loss": 4.374082883199056,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 120,
          "validation_loss": 4.37405526638031,
          "global_response_drift": 3.0532475649990313e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 130,
          "validation_loss": 4.37403138478597,
          "global_response_drift": 8.529922399520072e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 140,
          "validation_loss": 4.373996098836263,
          "global_response_drift": 3.015782985847835e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 150,
          "validation_loss": 4.3739713827768965,
          "global_response_drift": 5.135693366178993e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 160,
          "validation_loss": 4.373942097028096,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 170,
          "validation_loss": 4.373899777730306,
          "global_response_drift": 4.264961199760036e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 180,
          "validation_loss": 4.373857458432515,
          "global_response_drift": 4.264961199760036e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 190,
          "validation_loss": 4.373817205429077,
          "global_response_drift": 4.046097457045827e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 200,
          "validation_loss": 4.373767693837483,
          "global_response_drift": 4.046097457045827e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 210,
          "validation_loss": 4.373725334803264,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 220,
          "validation_loss": 4.373682260513306,
          "global_response_drift": 6.877043295791605e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 230,
          "validation_loss": 4.3736306031545,
          "global_response_drift": 3.3717478808715227e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 240,
          "validation_loss": 4.3735737800598145,
          "global_response_drift": 2.6973983046972182e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 250,
          "validation_loss": 4.3735179503758745,
          "global_response_drift": 7.688768146799611e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 260,
          "validation_loss": 4.373453179995219,
          "global_response_drift": 4.76837158203125e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 270,
          "validation_loss": 4.373397787412007,
          "global_response_drift": 5.7220458984375e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 280,
          "validation_loss": 4.373342315355937,
          "global_response_drift": 9.5367431640625e-07,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 290,
          "validation_loss": 4.3732757568359375,
          "global_response_drift": 0.0,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 300,
          "validation_loss": 4.373203158378601,
          "global_response_drift": 3.198720899820027e-06,
          "accepted_step_norm": 0.024999999999999998
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "source_identity",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374159057935079,
      "best_validation_loss": 4.3741583824157715,
      "max_global_response_drift": 1.9970313636180805e-05,
      "zero_step_fraction": 0.69,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 93,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 3.4942797114466386e-16,
      "max_projector_idempotence": 2.668519944724518e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 137.15062385599958,
      "geometry_seconds": 0.00534678100302699,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374187350273132,
          "global_response_drift": 5.56082906231432e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 20,
          "validation_loss": 4.37418270111084,
          "global_response_drift": 5.331201499700045e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 30,
          "validation_loss": 4.37417193253835,
          "global_response_drift": 1.3248867024850658e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.3741674820582075,
          "global_response_drift": 1.6689300537109375e-05,
          "accepted_step_norm": 0.0125
        },
        {
          "step": 50,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 1.89898564832114e-05,
          "accepted_step_norm": 0.0125
        },
        {
          "step": 60,
          "validation_loss": 4.3741592566172285,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0001953125
        },
        {
          "step": 70,
          "validation_loss": 4.37415858109792,
          "global_response_drift": 1.9573570014737885e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 80,
          "validation_loss": 4.374158461888631,
          "global_response_drift": 1.9741288704380467e-05,
          "accepted_step_norm": 1.52587890625e-06
        },
        {
          "step": 90,
          "validation_loss": 4.37415889898936,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.00039062499999999997
        },
        {
          "step": 100,
          "validation_loss": 4.374158620834351,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.37415877978007,
          "global_response_drift": 1.9457059634307065e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.37415870030721,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.3741583824157715,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.3741583824157715,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.37415870030721,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 2.4414062499999998e-05
        },
        {
          "step": 160,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.37415885925293,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.37415877978007,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.37415877978007,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.37415877978007,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374159057935079,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374159057935079,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374159057935079,
          "global_response_drift": 1.9844677015957654e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "adamw_budgeted",
      "budget": 2e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374176939328511,
      "best_validation_loss": 4.374176422754924,
      "max_global_response_drift": 1.9964620032747574e-05,
      "zero_step_fraction": 0.9,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 30,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 3.4942797114466386e-16,
      "max_projector_idempotence": 2.668519944724518e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 163.19861810599832,
      "geometry_seconds": 0.005473131001053844,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.3741770188013716,
          "global_response_drift": 1.8929894677428444e-05,
          "accepted_step_norm": 4.581765090262233e-05
        },
        {
          "step": 20,
          "validation_loss": 4.374176422754924,
          "global_response_drift": 1.9964620032747574e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 30,
          "validation_loss": 4.374176502227783,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 2.5735340094223945e-05
        },
        {
          "step": 40,
          "validation_loss": 4.374176502227783,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 50,
          "validation_loss": 4.374176502227783,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 60,
          "validation_loss": 4.37417717774709,
          "global_response_drift": 1.9896169705414835e-05,
          "accepted_step_norm": 0.00018543853068453382
        },
        {
          "step": 70,
          "validation_loss": 4.374176820119222,
          "global_response_drift": 1.9896169705414835e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 80,
          "validation_loss": 4.374176820119222,
          "global_response_drift": 1.9896169705414835e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 90,
          "validation_loss": 4.374177058537801,
          "global_response_drift": 1.9896169705414835e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.374177058537801,
          "global_response_drift": 1.9896169705414835e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.3741770188013716,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 1.1727186879456134e-05
        },
        {
          "step": 120,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374176939328511,
          "global_response_drift": 1.9919012617307112e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "adamw_unconstrained",
      "budget": 2e-05,
      "budget_constrained": false,
      "final_validation_loss": 4.368740200996399,
      "best_validation_loss": 4.368740200996399,
      "max_global_response_drift": 0.01866991422911621,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.05149332072783166,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.4942797114466386e-16,
      "max_projector_idempotence": 2.668519944724518e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 18.32588574499823,
      "geometry_seconds": 0.005286369010718772,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374172687530518,
          "global_response_drift": 3.291558238110031e-05,
          "accepted_step_norm": 0.023560063804119673
        },
        {
          "step": 20,
          "validation_loss": 4.374151229858398,
          "global_response_drift": 7.89248326728843e-05,
          "accepted_step_norm": 0.027762553517104955
        },
        {
          "step": 30,
          "validation_loss": 4.374122381210327,
          "global_response_drift": 0.00013828277587890625,
          "accepted_step_norm": 0.029389965025168344
        },
        {
          "step": 40,
          "validation_loss": 4.374089916547139,
          "global_response_drift": 0.0002145110135669018,
          "accepted_step_norm": 0.03053964498918134
        },
        {
          "step": 50,
          "validation_loss": 4.374049623807271,
          "global_response_drift": 0.00031677961628518884,
          "accepted_step_norm": 0.03663704874202339
        },
        {
          "step": 60,
          "validation_loss": 4.373995820681254,
          "global_response_drift": 0.0004387889108278552,
          "accepted_step_norm": 0.03424595285561704
        },
        {
          "step": 70,
          "validation_loss": 4.373940984408061,
          "global_response_drift": 0.0005899138633085744,
          "accepted_step_norm": 0.038790101135792315
        },
        {
          "step": 80,
          "validation_loss": 4.37387228012085,
          "global_response_drift": 0.0007855858534999915,
          "accepted_step_norm": 0.04059348266786625
        },
        {
          "step": 90,
          "validation_loss": 4.373802224795024,
          "global_response_drift": 0.0010388313087710808,
          "accepted_step_norm": 0.04454998564787671
        },
        {
          "step": 100,
          "validation_loss": 4.3737218379974365,
          "global_response_drift": 0.0013446071449027212,
          "accepted_step_norm": 0.04600252291576347
        },
        {
          "step": 110,
          "validation_loss": 4.373619596163432,
          "global_response_drift": 0.0016784584657505658,
          "accepted_step_norm": 0.048717969248754815
        },
        {
          "step": 120,
          "validation_loss": 4.373470187187195,
          "global_response_drift": 0.002059991105115939,
          "accepted_step_norm": 0.049753019248500376
        },
        {
          "step": 130,
          "validation_loss": 4.373330911000569,
          "global_response_drift": 0.002538255567028957,
          "accepted_step_norm": 0.050160491762705886
        },
        {
          "step": 140,
          "validation_loss": 4.373155434926351,
          "global_response_drift": 0.003096795529876225,
          "accepted_step_norm": 0.05115592780030938
        },
        {
          "step": 150,
          "validation_loss": 4.37297519048055,
          "global_response_drift": 0.003698689089518443,
          "accepted_step_norm": 0.05956681084461667
        },
        {
          "step": 160,
          "validation_loss": 4.372806787490845,
          "global_response_drift": 0.004336719107719393,
          "accepted_step_norm": 0.051411851618154446
        },
        {
          "step": 170,
          "validation_loss": 4.372609257698059,
          "global_response_drift": 0.005020389910364506,
          "accepted_step_norm": 0.05180712814511954
        },
        {
          "step": 180,
          "validation_loss": 4.372373700141907,
          "global_response_drift": 0.005767019360405456,
          "accepted_step_norm": 0.055220273619890306
        },
        {
          "step": 190,
          "validation_loss": 4.372127731641133,
          "global_response_drift": 0.006534122823888079,
          "accepted_step_norm": 0.054215929862377256
        },
        {
          "step": 200,
          "validation_loss": 4.371881405512492,
          "global_response_drift": 0.007386440546783556,
          "accepted_step_norm": 0.05195028032294047
        },
        {
          "step": 210,
          "validation_loss": 4.371626218159993,
          "global_response_drift": 0.008260335904178675,
          "accepted_step_norm": 0.05303037121730448
        },
        {
          "step": 220,
          "validation_loss": 4.371327638626099,
          "global_response_drift": 0.009192636229381863,
          "accepted_step_norm": 0.05517031839686987
        },
        {
          "step": 230,
          "validation_loss": 4.371012926101685,
          "global_response_drift": 0.010278440575847498,
          "accepted_step_norm": 0.058926000499232585
        },
        {
          "step": 240,
          "validation_loss": 4.370702346165975,
          "global_response_drift": 0.011382013521534198,
          "accepted_step_norm": 0.055370671878935136
        },
        {
          "step": 250,
          "validation_loss": 4.3703804810841875,
          "global_response_drift": 0.01250345278029131,
          "accepted_step_norm": 0.05444514020233588
        },
        {
          "step": 260,
          "validation_loss": 4.370059092839559,
          "global_response_drift": 0.013635434874782557,
          "accepted_step_norm": 0.058006804845661245
        },
        {
          "step": 270,
          "validation_loss": 4.369756778081258,
          "global_response_drift": 0.014843010712435389,
          "accepted_step_norm": 0.05387287935241968
        },
        {
          "step": 280,
          "validation_loss": 4.369447231292725,
          "global_response_drift": 0.016009078374916212,
          "accepted_step_norm": 0.053794303596698755
        },
        {
          "step": 290,
          "validation_loss": 4.369068066279094,
          "global_response_drift": 0.01732279818523898,
          "accepted_step_norm": 0.05485018392777816
        },
        {
          "step": 300,
          "validation_loss": 4.368740200996399,
          "global_response_drift": 0.01866991422911621,
          "accepted_step_norm": 0.04955877842914569
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "current_identity",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.373203158378601,
      "best_validation_loss": 4.373203158378601,
      "max_global_response_drift": 1.3283146168884391e-05,
      "zero_step_fraction": 0.0,
      "median_accepted_step_norm": 0.025,
      "accepted_steps": 300,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 0.0,
      "max_projector_leakage": 3.536713674849702e-16,
      "max_projector_idempotence": 3.6187954514672666e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 29.417143332000705,
      "geometry_seconds": 11.270281123997847,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374186992645264,
          "global_response_drift": 5.741879739185474e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 20,
          "validation_loss": 4.374181389808655,
          "global_response_drift": 1.3486991523486091e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 30,
          "validation_loss": 4.37417205174764,
          "global_response_drift": 3.198720899820027e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374166011810303,
          "global_response_drift": 2.86102294921875e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.374157190322876,
          "global_response_drift": 5.761645304486548e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 60,
          "validation_loss": 4.374150991439819,
          "global_response_drift": 4.862803948967728e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.374142607053121,
          "global_response_drift": 4.76837158203125e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 80,
          "validation_loss": 4.374127944310506,
          "global_response_drift": 8.203816668551089e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 90,
          "validation_loss": 4.374114712079366,
          "global_response_drift": 3.337860107421875e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 100,
          "validation_loss": 4.374101916948955,
          "global_response_drift": 8.70146159691556e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 110,
          "validation_loss": 4.374082883199056,
          "global_response_drift": 5.820539291254855e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 120,
          "validation_loss": 4.37405526638031,
          "global_response_drift": 3.0532475649990313e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 130,
          "validation_loss": 4.37403138478597,
          "global_response_drift": 8.529922399520072e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 140,
          "validation_loss": 4.373996098836263,
          "global_response_drift": 3.015782985847835e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 150,
          "validation_loss": 4.3739713827768965,
          "global_response_drift": 5.135693366178993e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 160,
          "validation_loss": 4.373942097028096,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 170,
          "validation_loss": 4.373899777730306,
          "global_response_drift": 4.264961199760036e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 180,
          "validation_loss": 4.373857458432515,
          "global_response_drift": 4.264961199760036e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 190,
          "validation_loss": 4.373817205429077,
          "global_response_drift": 4.046097457045827e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 200,
          "validation_loss": 4.373767693837483,
          "global_response_drift": 4.046097457045827e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 210,
          "validation_loss": 4.373725334803264,
          "global_response_drift": 3.4385216478958026e-06,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 220,
          "validation_loss": 4.373682260513306,
          "global_response_drift": 6.877043295791605e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 230,
          "validation_loss": 4.3736306031545,
          "global_response_drift": 3.3717478808715227e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 240,
          "validation_loss": 4.3735737800598145,
          "global_response_drift": 2.6973983046972182e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 250,
          "validation_loss": 4.3735179503758745,
          "global_response_drift": 7.688768146799611e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 260,
          "validation_loss": 4.373453179995219,
          "global_response_drift": 4.76837158203125e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 270,
          "validation_loss": 4.373397787412007,
          "global_response_drift": 5.7220458984375e-06,
          "accepted_step_norm": 0.025
        },
        {
          "step": 280,
          "validation_loss": 4.373342315355937,
          "global_response_drift": 9.5367431640625e-07,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 290,
          "validation_loss": 4.3732757568359375,
          "global_response_drift": 0.0,
          "accepted_step_norm": 0.025000000000000005
        },
        {
          "step": 300,
          "validation_loss": 4.373203158378601,
          "global_response_drift": 3.198720899820027e-06,
          "accepted_step_norm": 0.024999999999999998
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "source_identity",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.37413473924001,
      "best_validation_loss": 4.37413473924001,
      "max_global_response_drift": 4.999518790991239e-05,
      "zero_step_fraction": 0.5333333333333333,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 140,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 3.4942797114466386e-16,
      "max_projector_idempotence": 2.668519944724518e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 118.74426445499921,
      "geometry_seconds": 0.005032527988078073,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374187350273132,
          "global_response_drift": 5.56082906231432e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 20,
          "validation_loss": 4.37418270111084,
          "global_response_drift": 5.331201499700045e-06,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 30,
          "validation_loss": 4.37417193253835,
          "global_response_drift": 1.3248867024850658e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 40,
          "validation_loss": 4.374166806538899,
          "global_response_drift": 2.0275394082516956e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 50,
          "validation_loss": 4.374156792958577,
          "global_response_drift": 2.807271200091367e-05,
          "accepted_step_norm": 0.024999999999999998
        },
        {
          "step": 60,
          "validation_loss": 4.374150991439819,
          "global_response_drift": 3.5440262176632014e-05,
          "accepted_step_norm": 0.025
        },
        {
          "step": 70,
          "validation_loss": 4.37414010365804,
          "global_response_drift": 4.901923094137725e-05,
          "accepted_step_norm": 0.0015625
        },
        {
          "step": 80,
          "validation_loss": 4.374138832092285,
          "global_response_drift": 4.985856389448914e-05,
          "accepted_step_norm": 0.0007812500000000002
        },
        {
          "step": 90,
          "validation_loss": 4.374135613441467,
          "global_response_drift": 4.997699295419747e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.374135653177897,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.374135176340739,
          "global_response_drift": 4.9803809346376496e-05,
          "accepted_step_norm": 1.52587890625e-06
        },
        {
          "step": 120,
          "validation_loss": 4.374135295550029,
          "global_response_drift": 4.963918332102479e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.3741350173950195,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374135295550029,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374135295550029,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374135295550029,
          "global_response_drift": 4.988819751606203e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374135136604309,
          "global_response_drift": 4.9563546982522135e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374135573705037,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 0.0015625
        },
        {
          "step": 190,
          "validation_loss": 4.374135375022888,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 3.0517578124999997e-06
        },
        {
          "step": 200,
          "validation_loss": 4.374135096867879,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374135375022888,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374135255813599,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 6.103515625e-06
        },
        {
          "step": 230,
          "validation_loss": 4.374135255813599,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374135414759318,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374135176340739,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.37413505713145,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.3741352160771685,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374135414759318,
          "global_response_drift": 4.999518790991239e-05,
          "accepted_step_norm": 4.8828125e-05
        },
        {
          "step": 290,
          "validation_loss": 4.3741350173950195,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.37413473924001,
          "global_response_drift": 4.973756717836494e-05,
          "accepted_step_norm": 0.0
        }
      ]
    },
    {
      "seed": 32229,
      "arm": "adamw_budgeted",
      "budget": 5e-05,
      "budget_constrained": true,
      "final_validation_loss": 4.374162594477336,
      "best_validation_loss": 4.374162197113037,
      "max_global_response_drift": 4.997471811894952e-05,
      "zero_step_fraction": 0.8566666666666667,
      "median_accepted_step_norm": 0.0,
      "accepted_steps": 43,
      "max_restored_state_response_repeat_error": 0.0,
      "median_backtracks": 15.0,
      "max_projector_leakage": 3.4942797114466386e-16,
      "max_projector_idempotence": 2.668519944724518e-16,
      "response_ranks": [
        2
      ],
      "timed_seconds": 157.18526903499878,
      "geometry_seconds": 0.004836903070099652,
      "trace": [
        {
          "step": 10,
          "validation_loss": 4.374172687530518,
          "global_response_drift": 3.291558238110031e-05,
          "accepted_step_norm": 0.023560063804119673
        },
        {
          "step": 20,
          "validation_loss": 4.374162316322327,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 30,
          "validation_loss": 4.374162197113037,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 5.3774650649491205e-05
        },
        {
          "step": 40,
          "validation_loss": 4.374162316322327,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 1.3148557565596988e-05
        },
        {
          "step": 50,
          "validation_loss": 4.374162435531616,
          "global_response_drift": 4.9942859545807506e-05,
          "accepted_step_norm": 1.4384040493047954e-05
        },
        {
          "step": 60,
          "validation_loss": 4.374162673950195,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 70,
          "validation_loss": 4.374162673950195,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 80,
          "validation_loss": 4.374162673950195,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 90,
          "validation_loss": 4.374162673950195,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 100,
          "validation_loss": 4.374162435531616,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 110,
          "validation_loss": 4.374162435531616,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 120,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 130,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 140,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 150,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 160,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 170,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 180,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 190,
          "validation_loss": 4.374162634213765,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 200,
          "validation_loss": 4.374162554740906,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 210,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 220,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 230,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 240,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 250,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 260,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 270,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 280,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 290,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        },
        {
          "step": 300,
          "validation_loss": 4.374162594477336,
          "global_response_drift": 4.997471811894952e-05,
          "accepted_step_norm": 0.0
        }
      ]
    }
  ],
  "gates": {
    "three_new_development_seeds": true,
    "pretrained_checkpoint_loaded": true,
    "all_dropout_modules_disabled": true,
    "all_fixed_state_response_repeat_errors_at_most_1e_12": true,
    "all_runs_finite": true,
    "all_constrained_global_budgets_respected": true,
    "all_restored_state_repeat_errors_at_most_1e_12": true,
    "float64_projector_leakage_at_most_1e_10": true,
    "projector_idempotence_at_most_1e_10": true,
    "response_rank_constant": true,
    "every_constrained_arm_budget_accepts_at_least_one_step": true,
    "all_constrained_zero_step_fractions_below_0_95": true,
    "both_budgets_support_multiseed_development_gate": true
  },
  "scientific_status": "R12B_MULTISEED_DEVELOPMENT_CANDIDATE_SUPPORTED",
  "wall_seconds": 1959.2620890140533,
  "claim_boundary": "Three-seed pretrained GPT-2 development only. The unconstrained AdamW arm is diagnostic and excluded from matched-budget gates. A positive result may freeze R13 but cannot confirm optimizer superiority, semantic transfer, universality, or a Picard theorem."
}


```

```
An exception has occurred, use %tb to see the full traceback.


```

```
SystemExit: 0

```

```
/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
  warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)

```