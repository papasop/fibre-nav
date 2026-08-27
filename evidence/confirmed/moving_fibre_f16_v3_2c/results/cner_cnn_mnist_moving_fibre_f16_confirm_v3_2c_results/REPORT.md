# Moving-Fibre F16 prospective confirmation v3.2c

```json
{
  "scientific_status": "MOVING_FIBRE_F16_V32C_PROSPECTIVE_CONFIRMATION_SUPPORTED",
  "seeds": 16,
  "fully_comparable": 14,
  "required_comparable": 14,
  "counts": {
    "natural_action_minimum_at_all_radii": 14,
    "natural_action_minimum_at_smallest_radius": 14,
    "natural_action_beats_wrong_at_all_radii": 14,
    "all_algorithms_positive_retraction_scaling": 14,
    "all_algorithms_high_quality_scaling_fit": 14,
    "all_algorithms_action_converged": 14,
    "natural_wins_under_wrong_metric_at_any_radius": 0,
    "natural_rowspace_rotation_signal": 14,
    "natural_smallest_radius_cost_near_minimum": 0,
    "natural_smallest_radius_cost_persistently_higher": 14
  },
  "moving_fibre_f16_scaling_gate": true,
  "excluded": [
    {
      "seed": 73730,
      "reasons": [
        "arm:natural_gradient_r0"
      ]
    },
    {
      "seed": 73732,
      "reasons": [
        "arm:natural_gradient_r0"
      ]
    }
  ],
  "per_seed": [
    {
      "seed": 73726,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "adam",
        "adam",
        "sign_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.7792628327597628,
          "retraction_r2": 0.9874764762992575
        },
        "normalized_sgd": {
          "retraction_exponent": 0.7144217939266972,
          "retraction_r2": 0.9634784840650534
        },
        "normalized_momentum": {
          "retraction_exponent": 0.7140348800864758,
          "retraction_r2": 0.9639999740742561
        },
        "sign_gradient": {
          "retraction_exponent": 0.8165283436884544,
          "retraction_r2": 0.9858192858058838
        },
        "natural_gradient": {
          "retraction_exponent": 0.8714398702992218,
          "retraction_r2": 0.9774026846101025
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.8316319919187304,
          "retraction_r2": 0.9800538377347993
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.0034608634345381886,
        "normalized_sgd": 0.0009097619473678907,
        "normalized_momentum": 0.0008978709609493284,
        "sign_gradient": 0.005757505836289806,
        "natural_gradient": 0.0021458171714547076,
        "wrong_fisher_natural_gradient": 0.0011772739662219331
      },
      "natural_retraction_cost_over_minimum": [
        1.9358310560233496,
        1.8978539876716856,
        1.3785979775935635,
        1.8985910613778307
      ],
      "natural_actions": [
        0.3602662155412333,
        0.36083440785329807,
        0.3601921042568661,
        0.3609666727415578
      ],
      "natural_retraction_costs": [
        0.004001448779625294,
        0.0019428393639926201,
        0.000954732369864198,
        0.0006770756724779226
      ],
      "wrong_natural_actions": [
        0.3889857215659274,
        0.389817795616549,
        0.39063911394882705,
        0.39109954526166485
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.42614870382647335,
            0.42597949149543257,
            0.4241408360472387,
            0.42267800519447757
          ],
          "retraction_costs": [
            0.0020670444185585117,
            0.0010237032862449673,
            0.0007071368186898165,
            0.00038633886629815345
          ],
          "steps": [
            6,
            12,
            23,
            45
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.3844924215295538,
            0.382655091626558,
            0.38329684498883426,
            0.3836458714039023
          ],
          "retraction_costs": [
            0.0022706702631256964,
            0.0010417473709287029,
            0.0007992757501605691,
            0.000476030412236668
          ],
          "steps": [
            6,
            11,
            21,
            42
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.3843846680802968,
            0.3826544116631256,
            0.3832583483268624,
            0.3836027741182828
          ],
          "retraction_costs": [
            0.0022717312877052523,
            0.0010443938165827405,
            0.0007988406536906935,
            0.00047716869943088423
          ],
          "steps": [
            6,
            11,
            21,
            42
          ]
        },
        "sign_gradient": {
          "actions": [
            0.4416762541011714,
            0.4413597002296344,
            0.44168809458886277,
            0.44424584262049777
          ],
          "retraction_costs": [
            0.0021909866547443102,
            0.001024191062901471,
            0.0006925386409827383,
            0.0003784021920676324
          ],
          "steps": [
            6,
            12,
            24,
            47
          ]
        },
        "natural_gradient": {
          "actions": [
            0.3602662155412333,
            0.36083440785329807,
            0.3601921042568661,
            0.3609666727415578
          ],
          "retraction_costs": [
            0.004001448779625294,
            0.0019428393639926201,
            0.000954732369864198,
            0.0006770756724779226
          ],
          "steps": [
            5,
            10,
            19,
            38
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.3889857215659274,
            0.389817795616549,
            0.39063911394882705,
            0.39109954526166485
          ],
          "retraction_costs": [
            0.0021824567945584344,
            0.0010351961142891792,
            0.0007443556940238432,
            0.00035662006750761824
          ],
          "steps": [
            6,
            11,
            21,
            42
          ]
        }
      }
    },
    {
      "seed": 73727,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "wrong_fisher_natural_gradient",
        "normalized_sgd",
        "normalized_sgd",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.7352139936925963,
          "retraction_r2": 0.9915875560861172
        },
        "normalized_sgd": {
          "retraction_exponent": 0.7019912867171022,
          "retraction_r2": 0.9997399076664856
        },
        "normalized_momentum": {
          "retraction_exponent": 0.7104203507009719,
          "retraction_r2": 0.9996322369009588
        },
        "sign_gradient": {
          "retraction_exponent": 0.7616925510480103,
          "retraction_r2": 0.9845786147389086
        },
        "natural_gradient": {
          "retraction_exponent": 0.6089857327510914,
          "retraction_r2": 0.9959014319664856
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.6862164952178152,
          "retraction_r2": 0.9812152287779193
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.003871519619911282,
        "normalized_sgd": 0.0005794788158299343,
        "normalized_momentum": 0.0006688910668185741,
        "sign_gradient": 0.002904979343634406,
        "natural_gradient": 0.0005441844941484833,
        "wrong_fisher_natural_gradient": 0.0004994370802075008
      },
      "natural_retraction_cost_over_minimum": [
        1.58370364360378,
        1.6027436822499626,
        1.6140407969492347,
        1.9093144311472927
      ],
      "natural_actions": [
        0.6180594354198183,
        0.6199774079688364,
        0.620208825989954,
        0.6198715015304566
      ],
      "natural_retraction_costs": [
        0.0027148429164963015,
        0.0018879932085177494,
        0.0011415962979858941,
        0.0007861355263289129
      ],
      "wrong_natural_actions": [
        0.775212379570476,
        0.7742967355099549,
        0.7747321589498958,
        0.7751192822610408
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.8019817716015359,
            0.7963775863940421,
            0.7942501313979455,
            0.7911870352678865
          ],
          "retraction_costs": [
            0.0019527467307548798,
            0.0013445939782287407,
            0.0007186466337515858,
            0.00044014890293338834
          ],
          "steps": [
            11,
            22,
            44,
            87
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.7800151242091249,
            0.7803457235516985,
            0.7797532595943991,
            0.7793016707850383
          ],
          "retraction_costs": [
            0.0018877236842782196,
            0.0011779757608324172,
            0.000707290856677026,
            0.0004419601186179579
          ],
          "steps": [
            11,
            22,
            43,
            85
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.7801880687584591,
            0.7806106053604084,
            0.7799448900337143,
            0.7794235405901454
          ],
          "retraction_costs": [
            0.0018924660156671128,
            0.0011896589403376115,
            0.0007121006318836126,
            0.00043497303967750576
          ],
          "steps": [
            11,
            22,
            43,
            85
          ]
        },
        "sign_gradient": {
          "actions": [
            0.8158876007427442,
            0.8056658141676317,
            0.8044742555483626,
            0.8021440436708793
          ],
          "retraction_costs": [
            0.0022092884069962863,
            0.0015358443076274688,
            0.0008817131635170039,
            0.0004573879144944746
          ],
          "steps": [
            12,
            23,
            44,
            88
          ]
        },
        "natural_gradient": {
          "actions": [
            0.6180594354198183,
            0.6199774079688364,
            0.620208825989954,
            0.6198715015304566
          ],
          "retraction_costs": [
            0.0027148429164963015,
            0.0018879932085177494,
            0.0011415962979858941,
            0.0007861355263289129
          ],
          "steps": [
            9,
            18,
            35,
            70
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.775212379570476,
            0.7742967355099549,
            0.7747321589498958,
            0.7751192822610408
          ],
          "retraction_costs": [
            0.0017142367054978605,
            0.0012357662209184549,
            0.0007666043875161311,
            0.00041173706829237655
          ],
          "steps": [
            11,
            22,
            43,
            85
          ]
        }
      }
    },
    {
      "seed": 73728,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_momentum",
        "normalized_momentum",
        "normalized_momentum",
        "normalized_momentum"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.6913393351891398,
          "retraction_r2": 0.9891989845538939
        },
        "normalized_sgd": {
          "retraction_exponent": 0.721839099330829,
          "retraction_r2": 0.9988728573743685
        },
        "normalized_momentum": {
          "retraction_exponent": 0.7288469619903434,
          "retraction_r2": 0.9989809889476418
        },
        "sign_gradient": {
          "retraction_exponent": 0.807305017186176,
          "retraction_r2": 0.9820232117227002
        },
        "natural_gradient": {
          "retraction_exponent": 0.775269417807155,
          "retraction_r2": 0.9946846556454526
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.7095062006073848,
          "retraction_r2": 0.9877327065367001
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.007612372482177666,
        "normalized_sgd": 0.0008508813297166294,
        "normalized_momentum": 0.0008788544199367945,
        "sign_gradient": 0.001759238638611891,
        "natural_gradient": 0.0034244483332994,
        "wrong_fisher_natural_gradient": 5.687572626050541e-05
      },
      "natural_retraction_cost_over_minimum": [
        4.393275129671254,
        4.892541024153678,
        4.075391709463616,
        4.194320396944421
      ],
      "natural_actions": [
        0.2797292032754417,
        0.28316685273594266,
        0.2821144014414319,
        0.2830838073135709
      ],
      "natural_retraction_costs": [
        0.0026967510748728747,
        0.0017266000887708015,
        0.0008980019552593769,
        0.0005591751390877823
      ],
      "wrong_natural_actions": [
        0.5006665531256388,
        0.5008483881775575,
        0.5013473349865472,
        0.5013188221144513
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.4840144478883731,
            0.4826249011300194,
            0.47766757078585637,
            0.47405885817892257
          ],
          "retraction_costs": [
            0.0007706391245351209,
            0.00042415646824632525,
            0.00026511037884932364,
            0.0001824615045334009
          ],
          "steps": [
            7,
            14,
            27,
            53
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.4750798900074189,
            0.47604049580373436,
            0.47603204782985986,
            0.4764374395519526
          ],
          "retraction_costs": [
            0.0006147017963606806,
            0.0003544132330396709,
            0.00022172631143664034,
            0.0001355953026439893
          ],
          "steps": [
            7,
            13,
            26,
            52
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.4751780391238657,
            0.47607766120907086,
            0.47606309393637414,
            0.4764818521181278
          ],
          "retraction_costs": [
            0.00061383614621802,
            0.0003529045704975918,
            0.00022034739707942524,
            0.00013331722094839097
          ],
          "steps": [
            7,
            13,
            26,
            52
          ]
        },
        "sign_gradient": {
          "actions": [
            0.5214553835712159,
            0.4837531385478061,
            0.49372276864052367,
            0.4945928755375515
          ],
          "retraction_costs": [
            0.0010175171741739937,
            0.0006629159303697019,
            0.0003022996589550005,
            0.00020470978157989512
          ],
          "steps": [
            8,
            14,
            28,
            55
          ]
        },
        "natural_gradient": {
          "actions": [
            0.2797292032754417,
            0.28316685273594266,
            0.2821144014414319,
            0.2830838073135709
          ],
          "retraction_costs": [
            0.0026967510748728747,
            0.0017266000887708015,
            0.0008980019552593769,
            0.0005591751390877823
          ],
          "steps": [
            5,
            10,
            20,
            39
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.5006665531256388,
            0.5008483881775575,
            0.5013473349865472,
            0.5013188221144513
          ],
          "retraction_costs": [
            0.0007158033527196987,
            0.0004443102931318648,
            0.00023811259993583459,
            0.00017106174543186618
          ],
          "steps": [
            7,
            14,
            28,
            55
          ]
        }
      }
    },
    {
      "seed": 73729,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "wrong_fisher_natural_gradient",
        "sign_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.6306989392365889,
          "retraction_r2": 0.9863229026798701
        },
        "normalized_sgd": {
          "retraction_exponent": 0.7447980069973638,
          "retraction_r2": 0.9971625251929935
        },
        "normalized_momentum": {
          "retraction_exponent": 0.750335731084882,
          "retraction_r2": 0.9964903444374184
        },
        "sign_gradient": {
          "retraction_exponent": 0.5300417819448524,
          "retraction_r2": 0.979561339012749
        },
        "natural_gradient": {
          "retraction_exponent": 0.708760684562762,
          "retraction_r2": 0.9991170778011296
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.7919538347741618,
          "retraction_r2": 0.9909366981779675
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.006517277240775406,
        "normalized_sgd": 0.00033077058665766747,
        "normalized_momentum": 0.00047458175136968014,
        "sign_gradient": 0.0052079565355308294,
        "natural_gradient": 0.002859276030475149,
        "wrong_fisher_natural_gradient": 0.0005492992487900923
      },
      "natural_retraction_cost_over_minimum": [
        2.5220620285080515,
        2.485136162429268,
        2.7072064344596027,
        2.9275459622268363
      ],
      "natural_actions": [
        0.3578717717610047,
        0.3578732292953436,
        0.35691313179467565,
        0.35793657125329065
      ],
      "natural_retraction_costs": [
        0.0022376629137540113,
        0.0014231853155046038,
        0.0008616086918404511,
        0.0005143408745160122
      ],
      "wrong_natural_actions": [
        0.5672581968815695,
        0.5675041479321248,
        0.5682483469852252,
        0.5685606569269668
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.5274725685984203,
            0.5231573009218429,
            0.5265056331443038,
            0.5299595263036144
          ],
          "retraction_costs": [
            0.0010350554787630696,
            0.0005872244139514298,
            0.0004401997473491802,
            0.00026534896832280486
          ],
          "steps": [
            9,
            17,
            33,
            66
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.5576419168584045,
            0.5582181862693657,
            0.557665828012563,
            0.557481429553059
          ],
          "retraction_costs": [
            0.0009536076016373764,
            0.000617292426876998,
            0.0003527008604213968,
            0.00020560927967008499
          ],
          "steps": [
            9,
            17,
            33,
            66
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.5580165028061087,
            0.5584928956189138,
            0.5577745334112588,
            0.5575099493630841
          ],
          "retraction_costs": [
            0.000952984187163226,
            0.0006185092380798526,
            0.00035445892155578236,
            0.00020265963112690795
          ],
          "steps": [
            9,
            17,
            33,
            66
          ]
        },
        "sign_gradient": {
          "actions": [
            0.5161954539873129,
            0.5278299682129908,
            0.528241204197797,
            0.5310066638230647
          ],
          "retraction_costs": [
            0.0009679884596746661,
            0.0005726790093116721,
            0.0004482171358221763,
            0.00030866178609615394
          ],
          "steps": [
            9,
            17,
            33,
            67
          ]
        },
        "natural_gradient": {
          "actions": [
            0.3578717717610047,
            0.3578732292953436,
            0.35691313179467565,
            0.35793657125329065
          ],
          "retraction_costs": [
            0.0022376629137540113,
            0.0014231853155046038,
            0.0008616086918404511,
            0.0005143408745160122
          ],
          "steps": [
            6,
            12,
            23,
            45
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.5672581968815695,
            0.5675041479321248,
            0.5682483469852252,
            0.5685606569269668
          ],
          "retraction_costs": [
            0.0008872354797228049,
            0.0005983186510264342,
            0.00031826486553562013,
            0.00017569011081376126
          ],
          "steps": [
            9,
            17,
            34,
            67
          ]
        }
      }
    },
    {
      "seed": 73731,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "wrong_fisher_natural_gradient",
        "normalized_sgd",
        "normalized_sgd",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.7133113826215609,
          "retraction_r2": 0.9899330820551624
        },
        "normalized_sgd": {
          "retraction_exponent": 0.7977780482956645,
          "retraction_r2": 0.9243002768027735
        },
        "normalized_momentum": {
          "retraction_exponent": 0.7954767055684446,
          "retraction_r2": 0.9254111311371981
        },
        "sign_gradient": {
          "retraction_exponent": 0.6849969020943318,
          "retraction_r2": 0.9680621467103341
        },
        "natural_gradient": {
          "retraction_exponent": 0.6621084718837831,
          "retraction_r2": 0.9986129426125245
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.716027313894867,
          "retraction_r2": 0.8786856350089066
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.006996302033481007,
        "normalized_sgd": 0.0009527238717556547,
        "normalized_momentum": 0.0010190787635455936,
        "sign_gradient": 0.0010893807695209548,
        "natural_gradient": 0.005982923245901199,
        "wrong_fisher_natural_gradient": 0.0014530391799314145
      },
      "natural_retraction_cost_over_minimum": [
        2.579713596850303,
        2.7484555450318773,
        3.2228530607973838,
        3.0539591988694097
      ],
      "natural_actions": [
        0.36606174462493957,
        0.3565059710439676,
        0.3616578485873606,
        0.3595069461223423
      ],
      "natural_retraction_costs": [
        0.002735824366335168,
        0.0017229183972392033,
        0.0011337784066241005,
        0.0006812168067217571
      ],
      "wrong_natural_actions": [
        0.5251814118968285,
        0.5250138319664299,
        0.5254062142144719,
        0.524643886092468
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.5100091439390485,
            0.49891379009605413,
            0.49608668618522456,
            0.49264002775725235
          ],
          "retraction_costs": [
            0.001087762432702728,
            0.000768119044351317,
            0.0004345751546584453,
            0.000253062733164519
          ],
          "steps": [
            8,
            15,
            29,
            57
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.5206254688073023,
            0.5203291864894962,
            0.5205938242246977,
            0.5200983141451517
          ],
          "retraction_costs": [
            0.0015263260690009958,
            0.0006268678423245956,
            0.0003517933908980592,
            0.00029292719005275496
          ],
          "steps": [
            8,
            15,
            30,
            60
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.5205426625182187,
            0.5205003385585484,
            0.5206829528376864,
            0.5201528760878681
          ],
          "retraction_costs": [
            0.0015325737097613612,
            0.0006342923448348723,
            0.0003553930650547976,
            0.0002958514582018831
          ],
          "steps": [
            8,
            15,
            30,
            60
          ]
        },
        "sign_gradient": {
          "actions": [
            0.5126441467190206,
            0.49449052354942785,
            0.5001231148196792,
            0.5006685334919696
          ],
          "retraction_costs": [
            0.0013152202488177082,
            0.0008103760354205656,
            0.00041833689827212205,
            0.00033679831558037896
          ],
          "steps": [
            8,
            15,
            29,
            57
          ]
        },
        "natural_gradient": {
          "actions": [
            0.36606174462493957,
            0.3565059710439676,
            0.3616578485873606,
            0.3595069461223423
          ],
          "retraction_costs": [
            0.002735824366335168,
            0.0017229183972392033,
            0.0011337784066241005,
            0.0006812168067217571
          ],
          "steps": [
            6,
            11,
            22,
            43
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.5251814118968285,
            0.5250138319664299,
            0.5254062142144719,
            0.524643886092468
          ],
          "retraction_costs": [
            0.0010605147678701497,
            0.0008216513250409162,
            0.0006173257016653834,
            0.0002230602186741548
          ],
          "steps": [
            8,
            15,
            30,
            60
          ]
        }
      }
    },
    {
      "seed": 73733,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "normalized_momentum",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.6527023553420076,
          "retraction_r2": 0.9803545518853939
        },
        "normalized_sgd": {
          "retraction_exponent": 0.6319785056518834,
          "retraction_r2": 0.9956502377754238
        },
        "normalized_momentum": {
          "retraction_exponent": 0.65724622662949,
          "retraction_r2": 0.9911318197879493
        },
        "sign_gradient": {
          "retraction_exponent": 0.8239716585330012,
          "retraction_r2": 0.9830601959976348
        },
        "natural_gradient": {
          "retraction_exponent": 0.5991026920021764,
          "retraction_r2": 0.9576288062282239
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.6540976822456342,
          "retraction_r2": 0.9901585052266899
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.007721265222368314,
        "normalized_sgd": 0.0006516929433810392,
        "normalized_momentum": 0.0003595059620656148,
        "sign_gradient": 0.002827351873648784,
        "natural_gradient": 8.852760624976515e-05,
        "wrong_fisher_natural_gradient": 0.0001676445035094462
      },
      "natural_retraction_cost_over_minimum": [
        1.4805317127157867,
        1.5199963989678393,
        1.9010349746302837,
        1.6021615238173263
      ],
      "natural_actions": [
        0.6006931258602016,
        0.5949070680997796,
        0.5945395124488364,
        0.5944868839480536
      ],
      "natural_retraction_costs": [
        0.0019494482894155966,
        0.0011898048548432429,
        0.0009972073272923666,
        0.0005179830426248101
      ],
      "wrong_natural_actions": [
        0.7403960739998041,
        0.7400860196930403,
        0.7399875271537166,
        0.7401116027959089
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.7484616998314471,
            0.7406891991347487,
            0.7337196337586983,
            0.7280977975559466
          ],
          "retraction_costs": [
            0.0016677189364456653,
            0.0008767475105158039,
            0.0006448726899192136,
            0.000408927597971882
          ],
          "steps": [
            11,
            22,
            42,
            83
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.744915383081206,
            0.7449894806210664,
            0.7450437484474844,
            0.7455296048300338
          ],
          "retraction_costs": [
            0.0013558801554747377,
            0.000941323347828985,
            0.0005586624636193882,
            0.00037463020901130277
          ],
          "steps": [
            11,
            21,
            42,
            84
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.744883473384459,
            0.74557508870171,
            0.745469521174604,
            0.7457376182945175
          ],
          "retraction_costs": [
            0.0013291464300947315,
            0.0009378548086655723,
            0.0005245602214584743,
            0.00035333241773720505
          ],
          "steps": [
            11,
            21,
            42,
            84
          ]
        },
        "sign_gradient": {
          "actions": [
            0.7722729524828912,
            0.7351192959053159,
            0.7532109911182843,
            0.7510874027428652
          ],
          "retraction_costs": [
            0.0022634940588603437,
            0.0016009230850716671,
            0.0007826203294811115,
            0.00042813939427707617
          ],
          "steps": [
            12,
            22,
            44,
            87
          ]
        },
        "natural_gradient": {
          "actions": [
            0.6006931258602016,
            0.5949070680997796,
            0.5945395124488364,
            0.5944868839480536
          ],
          "retraction_costs": [
            0.0019494482894155966,
            0.0011898048548432429,
            0.0009972073272923666,
            0.0005179830426248101
          ],
          "steps": [
            9,
            17,
            33,
            65
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.7403960739998041,
            0.7400860196930403,
            0.7399875271537166,
            0.7401116027959089
          ],
          "retraction_costs": [
            0.0013167217376517126,
            0.0007827682063267949,
            0.0005678808657638804,
            0.000323302635174173
          ],
          "steps": [
            11,
            21,
            42,
            83
          ]
        }
      }
    },
    {
      "seed": 73734,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_momentum",
        "wrong_fisher_natural_gradient",
        "normalized_momentum",
        "normalized_momentum"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.6985695799226959,
          "retraction_r2": 0.9997191491394657
        },
        "normalized_sgd": {
          "retraction_exponent": 0.5618831470891529,
          "retraction_r2": 0.9908993101749505
        },
        "normalized_momentum": {
          "retraction_exponent": 0.5641419240718888,
          "retraction_r2": 0.989725048870059
        },
        "sign_gradient": {
          "retraction_exponent": 0.5032343722299224,
          "retraction_r2": 0.9770394190988188
        },
        "natural_gradient": {
          "retraction_exponent": 0.8239839919867032,
          "retraction_r2": 0.9917649580588547
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.6591562312954385,
          "retraction_r2": 0.9836283959234424
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.0045556737528047,
        "normalized_sgd": 0.0001518913421387457,
        "normalized_momentum": 0.00024314431490531527,
        "sign_gradient": 0.001710443575885428,
        "natural_gradient": 0.0014616498568336574,
        "wrong_fisher_natural_gradient": 0.0005153157367261742
      },
      "natural_retraction_cost_over_minimum": [
        4.09116852508514,
        2.9821901954989056,
        3.0063877529200482,
        2.2049550721234756
      ],
      "natural_actions": [
        0.36105476642516304,
        0.35959593766929904,
        0.36008979244992606,
        0.35956423543667754
      ],
      "natural_retraction_costs": [
        0.0022571178008958235,
        0.001198368137894883,
        0.000785019070431131,
        0.00038723795344024214
      ],
      "wrong_natural_actions": [
        0.468381362803881,
        0.4683368000580118,
        0.4681835624006567,
        0.4679424243055303
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.4678532167569915,
            0.46935246458381374,
            0.4729809446800186,
            0.47514555280368825
          ],
          "retraction_costs": [
            0.0008605534009312516,
            0.0005383378089045508,
            0.0003330911675085368,
            0.0002010513939272753
          ],
          "steps": [
            9,
            16,
            32,
            65
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.49022814225393024,
            0.4905482677025881,
            0.49053685373746964,
            0.49061137335743743
          ],
          "retraction_costs": [
            0.0005534660372727002,
            0.0004193499117384671,
            0.0002635787007933367,
            0.0001764008957027407
          ],
          "steps": [
            8,
            16,
            32,
            64
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.4900676028591411,
            0.49032418329806227,
            0.49039571099916074,
            0.4905149769271765
          ],
          "retraction_costs": [
            0.0005517049192806965,
            0.0004204221302050884,
            0.00026111703976596385,
            0.00017562169784589475
          ],
          "steps": [
            8,
            16,
            32,
            64
          ]
        },
        "sign_gradient": {
          "actions": [
            0.4874000911705418,
            0.4812939672795117,
            0.48583485518822195,
            0.485005281021029
          ],
          "retraction_costs": [
            0.0008834737498678977,
            0.0005334547132408788,
            0.000405642168899513,
            0.0003026098084338911
          ],
          "steps": [
            9,
            17,
            33,
            66
          ]
        },
        "natural_gradient": {
          "actions": [
            0.36105476642516304,
            0.35959593766929904,
            0.36008979244992606,
            0.35956423543667754
          ],
          "retraction_costs": [
            0.0022571178008958235,
            0.001198368137894883,
            0.000785019070431131,
            0.00038723795344024214
          ],
          "steps": [
            7,
            13,
            25,
            49
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.468381362803881,
            0.4683368000580118,
            0.4681835624006567,
            0.4679424243055303
          ],
          "retraction_costs": [
            0.0007427647464163975,
            0.00040184161952635015,
            0.00026615781794407645,
            0.00018581061495404926
          ],
          "steps": [
            8,
            16,
            31,
            62
          ]
        }
      }
    },
    {
      "seed": 73735,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "sign_gradient",
        "adam",
        "adam",
        "adam"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 1.0070696238773886,
          "retraction_r2": 0.977956693310732
        },
        "normalized_sgd": {
          "retraction_exponent": 0.7496452584604729,
          "retraction_r2": 0.9625807719790339
        },
        "normalized_momentum": {
          "retraction_exponent": 0.7492100749431073,
          "retraction_r2": 0.9617646593148995
        },
        "sign_gradient": {
          "retraction_exponent": 0.5791535039731812,
          "retraction_r2": 0.9709395921098054
        },
        "natural_gradient": {
          "retraction_exponent": 0.7246137797101193,
          "retraction_r2": 0.9899746504727786
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.7612104666077575,
          "retraction_r2": 0.9659307458714783
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.002331518488231396,
        "normalized_sgd": 0.0002028210167550852,
        "normalized_momentum": 9.138648257015943e-05,
        "sign_gradient": 0.011051083459820414,
        "natural_gradient": 0.0012515875016639898,
        "wrong_fisher_natural_gradient": 0.00017254930369497773
      },
      "natural_retraction_cost_over_minimum": [
        1.9671831822894648,
        2.024195498919558,
        2.1686088798639758,
        2.6591029567543676
      ],
      "natural_actions": [
        0.37102800709716066,
        0.3741566328948715,
        0.37476151023238746,
        0.37523114484350867
      ],
      "natural_retraction_costs": [
        0.0026431513461364,
        0.0013748841046918316,
        0.000926841714865971,
        0.0005650765190662236
      ],
      "wrong_natural_actions": [
        0.5151504181551554,
        0.5157849112684085,
        0.5161955449577929,
        0.5161064911421138
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.5454272324853878,
            0.5561801607594636,
            0.5572214361419898,
            0.5585236443448943
          ],
          "retraction_costs": [
            0.0018656759993318241,
            0.0006792249589655223,
            0.0004273899841838268,
            0.0002125064460670381
          ],
          "steps": [
            8,
            16,
            31,
            61
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.5243099531502788,
            0.5246571130297697,
            0.5245030245303055,
            0.5243966658653517
          ],
          "retraction_costs": [
            0.0014575741584104513,
            0.0006886416313833061,
            0.0005557742311969357,
            0.0002769767103549117
          ],
          "steps": [
            8,
            15,
            29,
            58
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.5240743186238741,
            0.5245613503540891,
            0.5244083284109072,
            0.5243604089575336
          ],
          "retraction_costs": [
            0.0014597285510481463,
            0.0006877512268243563,
            0.0005569766566522388,
            0.0002773455553749203
          ],
          "steps": [
            8,
            15,
            29,
            58
          ]
        },
        "sign_gradient": {
          "actions": [
            0.6180638149821888,
            0.5840225720082582,
            0.5897302369959543,
            0.5963202215328929
          ],
          "retraction_costs": [
            0.0013436223784000756,
            0.001059294943354483,
            0.0005775015569179531,
            0.0004314759091134395
          ],
          "steps": [
            9,
            17,
            33,
            66
          ]
        },
        "natural_gradient": {
          "actions": [
            0.37102800709716066,
            0.3741566328948715,
            0.37476151023238746,
            0.37523114484350867
          ],
          "retraction_costs": [
            0.0026431513461364,
            0.0013748841046918316,
            0.000926841714865971,
            0.0005650765190662236
          ],
          "steps": [
            6,
            11,
            22,
            43
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.5151504181551554,
            0.5157849112684085,
            0.5161955449577929,
            0.5161064911421138
          ],
          "retraction_costs": [
            0.001562978729478601,
            0.0006877858830971757,
            0.000511693890421959,
            0.0002971279214744092
          ],
          "steps": [
            8,
            15,
            29,
            57
          ]
        }
      }
    },
    {
      "seed": 73736,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_sgd",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.8371418348334283,
          "retraction_r2": 0.9975881237065669
        },
        "normalized_sgd": {
          "retraction_exponent": 0.759292723387351,
          "retraction_r2": 0.9984881835425541
        },
        "normalized_momentum": {
          "retraction_exponent": 0.7616013811810398,
          "retraction_r2": 0.998555070306629
        },
        "sign_gradient": {
          "retraction_exponent": 0.7314570189190099,
          "retraction_r2": 0.9980840218114062
        },
        "natural_gradient": {
          "retraction_exponent": 0.7253132494323205,
          "retraction_r2": 0.9657089523789156
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.8126004563047975,
          "retraction_r2": 0.9992296056634594
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.002936737746375711,
        "normalized_sgd": 0.00032190555476250605,
        "normalized_momentum": 0.00039033187198835446,
        "sign_gradient": 0.013087107666975142,
        "natural_gradient": 0.0022618029705107367,
        "wrong_fisher_natural_gradient": 0.00012601993920912475
      },
      "natural_retraction_cost_over_minimum": [
        1.5765612046777393,
        1.650112406259331,
        1.459550788245842,
        1.978231775445699
      ],
      "natural_actions": [
        0.2935093670012289,
        0.29702305325413786,
        0.2975637563979507,
        0.2982383126995346
      ],
      "natural_retraction_costs": [
        0.0013989996495888508,
        0.0008709451673177258,
        0.0004185824763415612,
        0.00033425972241843074
      ],
      "wrong_natural_actions": [
        0.44276074203749777,
        0.44272321336115716,
        0.44335654574008404,
        0.44330068101521114
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.4562247929120624,
            0.4586624494295372,
            0.4506666101974833,
            0.44934699591336397
          ],
          "retraction_costs": [
            0.0010914324309613493,
            0.0006171106413610018,
            0.00032205426500506314,
            0.0001959410832828216
          ],
          "steps": [
            6,
            12,
            24,
            47
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.4384516120578991,
            0.43876405659596474,
            0.438880669150372,
            0.4387394364886729
          ],
          "retraction_costs": [
            0.0008873741440788634,
            0.0005540869613794692,
            0.00030928150025400994,
            0.00018647265252571532
          ],
          "steps": [
            6,
            12,
            24,
            47
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.4385020263719085,
            0.43884330639735264,
            0.43896063991039413,
            0.43878936643558475
          ],
          "retraction_costs": [
            0.000887896411296244,
            0.0005533840024653594,
            0.0003089751181949766,
            0.00018557257616810458
          ],
          "steps": [
            6,
            12,
            24,
            47
          ]
        },
        "sign_gradient": {
          "actions": [
            0.48386266490500435,
            0.4770624422964166,
            0.46926159294301867,
            0.46319964926182405
          ],
          "retraction_costs": [
            0.0010250550424484946,
            0.0006575464489285476,
            0.00038592526634094465,
            0.0002259019173182215
          ],
          "steps": [
            7,
            13,
            25,
            49
          ]
        },
        "natural_gradient": {
          "actions": [
            0.2935093670012289,
            0.29702305325413786,
            0.2975637563979507,
            0.2982383126995346
          ],
          "retraction_costs": [
            0.0013989996495888508,
            0.0008709451673177258,
            0.0004185824763415612,
            0.00033425972241843074
          ],
          "steps": [
            4,
            8,
            16,
            32
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.44276074203749777,
            0.44272321336115716,
            0.44335654574008404,
            0.44330068101521114
          ],
          "retraction_costs": [
            0.0009013471638040967,
            0.0005278095989182256,
            0.00028678856516163695,
            0.00016896893810287798
          ],
          "steps": [
            6,
            12,
            24,
            48
          ]
        }
      }
    },
    {
      "seed": 73737,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_sgd",
        "normalized_momentum",
        "normalized_sgd",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.7345134600937462,
          "retraction_r2": 0.9991916184562895
        },
        "normalized_sgd": {
          "retraction_exponent": 0.6802009759664401,
          "retraction_r2": 0.9956993348597288
        },
        "normalized_momentum": {
          "retraction_exponent": 0.6791223343945189,
          "retraction_r2": 0.9958522533969179
        },
        "sign_gradient": {
          "retraction_exponent": 0.7580098875691349,
          "retraction_r2": 0.9496680842278913
        },
        "natural_gradient": {
          "retraction_exponent": 0.6448284198744325,
          "retraction_r2": 0.9923132894394925
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.7762025973057176,
          "retraction_r2": 0.9838021037690782
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.017402576444403774,
        "normalized_sgd": 0.0005037323506022033,
        "normalized_momentum": 0.0005015065863913284,
        "sign_gradient": 0.0026125181179379206,
        "natural_gradient": 0.0007675590492869916,
        "wrong_fisher_natural_gradient": 0.000775131167110627
      },
      "natural_retraction_cost_over_minimum": [
        1.5925836405239462,
        1.848645443145336,
        1.9797534291083552,
        1.8709231982023788
      ],
      "natural_actions": [
        0.39506078700870895,
        0.3949846922220541,
        0.39390671497426666,
        0.393604600201541
      ],
      "natural_retraction_costs": [
        0.00162997137893694,
        0.0011160331409201041,
        0.0007328352342415522,
        0.0004226957750514027
      ],
      "wrong_natural_actions": [
        0.4420435013562056,
        0.44302447544451606,
        0.4427691244993504,
        0.44311259488218285
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.47833963144665503,
            0.47148734961755084,
            0.4573376826054173,
            0.44951496408010977
          ],
          "retraction_costs": [
            0.001305251835134466,
            0.0007511495125680323,
            0.0004642219439179233,
            0.00028075399283174093
          ],
          "steps": [
            7,
            13,
            24,
            48
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.44983765986166013,
            0.4498313568201048,
            0.44950469086986133,
            0.4492783748180395
          ],
          "retraction_costs": [
            0.0010234761537552236,
            0.0006041643496814321,
            0.0003701649020866239,
            0.00025030029296180554
          ],
          "steps": [
            6,
            12,
            24,
            47
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.4498694131797853,
            0.4498468638530216,
            0.44965786818771897,
            0.4494324748414478
          ],
          "retraction_costs": [
            0.0010240335182917523,
            0.0006037031844360889,
            0.0003717500744637613,
            0.00025064035381860153
          ],
          "steps": [
            6,
            12,
            24,
            47
          ]
        },
        "sign_gradient": {
          "actions": [
            0.5299276072434437,
            0.5280889602142494,
            0.5179442226397746,
            0.5193009056644846
          ],
          "retraction_costs": [
            0.0012857969294213056,
            0.001081717360829451,
            0.0004743182349879449,
            0.00029370274122057435
          ],
          "steps": [
            7,
            14,
            28,
            55
          ]
        },
        "natural_gradient": {
          "actions": [
            0.39506078700870895,
            0.3949846922220541,
            0.39390671497426666,
            0.393604600201541
          ],
          "retraction_costs": [
            0.00162997137893694,
            0.0011160331409201041,
            0.0007328352342415522,
            0.0004226957750514027
          ],
          "steps": [
            6,
            11,
            21,
            42
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.4420435013562056,
            0.44302447544451606,
            0.4427691244993504,
            0.44311259488218285
          ],
          "retraction_costs": [
            0.0011366091187748186,
            0.0007720752969369181,
            0.00045287107086163387,
            0.00022592898279177756
          ],
          "steps": [
            6,
            12,
            23,
            46
          ]
        }
      }
    },
    {
      "seed": 73738,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_momentum",
        "normalized_momentum",
        "normalized_momentum",
        "normalized_sgd"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.5540377344051342,
          "retraction_r2": 0.958291493719138
        },
        "normalized_sgd": {
          "retraction_exponent": 0.5457918309519103,
          "retraction_r2": 0.9706330938762787
        },
        "normalized_momentum": {
          "retraction_exponent": 0.5434675789404787,
          "retraction_r2": 0.9713346546523806
        },
        "sign_gradient": {
          "retraction_exponent": 0.6167576486210605,
          "retraction_r2": 0.9885309292801796
        },
        "natural_gradient": {
          "retraction_exponent": 0.6434575722753754,
          "retraction_r2": 0.9977607139221607
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.5802900052895485,
          "retraction_r2": 0.9832113740296458
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.01744030416233858,
        "normalized_sgd": 0.0002892690300845235,
        "normalized_momentum": 0.0005269565033776987,
        "sign_gradient": 0.011768858681993228,
        "natural_gradient": 0.004464471096472817,
        "wrong_fisher_natural_gradient": 3.921773632773437e-05
      },
      "natural_retraction_cost_over_minimum": [
        3.38264227389472,
        3.702508910616248,
        3.1301326567809222,
        2.852166019630467
      ],
      "natural_actions": [
        0.2926007280520428,
        0.29388042229401606,
        0.29249840093042373,
        0.2911983542943362
      ],
      "natural_retraction_costs": [
        0.0018100048705867024,
        0.001125630314917662,
        0.000766187064001621,
        0.00046526455131402694
      ],
      "wrong_natural_actions": [
        0.46029039808021355,
        0.46012701398211914,
        0.46002367032936253,
        0.4600056299498579
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.43459262223672995,
            0.426255887732069,
            0.41310216978983455,
            0.4060210393669658
          ],
          "retraction_costs": [
            0.0005710846054903551,
            0.0003485308617554076,
            0.00029954185538326044,
            0.00016698922101703
          ],
          "steps": [
            6,
            12,
            23,
            45
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.44334371802967304,
            0.44312882450358954,
            0.44281633712843543,
            0.44268828111872643
          ],
          "retraction_costs": [
            0.0005370856484471648,
            0.00030482732386356864,
            0.00024752092621385934,
            0.00016312674231155297
          ],
          "steps": [
            7,
            13,
            25,
            49
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.44361435668531785,
            0.44348962778944867,
            0.44312955567748086,
            0.44289616866108383
          ],
          "retraction_costs": [
            0.0005350861025285691,
            0.0003040182595348059,
            0.00024477782510009654,
            0.00016385740043638273
          ],
          "steps": [
            7,
            13,
            25,
            49
          ]
        },
        "sign_gradient": {
          "actions": [
            0.41390426739838093,
            0.42554196162695546,
            0.40981034186627713,
            0.4146907790415427
          ],
          "retraction_costs": [
            0.0008166381001278795,
            0.0004827948874074702,
            0.0003126215366806706,
            0.00022702257918459393
          ],
          "steps": [
            6,
            12,
            23,
            46
          ]
        },
        "natural_gradient": {
          "actions": [
            0.2926007280520428,
            0.29388042229401606,
            0.29249840093042373,
            0.2911983542943362
          ],
          "retraction_costs": [
            0.0018100048705867024,
            0.001125630314917662,
            0.000766187064001621,
            0.00046526455131402694
          ],
          "steps": [
            5,
            9,
            17,
            34
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.46029039808021355,
            0.46012701398211914,
            0.46002367032936253,
            0.4600056299498579
          ],
          "retraction_costs": [
            0.0005699081391166806,
            0.000383518372346303,
            0.0002872942972030689,
            0.0001641883188950273
          ],
          "steps": [
            7,
            13,
            25,
            50
          ]
        }
      }
    },
    {
      "seed": 73739,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "adam",
        "adam",
        "wrong_fisher_natural_gradient",
        "adam"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.7769899083354853,
          "retraction_r2": 0.9959752916305815
        },
        "normalized_sgd": {
          "retraction_exponent": 0.8980949563704582,
          "retraction_r2": 0.9947139081792492
        },
        "normalized_momentum": {
          "retraction_exponent": 1.0125790297321962,
          "retraction_r2": 0.9926835869532152
        },
        "sign_gradient": {
          "retraction_exponent": 0.7307033765956344,
          "retraction_r2": 0.9786473639288882
        },
        "natural_gradient": {
          "retraction_exponent": 0.8361005812929109,
          "retraction_r2": 0.9959114808967848
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.8654598796198032,
          "retraction_r2": 0.9969432182269826
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.002504996965359068,
        "normalized_sgd": 0.011706909131641142,
        "normalized_momentum": 0.0987411857874094,
        "sign_gradient": 2.8696678128823617e-05,
        "natural_gradient": 0.0007653136024134735,
        "wrong_fisher_natural_gradient": 0.00012832616352877795
      },
      "natural_retraction_cost_over_minimum": [
        2.079504017347486,
        1.7132934639961666,
        1.755367531728919,
        1.8258021539103026
      ],
      "natural_actions": [
        1.0005275995870777,
        0.993766339296574,
        0.9971801517582667,
        0.9964175798307383
      ],
      "natural_retraction_costs": [
        0.004907910812434744,
        0.0026141530010940885,
        0.0014062381542348803,
        0.0008743450505019875
      ],
      "wrong_natural_actions": [
        1.22503118729419,
        1.223088516538861,
        1.2218251859444709,
        1.221981998206202
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            1.1056932926952796,
            1.1125275159053254,
            1.1052841643125033,
            1.102522349173583
          ],
          "retraction_costs": [
            0.0023601352877860924,
            0.0015258057396639539,
            0.0008368542563604832,
            0.0004788826919879633
          ],
          "steps": [
            12,
            24,
            48,
            94
          ]
        },
        "normalized_sgd": {
          "actions": [
            1.8015806629871782,
            1.7028963749896815,
            1.664305560254963,
            1.64504714283651
          ],
          "retraction_costs": [
            0.00326979653350407,
            0.0017175028402526187,
            0.0008456673316786164,
            0.0005198862080774732
          ],
          "steps": [
            18,
            34,
            66,
            130
          ]
        },
        "normalized_momentum": {
          "actions": [
            3.694588276572029,
            2.855829744395978,
            2.3251684399152714,
            2.1162112333569683
          ],
          "retraction_costs": [
            0.005227582257178677,
            0.0024806314484360293,
            0.0011106322946173475,
            0.0006585306422567203
          ],
          "steps": [
            32,
            51,
            85,
            158
          ]
        },
        "sign_gradient": {
          "actions": [
            1.1599555735511964,
            1.1295926469063458,
            1.1378473401762044,
            1.1378146886743132
          ],
          "retraction_costs": [
            0.002866355949877023,
            0.0018326311770690427,
            0.0009034263799225637,
            0.0006706749840892384
          ],
          "steps": [
            13,
            25,
            49,
            98
          ]
        },
        "natural_gradient": {
          "actions": [
            1.0005275995870777,
            0.993766339296574,
            0.9971801517582667,
            0.9964175798307383
          ],
          "retraction_costs": [
            0.004907910812434744,
            0.0026141530010940885,
            0.0014062381542348803,
            0.0008743450505019875
          ],
          "steps": [
            11,
            22,
            44,
            87
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            1.22503118729419,
            1.223088516538861,
            1.2218251859444709,
            1.221981998206202
          ],
          "retraction_costs": [
            0.0028810839788864255,
            0.0015432786975190529,
            0.0008011075337879984,
            0.0004853359777638689
          ],
          "steps": [
            13,
            26,
            51,
            102
          ]
        }
      }
    },
    {
      "seed": 73740,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_momentum",
        "adam",
        "adam",
        "wrong_fisher_natural_gradient"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.6891854958350261,
          "retraction_r2": 0.9883359003092352
        },
        "normalized_sgd": {
          "retraction_exponent": 0.6231261595063389,
          "retraction_r2": 0.9599245904525455
        },
        "normalized_momentum": {
          "retraction_exponent": 0.6228146412979636,
          "retraction_r2": 0.9613768765838434
        },
        "sign_gradient": {
          "retraction_exponent": 0.7190721765250727,
          "retraction_r2": 0.9813321100132669
        },
        "natural_gradient": {
          "retraction_exponent": 0.5729175788506744,
          "retraction_r2": 0.9860131903018454
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.6768743704749937,
          "retraction_r2": 0.9805697186014772
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.006321457862488438,
        "normalized_sgd": 0.00022608900844473803,
        "normalized_momentum": 0.00012275321408551087,
        "sign_gradient": 0.0025197784694715912,
        "natural_gradient": 0.001549900577477188,
        "wrong_fisher_natural_gradient": 0.00014647350292651819
      },
      "natural_retraction_cost_over_minimum": [
        1.8896057599547844,
        1.5537388956515303,
        1.9160997491603324,
        2.297849870390216
      ],
      "natural_actions": [
        0.43303127877860814,
        0.4318788963101295,
        0.4324930464011922,
        0.43316440816755364
      ],
      "natural_retraction_costs": [
        0.0013630760130800715,
        0.0008407147360189823,
        0.0005503257838842199,
        0.00041780960592205423
      ],
      "wrong_natural_actions": [
        0.5526011692727645,
        0.5520735163008003,
        0.5516225645216564,
        0.5517033744474881
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.5660085127745733,
            0.5577462996866671,
            0.5470456866008915,
            0.5436092834220813
          ],
          "retraction_costs": [
            0.0007847813467133356,
            0.0005410913882454135,
            0.0002872114482168175,
            0.00019719078117917462
          ],
          "steps": [
            7,
            14,
            26,
            52
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.5509986370216867,
            0.5504202087212863,
            0.550108398689544,
            0.5502328002777725
          ],
          "retraction_costs": [
            0.0007237046567120396,
            0.0005851477782356766,
            0.00036654852758691764,
            0.0002004505153638864
          ],
          "steps": [
            7,
            14,
            27,
            53
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.5509814826952161,
            0.5504451247048553,
            0.5501776035700693,
            0.5502451479305127
          ],
          "retraction_costs": [
            0.0007213547089910902,
            0.0005830664810896441,
            0.00036311304404826203,
            0.00020033399443522928
          ],
          "steps": [
            7,
            14,
            27,
            53
          ]
        },
        "sign_gradient": {
          "actions": [
            0.582478710700488,
            0.5665662350987958,
            0.5647879407236771,
            0.5633683772163861
          ],
          "retraction_costs": [
            0.001103830262852667,
            0.0005461748973659472,
            0.0003863807720510981,
            0.0002352159272581091
          ],
          "steps": [
            7,
            14,
            27,
            54
          ]
        },
        "natural_gradient": {
          "actions": [
            0.43303127877860814,
            0.4318788963101295,
            0.4324930464011922,
            0.43316440816755364
          ],
          "retraction_costs": [
            0.0013630760130800715,
            0.0008407147360189823,
            0.0005503257838842199,
            0.00041780960592205423
          ],
          "steps": [
            7,
            12,
            24,
            48
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.5526011692727645,
            0.5520735163008003,
            0.5516225645216564,
            0.5517033744474881
          ],
          "retraction_costs": [
            0.000723189878329932,
            0.0005463027492034907,
            0.0003152283041735393,
            0.00018182632873709137
          ],
          "steps": [
            7,
            14,
            27,
            54
          ]
        }
      }
    },
    {
      "seed": 73741,
      "step_radii": [
        0.08,
        0.04,
        0.02,
        0.01
      ],
      "action_winners": [
        "natural_gradient",
        "natural_gradient",
        "natural_gradient",
        "natural_gradient"
      ],
      "retraction_cost_winners": [
        "normalized_momentum",
        "wrong_fisher_natural_gradient",
        "normalized_momentum",
        "normalized_momentum"
      ],
      "wrong_metric_action_winners": [
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient",
        "wrong_fisher_natural_gradient"
      ],
      "fits": {
        "adam": {
          "retraction_exponent": 0.6983871306679764,
          "retraction_r2": 0.9973056980242171
        },
        "normalized_sgd": {
          "retraction_exponent": 0.6587833590001152,
          "retraction_r2": 0.9995283720284728
        },
        "normalized_momentum": {
          "retraction_exponent": 0.6693476655833779,
          "retraction_r2": 0.997665316272078
        },
        "sign_gradient": {
          "retraction_exponent": 0.6961651170573376,
          "retraction_r2": 0.9936469475043712
        },
        "natural_gradient": {
          "retraction_exponent": 0.7223916629620218,
          "retraction_r2": 0.99459775118049
        },
        "wrong_fisher_natural_gradient": {
          "retraction_exponent": 0.6469096733812527,
          "retraction_r2": 0.9994057281324662
        }
      },
      "small_step_action_relative_changes": {
        "adam": 0.004597907484832405,
        "normalized_sgd": 5.588549114649164e-05,
        "normalized_momentum": 0.000207997770112928,
        "sign_gradient": 0.001141708316051326,
        "natural_gradient": 0.000579863487305565,
        "wrong_fisher_natural_gradient": 0.0003995374240889734
      },
      "natural_retraction_cost_over_minimum": [
        2.967615320782806,
        2.489520128319899,
        2.6245582272613395,
        2.5552974885337374
      ],
      "natural_actions": [
        0.576845789083666,
        0.5830933938371943,
        0.5857412565282226,
        0.5854018033970299
      ],
      "natural_retraction_costs": [
        0.003774302188523745,
        0.0020531216383572293,
        0.0013139245962012754,
        0.0008252386327753006
      ],
      "wrong_natural_actions": [
        0.7467661426530876,
        0.747643496173505,
        0.7470952303688576,
        0.7473938421793419
      ],
      "all_algorithms": {
        "adam": {
          "actions": [
            0.7521702052193882,
            0.7392246924941629,
            0.7348655233250604,
            0.731502144141342
          ],
          "retraction_costs": [
            0.0016378904941824765,
            0.0010335460456316699,
            0.0005943455331074264,
            0.0003922806517953212
          ],
          "steps": [
            11,
            21,
            41,
            80
          ]
        },
        "normalized_sgd": {
          "actions": [
            0.7581256948449981,
            0.7586991681065424,
            0.7587913858824767,
            0.7588337936817352
          ],
          "retraction_costs": [
            0.0012811741092906015,
            0.0008360203094114714,
            0.0005192981366231627,
            0.00032771501182244316
          ],
          "steps": [
            11,
            21,
            42,
            83
          ]
        },
        "normalized_momentum": {
          "actions": [
            0.758797724196569,
            0.7594433279090628,
            0.7592822574117866,
            0.7591243612374108
          ],
          "retraction_costs": [
            0.0012718299983463317,
            0.0008483567782058625,
            0.0005006269560162598,
            0.0003229520775871905
          ],
          "steps": [
            11,
            21,
            42,
            83
          ]
        },
        "sign_gradient": {
          "actions": [
            0.7694144466048234,
            0.7684383480765762,
            0.7674873657756129,
            0.7666121183449128
          ],
          "retraction_costs": [
            0.0017320359996551484,
            0.0009595428281716959,
            0.0006564803096585288,
            0.00039350403828458094
          ],
          "steps": [
            11,
            21,
            42,
            84
          ]
        },
        "natural_gradient": {
          "actions": [
            0.576845789083666,
            0.5830933938371943,
            0.5857412565282226,
            0.5854018033970299
          ],
          "retraction_costs": [
            0.003774302188523745,
            0.0020531216383572293,
            0.0013139245962012754,
            0.0008252386327753006
          ],
          "steps": [
            9,
            17,
            33,
            66
          ]
        },
        "wrong_fisher_natural_gradient": {
          "actions": [
            0.7467661426530876,
            0.747643496173505,
            0.7470952303688576,
            0.7473938421793419
          ],
          "retraction_costs": [
            0.0012798388237898494,
            0.0008247057796406805,
            0.0005360281028583229,
            0.0003314324694019603
          ],
          "steps": [
            11,
            21,
            41,
            82
          ]
        }
      }
    }
  ],
  "claim_boundary": "Sixteen-seed, four-radius prospective confirmation testing whether the higher finite-step retraction cost of moving-fibre natural gradient vanishes as a discretization effect while its restricted action ordering converges. Six algorithms are generated causally in a transported eight-dimensional chart projected into the current response kernel at every step, with a common 20% capability endpoint. No fitted lambda or scalar composite is used. A pass prospectively confirms the restricted six-algorithm Moving-Fibre F16 scaling result in this frozen CNN--MNIST model; it is not complete-kernel optimization, continuum geometry, local stationarity, GPT-2 transfer, global variation, or a universal learning law."
}
```
