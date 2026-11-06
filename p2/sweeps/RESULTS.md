# Phase 2 tuning sweep (dev strict macro-F1)

## One change at a time (dev split 0)
```
variant                base             base F1      F1   delta   span     train ep  min
a6_e2_invsqrt          E2_a4_lorma       0.4857  0.5007 +0.0150  0.605     2.23M  8  5.0
c1_a4_lora             E2_a4_lorma       0.4857  0.4974 +0.0117  0.605     2.19M 13  5.8
a5_e2_focal            E2_a4_lorma       0.4857  0.4945 +0.0088  0.589     2.23M 15  5.8
c7_e2_r32              E2_a4_lorma       0.4857  0.4902 +0.0045  0.605     4.44M 15  5.8
a7_e2_casenorm         E2_a4_lorma       0.4857  0.4853 -0.0004  0.580     2.23M 13  5.7
c10_e2_causal          E2_a4_lorma       0.4857  0.4803 -0.0054  0.562     2.23M 13  5.6
c22_e2_lr2e-3          E2_a4_lorma       0.4857  0.4803 -0.0054  0.584     2.23M 15  5.4
c20_e2_ep25            E2_a4_lorma       0.4857  0.4793 -0.0064  0.586     2.23M 20  8.4
b1_e2_mlphead          E2_a4_lorma       0.4857  0.4782 -0.0075  0.568     2.82M 15  5.9
a9_e2_nopresplit       E2_a4_lorma       0.4857  0.4763 -0.0094  0.555     2.23M 15  5.6
a3_e2_oweight0.5       E2_a4_lorma       0.4857  0.4732 -0.0125  0.549     2.23M 12  5.9
a1_e2_win1024          E2_a4_lorma       0.4857  0.4716 -0.0141  0.570     2.23M 10  5.8
c19_a4_full            E5_a4_partial     0.4697  0.4627 -0.0070  0.603    87.53M  6  3.4
c17_e5_top6            E5_a4_partial     0.4697  0.4618 -0.0079  0.572    37.78M 14  3.8
c6_e2_r8               E2_a4_lorma       0.4857  0.4557 -0.0300  0.561     1.13M  8  5.0
c21_e2_lr5e-4          E2_a4_lorma       0.4857  0.4526 -0.0331  0.542     2.23M 12  5.4
c4_a3_lora             E1_a3_full        0.4267  0.4510 +0.0243  0.576     1.79M 14  5.3
c3_a3_lorma            E1_a3_full        0.4267  0.4384 +0.0117  0.559     1.79M 12  5.5
c8_a4_lorma_pi         E2_a4_lorma       0.4857  0.4366 -0.0491  0.577     2.23M 12  5.4
c11_e1_llrd0.8         E1_a3_full        0.4267  0.4343 +0.0076  0.563    81.30M  7  2.6
a4_e1_oweight0.5       E1_a3_full        0.4267  0.4317 +0.0050  0.581    81.30M 12  4.0
b2_e2_crf              E2_a4_lorma       0.4857  0.4303 -0.0554  0.580     2.23M 11 10.9
c15_a2_full_dec        E4_a2_adapter     0.3892  0.4296 +0.0404  0.513    89.43M 12  4.5
c5_e2_attnonly         E2_a4_lorma       0.4857  0.4274 -0.0583  0.549     0.67M 14  5.2
b3_e1_crf              E1_a3_full        0.4267  0.4263 -0.0004  0.604    81.30M  6  6.6
c12_e1_regular         E1_a3_full        0.4267  0.4224 -0.0043  0.575    81.30M  9  3.0
a2_e1_win1024          E1_a3_full        0.4267  0.4162 -0.0105  0.560    81.30M 14  4.0
c9_e3_causal           E3_a1_lora        0.3564  0.4161 +0.0597  0.537     1.79M  6  3.0
c18_e5_top2            E5_a4_partial     0.4697  0.4161 -0.0536  0.522    12.61M  6  1.9
a8_e1_casenorm         E1_a3_full        0.4267  0.4155 -0.0112  0.613    81.30M  3  1.8
c16_a2_lorma_dec       E4_a2_adapter     0.3892  0.3996 +0.0104  0.488     2.72M 12  6.0
d2_e6_a0.3_T4          E6_kd_a3_small    0.3443  0.3631 +0.0188  0.490    52.95M 12  2.5
d3_e6_teacherE2        E6_kd_a3_small    0.3443  0.3579 +0.0136  0.519    52.95M 12  2.6
c2_a1_lorma            E3_a1_lora        0.3564  0.3464 -0.0100  0.490     1.79M 14  4.0
c14_a3_bitfit          E1_a3_full        0.4267  0.3322 -0.0945  0.475     0.10M  9  2.8
c13_a4_ia3             E2_a4_lorma       0.4857  0.3257 -0.1600  0.433     0.04M 10  4.2
d1_e6_noKD             E6_kd_a3_small    0.3443  0.3024 -0.0419  0.447    52.95M  3  1.4
```

## Inference-only (no retraining)
```

== runs/E2_a4_lorma  (window 512)
  stride  64  viterbi                      macro-F1 0.4831  P 0.5138  R 0.4764  span 0.5698
  stride  64  argmax                       macro-F1 0.3677  P 0.3594  R 0.4005  span 0.4480
  stride  64  viterbi+merge_space          macro-F1 0.4844  P 0.5192  R 0.4730  span 0.5722
  stride  64  viterbi+merge_space_dash     macro-F1 0.4844  P 0.5192  R 0.4730  span 0.5722
  stride  64  viterbi+propagate5           macro-F1 0.5181  P 0.5062  R 0.5523  span 0.6131
  stride  64  viterbi+propagate10          macro-F1 0.4867  P 0.4937  R 0.4986  span 0.5832
  stride  64  viterbi+merge+propagate10    macro-F1 0.4889  P 0.4993  R 0.4970  span 0.5864
  stride 128  viterbi                      macro-F1 0.4857  P 0.5175  R 0.4781  span 0.5728
  stride 128  argmax                       macro-F1 0.3730  P 0.3643  R 0.4057  span 0.4509
  stride 128  viterbi+merge_space          macro-F1 0.4874  P 0.5233  R 0.4750  span 0.5760
  stride 128  viterbi+merge_space_dash     macro-F1 0.4874  P 0.5233  R 0.4750  span 0.5760
  stride 128  viterbi+propagate5           macro-F1 0.5142  P 0.4977  R 0.5536  span 0.6064
  stride 128  viterbi+propagate10          macro-F1 0.4906  P 0.5030  R 0.5016  span 0.5841
  stride 128  viterbi+merge+propagate10    macro-F1 0.4932  P 0.5088  R 0.5004  span 0.5880
  stride 256  viterbi                      macro-F1 0.4816  P 0.5135  R 0.4752  span 0.5762
  stride 256  argmax                       macro-F1 0.3740  P 0.3668  R 0.4056  span 0.4538
  stride 256  viterbi+merge_space          macro-F1 0.4838  P 0.5201  R 0.4721  span 0.5808
  stride 256  viterbi+merge_space_dash     macro-F1 0.4838  P 0.5201  R 0.4721  span 0.5808
  stride 256  viterbi+propagate5           macro-F1 0.5111  P 0.4984  R 0.5505  span 0.6174
  stride 256  viterbi+propagate10          macro-F1 0.4855  P 0.4957  R 0.4977  span 0.5889
  stride 256  viterbi+merge+propagate10    macro-F1 0.4885  P 0.5022  R 0.4965  span 0.5942

== runs/E1_a3_full  (window 512)
  stride  64  viterbi                      macro-F1 0.4261  P 0.4461  R 0.4413  span 0.5598
  stride  64  argmax                       macro-F1 0.3407  P 0.3278  R 0.3936  span 0.4508
  stride  64  viterbi+merge_space          macro-F1 0.4346  P 0.4576  R 0.4453  span 0.5690
  stride  64  viterbi+merge_space_dash     macro-F1 0.4346  P 0.4576  R 0.4453  span 0.5690
  stride  64  viterbi+propagate5           macro-F1 0.4863  P 0.4560  R 0.5584  span 0.6602
  stride  64  viterbi+propagate10          macro-F1 0.4509  P 0.4371  R 0.5021  span 0.6659
  stride  64  viterbi+merge+propagate10    macro-F1 0.4593  P 0.4481  R 0.5061  span 0.6754
  stride 128  viterbi                      macro-F1 0.4267  P 0.4468  R 0.4408  span 0.5600
  stride 128  argmax                       macro-F1 0.3403  P 0.3272  R 0.3926  span 0.4499
  stride 128  viterbi+merge_space          macro-F1 0.4354  P 0.4585  R 0.4448  span 0.5693
  stride 128  viterbi+merge_space_dash     macro-F1 0.4354  P 0.4585  R 0.4448  span 0.5693
  stride 128  viterbi+propagate5           macro-F1 0.4864  P 0.4562  R 0.5579  span 0.6667
  stride 128  viterbi+propagate10          macro-F1 0.4511  P 0.4371  R 0.5016  span 0.6728
  stride 128  viterbi+merge+propagate10    macro-F1 0.4597  P 0.4484  R 0.5056  span 0.6824
  stride 256  viterbi                      macro-F1 0.4259  P 0.4461  R 0.4403  span 0.5631
  stride 256  argmax                       macro-F1 0.3455  P 0.3332  R 0.3969  span 0.4553
  stride 256  viterbi+merge_space          macro-F1 0.4333  P 0.4562  R 0.4434  span 0.5713
  stride 256  viterbi+merge_space_dash     macro-F1 0.4333  P 0.4562  R 0.4434  span 0.5713
  stride 256  viterbi+propagate5           macro-F1 0.4848  P 0.4542  R 0.5574  span 0.6720
  stride 256  viterbi+propagate10          macro-F1 0.4506  P 0.4362  R 0.5011  span 0.6782
  stride 256  viterbi+merge+propagate10    macro-F1 0.4583  P 0.4466  R 0.5042  span 0.6868

== runs/E5_a4_partial  (window 512)
  stride  64  viterbi                      macro-F1 0.4649  P 0.4825  R 0.4682  span 0.5608
  stride  64  argmax                       macro-F1 0.3618  P 0.3394  R 0.4109  span 0.4406
  stride  64  viterbi+merge_space          macro-F1 0.4776  P 0.4995  R 0.4747  span 0.5682
  stride  64  viterbi+merge_space_dash     macro-F1 0.4776  P 0.4995  R 0.4747  span 0.5682
  stride  64  viterbi+propagate5           macro-F1 0.4909  P 0.4680  R 0.5288  span 0.6067
  stride  64  viterbi+propagate10          macro-F1 0.4927  P 0.4798  R 0.5175  span 0.6186
  stride  64  viterbi+merge+propagate10    macro-F1 0.5059  P 0.4979  R 0.5240  span 0.6262
  stride 128  viterbi                      macro-F1 0.4697  P 0.4879  R 0.4740  span 0.5591
  stride 128  argmax                       macro-F1 0.3619  P 0.3394  R 0.4132  span 0.4410
  stride 128  viterbi+merge_space          macro-F1 0.4811  P 0.5035  R 0.4794  span 0.5670
  stride 128  viterbi+merge_space_dash     macro-F1 0.4811  P 0.5035  R 0.4794  span 0.5670
  stride 128  viterbi+propagate5           macro-F1 0.4918  P 0.4652  R 0.5349  span 0.6011
  stride 128  viterbi+propagate10          macro-F1 0.4909  P 0.4721  R 0.5236  span 0.6124
  stride 128  viterbi+merge+propagate10    macro-F1 0.5025  P 0.4884  R 0.5290  span 0.6202
  stride 256  viterbi                      macro-F1 0.4649  P 0.4809  R 0.4715  span 0.5581
  stride 256  argmax                       macro-F1 0.3628  P 0.3412  R 0.4115  span 0.4381
  stride 256  viterbi+merge_space          macro-F1 0.4782  P 0.4987  R 0.4783  span 0.5675
  stride 256  viterbi+merge_space_dash     macro-F1 0.4782  P 0.4987  R 0.4783  span 0.5675
  stride 256  viterbi+propagate5           macro-F1 0.4978  P 0.4771  R 0.5350  span 0.6062
  stride 256  viterbi+propagate10          macro-F1 0.4982  P 0.4858  R 0.5236  span 0.6197
  stride 256  viterbi+merge+propagate10    macro-F1 0.5114  P 0.5035  R 0.5305  span 0.6287
```

## Confirmation: mean +- std over dev splits 0,1,2
```

candidate                   none           merge           prop5          prop10     merge+prop5    merge+prop10   (mean +- std over dev splits)
E2invsqrt_r32      0.4974+-0.012   0.4958+-0.013   0.5077+-0.013   0.5048+-0.023   0.5076+-0.007   0.5022+-0.018   n=3
A4lora             0.4902+-0.018   0.4854+-0.014   0.5172+-0.043   0.5131+-0.022   0.5131+-0.042   0.5084+-0.021   n=3
A4lora_invsqrt     0.4896+-0.008   0.4902+-0.006   0.4805+-0.016   0.4886+-0.018   0.4811+-0.013   0.4892+-0.018   n=3
E2                 0.4880+-0.007   0.4859+-0.009   0.5061+-0.027   0.5180+-0.016   0.5043+-0.032   0.5160+-0.018   n=3
E2invsqrt          0.4878+-0.017   0.4851+-0.015   0.4898+-0.025   0.5047+-0.023   0.4862+-0.023   0.5009+-0.021   n=3
E2focal            0.4857+-0.014   0.4850+-0.009   0.4982+-0.029   0.4930+-0.015   0.4983+-0.024   0.4921+-0.013   n=3
```
