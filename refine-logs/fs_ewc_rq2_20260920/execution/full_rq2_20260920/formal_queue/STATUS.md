# RQ2 自动执行状态

更新时间：2026-09-21T09:57:36.851537+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| anchor_sgd_seed0 | 无 | complete |
| anchor_gam_seed0 | 无 | complete |
| anchor_sgd_seed1 | 无 | complete |
| anchor_gam_seed1 | 无 | complete |
| anchor_sgd_seed2 | 无 | complete |
| anchor_gam_seed2 | 无 | complete |
| B3_A_sgd_lam0_seed0 | anchor_sgd_seed0 | complete |
| B3_B_gam_lam0_seed0 | anchor_gam_seed0 | complete |
| B3_D_gam_lam2000_seed0 | anchor_gam_seed0 | complete |
| B3_A_sgd_lam0_seed1 | anchor_sgd_seed1 | complete |
| B3_B_gam_lam0_seed1 | anchor_gam_seed1 | complete |
| B3_D_gam_lam2000_seed1 | anchor_gam_seed1 | complete |
| B3_A_sgd_lam0_seed2 | anchor_sgd_seed2 | complete |
| B3_B_gam_lam0_seed2 | anchor_gam_seed2 | complete |
| B3_D_gam_lam2000_seed2 | anchor_gam_seed2 | complete |
| sequence_sgd_seed0 | anchor_sgd_seed0 | complete |
| sequence_gam_seed0 | anchor_gam_seed0 | complete |
| sequence_sgd_seed1 | anchor_sgd_seed1 | complete |
| sequence_gam_seed1 | anchor_gam_seed1 | complete |
| sequence_sgd_seed2 | anchor_sgd_seed2 | complete |
| sequence_gam_seed2 | anchor_gam_seed2 | complete |
| endpoint_A_seed0 | B3_A_sgd_lam0_seed0 | complete |
| endpoint_C_seed0 | sequence_sgd_seed0 | complete |
| endpoint_B_seed0 | B3_B_gam_lam0_seed0 | complete |
| endpoint_D_seed0 | B3_D_gam_lam2000_seed0 | complete |
| endpoint_A_seed1 | B3_A_sgd_lam0_seed1 | complete |
| endpoint_C_seed1 | sequence_sgd_seed1 | complete |
| endpoint_B_seed1 | B3_B_gam_lam0_seed1 | complete |
| endpoint_D_seed1 | B3_D_gam_lam2000_seed1 | complete |
| endpoint_A_seed2 | B3_A_sgd_lam0_seed2 | complete |
| endpoint_C_seed2 | sequence_sgd_seed2 | complete |
| endpoint_B_seed2 | B3_B_gam_lam0_seed2 | complete |
| endpoint_D_seed2 | B3_D_gam_lam2000_seed2 | complete |
| sgd_seed0_sparse_train | anchor_sgd_seed0 | complete |
| sgd_seed0_sparse_test | anchor_sgd_seed0 | complete |
| sgd_seed0_sparse_train_full | anchor_sgd_seed0 | complete |
| gam_seed0_sparse_train | anchor_gam_seed0 | complete |
| gam_seed0_sparse_test | anchor_gam_seed0 | complete |
| gam_seed0_sparse_train_full | anchor_gam_seed0 | complete |
| sgd_seed1_sparse_train | anchor_sgd_seed1 | complete |
| sgd_seed1_sparse_test | anchor_sgd_seed1 | complete |
| sgd_seed1_sparse_train_full | anchor_sgd_seed1 | complete |
| gam_seed1_sparse_train | anchor_gam_seed1 | complete |
| gam_seed1_sparse_test | anchor_gam_seed1 | complete |
| gam_seed1_sparse_train_full | anchor_gam_seed1 | complete |
| sgd_seed2_sparse_train | anchor_sgd_seed2 | complete |
| sgd_seed2_sparse_test | anchor_sgd_seed2 | complete |
| sgd_seed2_sparse_train_full | anchor_sgd_seed2 | complete |
| gam_seed2_sparse_train | anchor_gam_seed2 | complete |
| gam_seed2_sparse_test | anchor_gam_seed2 | complete |
| gam_seed2_sparse_train_full | anchor_gam_seed2 | complete |
| sgd_seed0_trajectory_fisher | sequence_sgd_seed0 | complete |
| gam_seed0_trajectory_fisher | sequence_gam_seed0 | complete |
| sgd_seed1_trajectory_fisher | sequence_sgd_seed1 | complete |
| gam_seed1_trajectory_fisher | sequence_gam_seed1 | complete |
| sgd_seed2_trajectory_fisher | sequence_sgd_seed2 | complete |
| gam_seed2_trajectory_fisher | sequence_gam_seed2 | complete |
| sgd_seed0_path_t01 | sequence_sgd_seed0 | complete |
| sgd_seed0_path_t05 | sequence_sgd_seed0 | complete |
| sgd_seed0_path_t09 | sequence_sgd_seed0 | complete |
| gam_seed0_path_t01 | sequence_gam_seed0 | complete |
| gam_seed0_path_t05 | sequence_gam_seed0 | complete |
| gam_seed0_path_t09 | sequence_gam_seed0 | complete |
| sgd_seed1_path_t01 | sequence_sgd_seed1 | complete |
| sgd_seed1_path_t05 | sequence_sgd_seed1 | complete |
| sgd_seed1_path_t09 | sequence_sgd_seed1 | complete |
| gam_seed1_path_t01 | sequence_gam_seed1 | complete |
| gam_seed1_path_t05 | sequence_gam_seed1 | complete |
| gam_seed1_path_t09 | sequence_gam_seed1 | complete |
| sgd_seed2_path_t01 | sequence_sgd_seed2 | complete |
| sgd_seed2_path_t05 | sequence_sgd_seed2 | complete |
| sgd_seed2_path_t09 | sequence_sgd_seed2 | complete |
| gam_seed2_path_t01 | sequence_gam_seed2 | complete |
| gam_seed2_path_t05 | sequence_gam_seed2 | complete |
| gam_seed2_path_t09 | sequence_gam_seed2 | complete |
| sgd_seed0_bridge_full | sequence_sgd_seed0, sequence_gam_seed0 | complete |
| gam_seed0_bridge_full | sequence_sgd_seed0, sequence_gam_seed0 | complete |
| sgd_seed1_bridge_full | sequence_sgd_seed1, sequence_gam_seed1 | complete |
| gam_seed1_bridge_full | sequence_sgd_seed1, sequence_gam_seed1 | complete |
| sgd_seed2_bridge_full | sequence_sgd_seed2, sequence_gam_seed2 | complete |
| gam_seed2_bridge_full | sequence_sgd_seed2, sequence_gam_seed2 | complete |
| sgd_seed0_dense | anchor_sgd_seed0 | complete |
| gam_seed0_dense | anchor_gam_seed0 | complete |
| sgd_seed1_dense | anchor_sgd_seed1 | complete |
| gam_seed1_dense | anchor_gam_seed1 | complete |
| sgd_seed2_dense | anchor_sgd_seed2 | complete |
| gam_seed2_dense | anchor_gam_seed2 | complete |
| final_report | anchor_sgd_seed0, anchor_gam_seed0, anchor_sgd_seed1, anchor_gam_seed1, anchor_sgd_seed2, anchor_gam_seed2, B3_A_sgd_lam0_seed0, B3_B_gam_lam0_seed0, B3_D_gam_lam2000_seed0, B3_A_sgd_lam0_seed1, B3_B_gam_lam0_seed1, B3_D_gam_lam2000_seed1, B3_A_sgd_lam0_seed2, B3_B_gam_lam0_seed2, B3_D_gam_lam2000_seed2, sequence_sgd_seed0, sequence_gam_seed0, sequence_sgd_seed1, sequence_gam_seed1, sequence_sgd_seed2, sequence_gam_seed2, endpoint_A_seed0, endpoint_C_seed0, endpoint_B_seed0, endpoint_D_seed0, endpoint_A_seed1, endpoint_C_seed1, endpoint_B_seed1, endpoint_D_seed1, endpoint_A_seed2, endpoint_C_seed2, endpoint_B_seed2, endpoint_D_seed2, sgd_seed0_sparse_train, sgd_seed0_sparse_test, sgd_seed0_sparse_train_full, gam_seed0_sparse_train, gam_seed0_sparse_test, gam_seed0_sparse_train_full, sgd_seed1_sparse_train, sgd_seed1_sparse_test, sgd_seed1_sparse_train_full, gam_seed1_sparse_train, gam_seed1_sparse_test, gam_seed1_sparse_train_full, sgd_seed2_sparse_train, sgd_seed2_sparse_test, sgd_seed2_sparse_train_full, gam_seed2_sparse_train, gam_seed2_sparse_test, gam_seed2_sparse_train_full, sgd_seed0_trajectory_fisher, gam_seed0_trajectory_fisher, sgd_seed1_trajectory_fisher, gam_seed1_trajectory_fisher, sgd_seed2_trajectory_fisher, gam_seed2_trajectory_fisher, sgd_seed0_path_t01, sgd_seed0_path_t05, sgd_seed0_path_t09, gam_seed0_path_t01, gam_seed0_path_t05, gam_seed0_path_t09, sgd_seed1_path_t01, sgd_seed1_path_t05, sgd_seed1_path_t09, gam_seed1_path_t01, gam_seed1_path_t05, gam_seed1_path_t09, sgd_seed2_path_t01, sgd_seed2_path_t05, sgd_seed2_path_t09, gam_seed2_path_t01, gam_seed2_path_t05, gam_seed2_path_t09, sgd_seed0_bridge_full, gam_seed0_bridge_full, sgd_seed1_bridge_full, gam_seed1_bridge_full, sgd_seed2_bridge_full, gam_seed2_bridge_full, sgd_seed0_dense, gam_seed0_dense, sgd_seed1_dense, gam_seed1_dense, sgd_seed2_dense, gam_seed2_dense | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| anchor_sgd_seed0 | 4 | complete | [anchor_sgd_seed0.attempt1.log](anchor_sgd_seed0.attempt1.log) |
| anchor_gam_seed0 | 5 | complete | [anchor_gam_seed0.attempt1.log](anchor_gam_seed0.attempt1.log) |
| anchor_sgd_seed1 | 6 | complete | [anchor_sgd_seed1.attempt1.log](anchor_sgd_seed1.attempt1.log) |
| anchor_gam_seed1 | 2 | complete | [anchor_gam_seed1.attempt1.log](anchor_gam_seed1.attempt1.log) |
| anchor_sgd_seed2 | 3 | complete | [anchor_sgd_seed2.attempt1.log](anchor_sgd_seed2.attempt1.log) |
| anchor_gam_seed2 | 3 | complete | [anchor_gam_seed2.attempt1.log](anchor_gam_seed2.attempt1.log) |
| B3_A_sgd_lam0_seed0 | 4 | complete | [B3_A_sgd_lam0_seed0.attempt1.log](B3_A_sgd_lam0_seed0.attempt1.log) |
| B3_B_gam_lam0_seed0 | 4 | complete | [B3_B_gam_lam0_seed0.attempt1.log](B3_B_gam_lam0_seed0.attempt1.log) |
| B3_D_gam_lam2000_seed0 | 5 | complete | [B3_D_gam_lam2000_seed0.attempt1.log](B3_D_gam_lam2000_seed0.attempt1.log) |
| B3_A_sgd_lam0_seed1 | 6 | complete | [B3_A_sgd_lam0_seed1.attempt1.log](B3_A_sgd_lam0_seed1.attempt1.log) |
| B3_B_gam_lam0_seed1 | 2 | complete | [B3_B_gam_lam0_seed1.attempt1.log](B3_B_gam_lam0_seed1.attempt1.log) |
| B3_D_gam_lam2000_seed1 | 2 | complete | [B3_D_gam_lam2000_seed1.attempt1.log](B3_D_gam_lam2000_seed1.attempt1.log) |
| B3_A_sgd_lam0_seed2 | 3 | complete | [B3_A_sgd_lam0_seed2.attempt1.log](B3_A_sgd_lam0_seed2.attempt1.log) |
| B3_B_gam_lam0_seed2 | 3 | complete | [B3_B_gam_lam0_seed2.attempt1.log](B3_B_gam_lam0_seed2.attempt1.log) |
| B3_D_gam_lam2000_seed2 | 4 | complete | [B3_D_gam_lam2000_seed2.attempt1.log](B3_D_gam_lam2000_seed2.attempt1.log) |
| sequence_sgd_seed0 | 4 | complete | [sequence_sgd_seed0.attempt1.log](sequence_sgd_seed0.attempt1.log) |
| sequence_gam_seed0 | 5 | complete | [sequence_gam_seed0.attempt1.log](sequence_gam_seed0.attempt1.log) |
| sequence_sgd_seed1 | 6 | complete | [sequence_sgd_seed1.attempt1.log](sequence_sgd_seed1.attempt1.log) |
| sequence_gam_seed1 | 2 | complete | [sequence_gam_seed1.attempt1.log](sequence_gam_seed1.attempt1.log) |
| sequence_sgd_seed2 | 3 | complete | [sequence_sgd_seed2.attempt1.log](sequence_sgd_seed2.attempt1.log) |
| sequence_gam_seed2 | 6 | complete | [sequence_gam_seed2.attempt1.log](sequence_gam_seed2.attempt1.log) |
| endpoint_A_seed0 | 1 | complete | [endpoint_A_seed0.attempt1.log](endpoint_A_seed0.attempt1.log) |
| endpoint_C_seed0 | 4 | complete | [endpoint_C_seed0.attempt1.log](endpoint_C_seed0.attempt1.log) |
| endpoint_B_seed0 | 1 | complete | [endpoint_B_seed0.attempt1.log](endpoint_B_seed0.attempt1.log) |
| endpoint_D_seed0 | 1 | complete | [endpoint_D_seed0.attempt1.log](endpoint_D_seed0.attempt1.log) |
| endpoint_A_seed1 | 1 | complete | [endpoint_A_seed1.attempt1.log](endpoint_A_seed1.attempt1.log) |
| endpoint_C_seed1 | 1 | complete | [endpoint_C_seed1.attempt1.log](endpoint_C_seed1.attempt1.log) |
| endpoint_B_seed1 | 1 | complete | [endpoint_B_seed1.attempt1.log](endpoint_B_seed1.attempt1.log) |
| endpoint_D_seed1 | 0 | complete | [endpoint_D_seed1.attempt1.log](endpoint_D_seed1.attempt1.log) |
| endpoint_A_seed2 | 0 | complete | [endpoint_A_seed2.attempt1.log](endpoint_A_seed2.attempt1.log) |
| endpoint_C_seed2 | 3 | complete | [endpoint_C_seed2.attempt1.log](endpoint_C_seed2.attempt1.log) |
| endpoint_B_seed2 | 1 | complete | [endpoint_B_seed2.attempt1.log](endpoint_B_seed2.attempt1.log) |
| endpoint_D_seed2 | 4 | complete | [endpoint_D_seed2.attempt1.log](endpoint_D_seed2.attempt1.log) |
| sgd_seed0_sparse_train | 0 | complete | [sgd_seed0_sparse_train.attempt1.log](sgd_seed0_sparse_train.attempt1.log) |
| sgd_seed0_sparse_test | 1 | complete | [sgd_seed0_sparse_test.attempt1.log](sgd_seed0_sparse_test.attempt1.log) |
| sgd_seed0_sparse_train_full | 1 | complete | [sgd_seed0_sparse_train_full.attempt1.log](sgd_seed0_sparse_train_full.attempt1.log) |
| gam_seed0_sparse_train | 1 | complete | [gam_seed0_sparse_train.attempt1.log](gam_seed0_sparse_train.attempt1.log) |
| gam_seed0_sparse_test | 0 | complete | [gam_seed0_sparse_test.attempt1.log](gam_seed0_sparse_test.attempt1.log) |
| gam_seed0_sparse_train_full | 0 | complete | [gam_seed0_sparse_train_full.attempt1.log](gam_seed0_sparse_train_full.attempt1.log) |
| sgd_seed1_sparse_train | 0 | complete | [sgd_seed1_sparse_train.attempt1.log](sgd_seed1_sparse_train.attempt1.log) |
| sgd_seed1_sparse_test | 1 | complete | [sgd_seed1_sparse_test.attempt1.log](sgd_seed1_sparse_test.attempt1.log) |
| sgd_seed1_sparse_train_full | 1 | complete | [sgd_seed1_sparse_train_full.attempt1.log](sgd_seed1_sparse_train_full.attempt1.log) |
| gam_seed1_sparse_train | 0 | complete | [gam_seed1_sparse_train.attempt1.log](gam_seed1_sparse_train.attempt1.log) |
| gam_seed1_sparse_test | 1 | complete | [gam_seed1_sparse_test.attempt1.log](gam_seed1_sparse_test.attempt1.log) |
| gam_seed1_sparse_train_full | 4 | complete | [gam_seed1_sparse_train_full.attempt1.log](gam_seed1_sparse_train_full.attempt1.log) |
| sgd_seed2_sparse_train | 4 | complete | [sgd_seed2_sparse_train.attempt1.log](sgd_seed2_sparse_train.attempt1.log) |
| sgd_seed2_sparse_test | 1 | complete | [sgd_seed2_sparse_test.attempt1.log](sgd_seed2_sparse_test.attempt1.log) |
| sgd_seed2_sparse_train_full | 4 | complete | [sgd_seed2_sparse_train_full.attempt1.log](sgd_seed2_sparse_train_full.attempt1.log) |
| gam_seed2_sparse_train | 0 | complete | [gam_seed2_sparse_train.attempt1.log](gam_seed2_sparse_train.attempt1.log) |
| gam_seed2_sparse_test | 1 | complete | [gam_seed2_sparse_test.attempt1.log](gam_seed2_sparse_test.attempt1.log) |
| gam_seed2_sparse_train_full | 1 | complete | [gam_seed2_sparse_train_full.attempt1.log](gam_seed2_sparse_train_full.attempt1.log) |
| sgd_seed0_trajectory_fisher | 1 | complete | [sgd_seed0_trajectory_fisher.attempt1.log](sgd_seed0_trajectory_fisher.attempt1.log) |
| gam_seed0_trajectory_fisher | 2 | complete | [gam_seed0_trajectory_fisher.attempt1.log](gam_seed0_trajectory_fisher.attempt1.log) |
| sgd_seed1_trajectory_fisher | 4 | complete | [sgd_seed1_trajectory_fisher.attempt1.log](sgd_seed1_trajectory_fisher.attempt1.log) |
| gam_seed1_trajectory_fisher | 5 | complete | [gam_seed1_trajectory_fisher.attempt1.log](gam_seed1_trajectory_fisher.attempt1.log) |
| sgd_seed2_trajectory_fisher | 2 | complete | [sgd_seed2_trajectory_fisher.attempt1.log](sgd_seed2_trajectory_fisher.attempt1.log) |
| gam_seed2_trajectory_fisher | 6 | complete | [gam_seed2_trajectory_fisher.attempt1.log](gam_seed2_trajectory_fisher.attempt1.log) |
| sgd_seed0_path_t01 | 0 | complete | [sgd_seed0_path_t01.attempt1.log](sgd_seed0_path_t01.attempt1.log) |
| sgd_seed0_path_t05 | 0 | complete | [sgd_seed0_path_t05.attempt1.log](sgd_seed0_path_t05.attempt1.log) |
| sgd_seed0_path_t09 | 1 | complete | [sgd_seed0_path_t09.attempt1.log](sgd_seed0_path_t09.attempt1.log) |
| gam_seed0_path_t01 | 2 | complete | [gam_seed0_path_t01.attempt1.log](gam_seed0_path_t01.attempt1.log) |
| gam_seed0_path_t05 | 5 | complete | [gam_seed0_path_t05.attempt1.log](gam_seed0_path_t05.attempt1.log) |
| gam_seed0_path_t09 | 6 | complete | [gam_seed0_path_t09.attempt1.log](gam_seed0_path_t09.attempt1.log) |
| sgd_seed1_path_t01 | 0 | complete | [sgd_seed1_path_t01.attempt1.log](sgd_seed1_path_t01.attempt1.log) |
| sgd_seed1_path_t05 | 1 | complete | [sgd_seed1_path_t05.attempt1.log](sgd_seed1_path_t05.attempt1.log) |
| sgd_seed1_path_t09 | 0 | complete | [sgd_seed1_path_t09.attempt1.log](sgd_seed1_path_t09.attempt1.log) |
| gam_seed1_path_t01 | 5 | complete | [gam_seed1_path_t01.attempt1.log](gam_seed1_path_t01.attempt1.log) |
| gam_seed1_path_t05 | 5 | complete | [gam_seed1_path_t05.attempt1.log](gam_seed1_path_t05.attempt1.log) |
| gam_seed1_path_t09 | 2 | complete | [gam_seed1_path_t09.attempt1.log](gam_seed1_path_t09.attempt1.log) |
| sgd_seed2_path_t01 | 3 | complete | [sgd_seed2_path_t01.attempt1.log](sgd_seed2_path_t01.attempt1.log) |
| sgd_seed2_path_t05 | 3 | complete | [sgd_seed2_path_t05.attempt1.log](sgd_seed2_path_t05.attempt1.log) |
| sgd_seed2_path_t09 | 3 | complete | [sgd_seed2_path_t09.attempt1.log](sgd_seed2_path_t09.attempt1.log) |
| gam_seed2_path_t01 | 5 | complete | [gam_seed2_path_t01.attempt1.log](gam_seed2_path_t01.attempt1.log) |
| gam_seed2_path_t05 | 5 | complete | [gam_seed2_path_t05.attempt1.log](gam_seed2_path_t05.attempt1.log) |
| gam_seed2_path_t09 | 2 | complete | [gam_seed2_path_t09.attempt1.log](gam_seed2_path_t09.attempt1.log) |
| sgd_seed0_bridge_full | 2 | complete | [sgd_seed0_bridge_full.attempt1.log](sgd_seed0_bridge_full.attempt1.log) |
| gam_seed0_bridge_full | 2 | complete | [gam_seed0_bridge_full.attempt1.log](gam_seed0_bridge_full.attempt1.log) |
| sgd_seed1_bridge_full | 2 | complete | [sgd_seed1_bridge_full.attempt1.log](sgd_seed1_bridge_full.attempt1.log) |
| gam_seed1_bridge_full | 2 | complete | [gam_seed1_bridge_full.attempt1.log](gam_seed1_bridge_full.attempt1.log) |
| sgd_seed2_bridge_full | 2 | complete | [sgd_seed2_bridge_full.attempt1.log](sgd_seed2_bridge_full.attempt1.log) |
| gam_seed2_bridge_full | 0 | complete | [gam_seed2_bridge_full.attempt1.log](gam_seed2_bridge_full.attempt1.log) |
| sgd_seed0_dense | 3 | complete | [sgd_seed0_dense.attempt1.log](sgd_seed0_dense.attempt1.log) |
| gam_seed0_dense | 1 | complete | [gam_seed0_dense.attempt1.log](gam_seed0_dense.attempt1.log) |
| sgd_seed1_dense | 0 | complete | [sgd_seed1_dense.attempt1.log](sgd_seed1_dense.attempt1.log) |
| gam_seed1_dense | 4 | complete | [gam_seed1_dense.attempt1.log](gam_seed1_dense.attempt1.log) |
| sgd_seed2_dense | 2 | complete | [sgd_seed2_dense.attempt1.log](sgd_seed2_dense.attempt1.log) |
| gam_seed2_dense | 6 | complete | [gam_seed2_dense.attempt1.log](gam_seed2_dense.attempt1.log) |
| final_report | CPU | complete | [final_report.attempt1.log](final_report.attempt1.log) |
