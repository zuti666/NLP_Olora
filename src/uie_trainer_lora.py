import os
os.environ['WANDB_DISABLED'] = "True"
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import logging
logging.set_verbosity(logging.ERROR)

import logging
from contextlib import nullcontext
import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_torch_tpu_available
from types import SimpleNamespace

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX

from optimer_sam import SAM, disable_running_stats, enable_running_stats
from optimer_gam import GAM
from optimer_rwp import RWP, RWPAdamW, generate_pertubation
# disable_running_stats

import h5py
import copy
from datetime import datetime 
from torch.nn import functional as F
from torch.utils.data import DistributedSampler
import json

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    # logger.debug(f'model.config._name_or_path: {model.config._name_or_path}')
    
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:
            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
                # final_predictions.append(pred.strip())
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class UIETrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # logger.debug("UIETrainer args: %r", args)
        # logger.debug("UIETrainer kwargs: %r", kwargs)

        # 当 使用 rwp 来生成扰动
        logger.debug(f"rwp_type: {self.args.rwp_type}")

        if self.args.rwp_type != "":
            rwp_noise_type = getattr(self.args, "rwp_noise_type", "")
            noise_std= getattr(self.args, "noise_std", 0.002)

            if "fisher" in rwp_noise_type:

                # Fisher 路径参数,可为 ""（空字符串）或有效路径
                fisher_path = getattr(self.args, "fisher_load_path", "")  # 推荐加到 args 里

                # 1. 如果有路径且文件存在,则加载历史 Fisher 信息
                if fisher_path and os.path.isfile(fisher_path):
                    logger.info(f"加载历史 Fisher 信息: {fisher_path}")
                    self.fisher_dict = torch.load(fisher_path, map_location="cpu")
                    # --- 保证所有 Fisher buffer 跟参数 device 一致 ---
                    for name, param in self.model.named_parameters():
                        if name in self.fisher_dict and self.fisher_dict[name].device != param.device:
                            self.fisher_dict[name] = self.fisher_dict[name].to(param.device)
                    logger.info("已将所有加载的 Fisher buffer 迁移至参数 device。")
                else:
                    # 2. 否则初始化 Fisher 为零,只对新分支生效
                    logger.info("未指定 Fisher 信息路径或文件不存在,初始化为全零")
                    self.fisher_dict = {}
                    initialized_names = []
                    for name, param in self.model.named_parameters():
                        if ("seqlora"  not in self.args.lora_strategy) and ("loranew_" in name):
                            self.fisher_dict[name] = torch.zeros_like(param) + 1e-8
                            initialized_names.append(name)
                            # logger.debug(f"初始化Fisher信息包含: {name}")
                        elif ("seqlora" in self.args.lora_strategy) and ("lora" in name):
                            self.fisher_dict[name] = torch.zeros_like(param) + 1e-8
                            initialized_names.append(name)
                            # logger.debug(f"初始化Fisher信息包含: {name}")
                    # 输出数值统计（确保确实初始化为零）
                    # 输出第一个参数的详细信息作为示例
                    first_param = initialized_names[0]
                    # logger.debug(f"首个初始化的 Fisher buffer: {first_param}")
                    # logger.debug(f"形状: {self.fisher_dict[first_param].shape}")
                    # logger.debug(f"设备: {self.fisher_dict[first_param].device}")
                    # logger.debug(f"数据类型: {self.fisher_dict[first_param].dtype}")
                    if self.fisher_dict[first_param].numel() > 0:
                        logger.debug(f"数值范围: [{self.fisher_dict[first_param].min()}, {self.fisher_dict[first_param].max()}]")
                        logger.debug(f"均值: {self.fisher_dict[first_param].mean()}")
                    else:
                        logger.debug("参数为空张量")

                    logger.info("已在参数 device 上初始化所有 Fisher ")

                # 为当前任务也初始化一个独立的信息
                self.fisher_dict_currentTask = {}
                for name, param in self.model.named_parameters():
                    if ("seqlora"  not in self.args.lora_strategy) and ("loranew_" in name):
                        self.fisher_dict_currentTask[name] = torch.zeros_like(param)
                        # 可选：保存快照，带任务编号
                        save_path = f"{self.args.output_dir}/fisher_snapshots/fisher_init_task.pt"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(self.fisher_dict_currentTask, save_path)
                    elif ("seqlora" in self.args.lora_strategy) and ("lora" in name):
                        self.fisher_dict_currentTask[name] = torch.zeros_like(param)
                        # 可选：保存快照，带任务编号
                        save_path = f"{self.args.output_dir}/fisher_snapshots/fisher_init_task.pt"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(self.fisher_dict_currentTask, save_path)

                    # # 检查是否存在应初始化但未初始化的参数
                    # missing_params = [name for name, _ in all_params 
                    #                 if "loranew_" in name and name not in self.fisher_dict]
                    # if missing_params:
                    #     logger.warning(f"发现 {len(missing_params)} 个包含'loranew_'但未初始化的参数")
                    #     logger.warning(f"未初始化的参数: {missing_params}")




        use_probe = getattr(self.args, "train_method", "")
        if  use_probe == "use_probe":
            # 尝试寻找 probe_head
            probe_num_classes = getattr(self.args, "probe_num_classes", "")
            probe_feature_mode = getattr(self.args, "probe_feature_model", "")
            logger.info(f"use_probe, probe_num_classes: {probe_num_classes}; probe_feature_mode:{probe_feature_mode}")

    def _is_deepspeed_enabled_compat(self, args):
        return bool(getattr(self, "is_deepspeed_enabled", False) or getattr(args, "deepspeed", None))

    def _is_fsdp_enabled_compat(self):
        legacy_fsdp = getattr(self, "fsdp", None)
        if legacy_fsdp is not None:
            return True
        return bool(
            getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_fsdp_xla_enabled", False)
        )

    def _is_sharded_ddp_enabled_compat(self):
        sharded_ddp = getattr(self, "sharded_ddp", None)
        return sharded_ddp is not None and sharded_ddp != ShardedDDPOption.SIMPLE

    def _should_delay_optimizer_creation(self):
        # Transformers 4.36+ replaced `self.sharded_ddp` / `self.fsdp` with boolean flags.
        return (
            self._is_sharded_ddp_enabled_compat()
            or is_sagemaker_mp_enabled()
            or self._is_fsdp_enabled_compat()
        )

    def generate_noise(self,param, mode, std, fisher_param=None,fisher_scaler=2): 
        """
        Generate noise to inject into a parameter tensor using specified mode.
        产生噪声

        """

        # logger.debug(f"generate_noise  std={std}")

        # 跳过空参数,初始的lora_a,b
        if param.numel() == 0:
            return torch.zeros_like(param)


        if mode == "Gauss_standard":
            # ε_ij ~ N(0, σ²) 
            # 正态分布,每个元素的标准差都是σ,所有噪声元素共享相同标准差,与参数的值无关
            # 标准高斯噪声：所有元素独立采样于 N(0, σ²)
            # randn_like 生成标准正态分布（标准差为 1）,再通过 * std 缩放为 N(0, sigma²)
            
            # 下面代码 与 noise = torch.normal(mean=0., std=std, size=param.shape)完全等价
            
            scaler =  std
            noise = torch.randn_like(param) * scaler  # 标准差为 std

            return noise
        elif mode == "Gauss_element":
            # ε_ij ~ N(0, σ² * |W_ij|²)
            # 每个元素都是独立地根据 param的绝对值进行缩放 
            # 元素级噪声：每个元素独立采样于 N(0, σ² * |W_ij|²)
            # noise = torch.normal(
            #     mean=0.0, 
            #     std=std * (param.abs()+ 1e-16)  # 标准差为 sigma * |W_ij|
            # )
            # return noise
            # 经过分析  torch.normal 和  torch.randn_like   的实现是一样的

            # 另一种实现 
            # 生成标准正态分布后 然后再乘以 param 的绝对值
            scaler = std * (param.abs()+ 1e-16)
            noise = torch.randn_like(param) * scaler
            return noise

        elif mode == "Gauss_matrix":
            # # ε_ij ~ N(0, σ² * ||W||_F²)
            # 计算 Frobenius 范数：||W||_F = sqrt(sum(|W_ij|^2))
            fro_norm = torch.norm(param, p='fro')  # 直接计算 Frobenius 范数
            # 生成噪声：ε_ij ~ N(0, σ² * ||W||_F²)
            # 所有噪声元素共享 相同标准差 σ * ||W||_F,因此生成标准正态分布后统一乘以该标准差
            scaler =  std * fro_norm
            noise = torch.randn_like(param)  * scaler

        

        elif mode == "lpf_sgd_Gauss":
            # LPF-SGD: ε_ij ~ N(0, σ² * ||W_i||^2)
            # ε_ij ~ N(0,σ² * ||W_i,:||^2)

            """# 参考的官方实现 
            # [LPF-SGD/codes/wrn_dataaug/example/lpf_train.py at master · devansh20la/LPF-SGD]
            # (https://github.com/devansh20la/LPF-SGD/blob/master/codes/wrn_dataaug/example/lpf_train.py)
            if len(mp.shape) > 1:
                    sh = mp.shape
                    sh_mul = np.prod(sh[1:])
            
            # 下面这行代码调用了三个代码 
            # mp.view(m, -1).norm(dim=1, keepdim=True)：取每行范数 ||W_i||_2, shape = [m, 1]
            # row_norm.repeat(1, n).view(m, n) ： 将它复制成和 param 同形状 scale 矩阵,shape = [m, n]
            
                    temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
            # torch.normal :从 N(0, (std * scale)^2) 中采样
                    temp = torch.normal(0, std*temp).to(mp.data.device)
            else: # 如果是偏置或 LayerNorm weight 这类1D参数
                    temp = torch.empty_like(mp, device=mp.data.device)
                    temp.normal_(0, std*(mp.view(-1).norm().item() + 1e-16))
            """
            # 按着原代码逻辑与风格的实现
            # if len(param.shape) > 1:
            #     sh = param.shape
            #     #  filter 也就是矩阵的每一行
            #     sh_mul = np.prod(sh[1:])
            #     #  param.view(sh[0], -1).norm(dim=1, keepdim=True):计算每一行的范数  shape = [m, 1]
            #     #  repeat(1, sh_mul): 将这一行的结果都变成这一行的范数
            #     noise_temp = param.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(param.shape)
            #     #  生成一个与 std * param 形状相同的张量,
            #     # 每个元素从均值为 0、标准差为 std * param 对应位置值的正态分布中采样
            #     noise_temp = torch.normal(0, std*noise_temp).to(param.data.device)
            # else:
            #     # 对于 bias 或 LayerNorm 的一维参数
            #     noise_temp = torch.empty_like(param)
            #     noise_temp.normal_(0, std*(param.view(-1).norm().item() + 1e-16))
            # return noise_temp

            

            # 从 N(0, 1)进行缩放,而不是 torch.normal 
            if len(param.shape) > 1:
                sh = param.shape
                sh_mul = np.prod(sh[1:])
                # 每行的L2范数：[m, 1]
                row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
                # 将其扩展为与param相同形状
                row_norms_param= row_norms.repeat(1, sh_mul).view(param.shape)


                # 进行归一化设置
                #---------原始内容
                # 将其扩展为与param相同形状
                # std_matrix = row_norms.repeat(1, sh_mul).view(param.shape)
                # 使用 randn_like,然后缩放为每个元素 std = std * row_norm
                #---------原始内容

                #---------修改
                # # **关键修改：对 row_norms 进行局部归一化**
                # # 方案 A: L2 归一化（推荐,通常效果稳定,使 row_norms 的 L2 范数总和为 1）
                # total_row_norm_l2 =row_norms_param.norm() + 1e-16 # 对所有行范数组成的向量求L2范数
                # normalized_row_norms = row_norms_param / total_row_norm_l2
                # # 将归一化后的因子扩展为与param相同形状
                # std_matrix = normalized_row_norms.repeat(1, sh_mul).view(param.shape)
                #---------修改

                scaler = std * row_norms_param
                noise_temp = torch.randn_like(param) * scaler
            else:
                # 对于一维参数,先计算整体L2范数（加上稳定项）
                scale = std * (param.view(-1).norm().item() + 1e-16)
                noise_temp = torch.randn_like(param) * scale




            return noise_temp

        elif mode == "RAW_fisher_1":

            
            scaler =  std
            noise_temp = torch.randn_like(param) * scaler  # 标准差为 std

            # 在生成噪声的代码中添加
            # assert noise_temp.shape == fisher_param.shape, "Fisher 信息形状与噪声不匹配"
            if fisher_param is not None:
                
                # --- 方法一变种：根号下 ---
                # fisher_mean = fisher_param.mean()
                # # alpha = 1e2 / fisher_mean
                scaler_1_2 = 1 / torch.sqrt(1 +  fisher_scaler* fisher_param*1e7)
                # 2. 当前任务权重因子
                # weight_factor = param.abs() + 1e-8  # 也可用其他自定义方式 * weight_factor
                # logger.info(f"[Scale-1-2] sqrt(1 + a * F * 1e7): min={scaler_1_2.min().item():.4e}, max={scaler_1_2.max().item():.4e}, mean={scaler_1_2.mean().item():.4e}, std={scaler_1_2.std().item():.4e}")
                # logger.info(f"weight_factor:{weight_factor}")
                scaler = torch.clamp(scaler_1_2, 0.3, 3.0)
                noise_temp = noise_temp * scaler  

                # logger.debug(f"std{std} fisher_scaler{fisher_scaler}")

            return noise_temp

        elif mode == "RAW_fisher_2":
            # sh = param.shape
            # sh_mul = np.prod(sh[1:])
            # 
            
            scaler =  std
            noise_temp = torch.randn_like(param) * scaler  # 标准差为 std

            # 在生成噪声的代码中添加
            # assert noise_temp.shape == fisher_param.shape, "Fisher 信息形状与噪声不匹配"
            if fisher_param is not None:
                # logger.debug("成功进入")

                # fisher =  fisher_param.view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                # epsilon = 1e-8
                # 对数归一化
                # fisher_log = torch.log(fisher_param + epsilon)
                # fisher_min, fisher_max = fisher_log.min(), fisher_log.max()
                # fisher_norm = (fisher_log - fisher_min) / (fisher_max - fisher_min + 1e-8)
                # 指数反比缩放
                
                # weight = torch.exp(-fisher_scaler* fisher_norm)
                # weight = torch.clamp(weight, 0.05, 1.0)  # 防止过小
                # noise_temp = noise_temp * weight

                #---------------
                

                #------------------------------------------------
                # 1. 输出原始 Fisher 统计指标
                # logger.info(f"[Fisher] 原始指标: shape={fisher_param.shape}, min={fisher_param.min().item():.4e}, max={fisher_param.max().item():.4e}, mean={fisher_param.mean().item():.4e}, std={fisher_param.std().item():.4e}")

                # 2. 不同缩放方式的 scale tensor 对比
                # --- 方法一：根号下 ---
                # fisher =  fisher_param.view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                # scaler_1 = 1 / torch.sqrt(1 +  fisher_scaler* fisher*1e6 )
                # logger.info(f"[Scale-1] sqrt(1 + a * F * 1e6): min={scaler_1.min().item():.4e}, max={scaler_1.max().item():.4e}, mean={scaler_1.mean().item():.4e}, std={scaler_1.std().item():.4e}")

                # --- 方法一变种：根号下 ---
                # fisher =  fisher_param.view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                # scaler_1_2 = 1 / torch.sqrt(1 +  fisher_scaler* fisher_param*1e6 )
                # logger.info(f"[Scale-1-2] sqrt(1 + a * F * 1e6): min={scaler_1_2.min().item():.4e}, max={scaler_1_2.max().item():.4e}, mean={scaler_1_2.mean().item():.4e}, std={scaler_1_2.std().item():.4e}")


                # 方法二： 对数归一化
                epsilon = 1e-8
                fisher_log = torch.log(fisher_param + epsilon)

                fisher_min, fisher_max = fisher_log.min(), fisher_log.max()
                # fisher_norm = (fisher_log - fisher_min) / (fisher_max - fisher_min + 1e-8)
                fisher_range = fisher_max - fisher_min
                if fisher_range < 1e-6:
                    fisher_norm = torch.zeros_like(fisher_log)
                else:
                    fisher_norm = (fisher_log - fisher_min) / (fisher_range + 1e-8)
                
                # 指数反比缩放
                fisher_scaler = 2.0
                weight = torch.exp(-fisher_scaler* fisher_norm)


                scaler = 0.5 + 1.5 * weight     
                # scaler = 0.3 + 2.7 * weight     

                # logger.info(f"[RWP-fisher2] scaler min={scaler.min().item()}, max={scaler.max().item()}")
                
                # re-map to [0.5, 2.0]
                scaler2 = torch.clamp(scaler, 0.2, 3.0)  # 防止过小

                # logger.info(f"[Scale-2] exp(-a * F_norm): min={scaler_2.min().item():.4e}, max={scaler_2.max().item():.4e}, mean={scaler_2.mean().item():.4e}, std={scaler_2.std().item():.4e}")


                # 2. 当前任务权重因子
                # weight_factor = param.abs() + 1e-8  # 也可用其他自定义方式
                # *weight_factor
                noise_temp = noise_temp * scaler2

            return noise_temp
        


        elif mode == "mARWP_fisher":

            """
            https://github.com/nblt/mARWP/blob/main/train_marwp.py

            官方实现： 
            with torch.no_grad():
                noise = []
                for ii, mp in enumerate(model.parameters()):
                    sh = mp.data.shape   # 当前参数张量的形状
                    sh_mul = int(np.prod(sh[1:])) # 每一行（如 conv/filter）中包含的元素个数,用于广播
                    # 如果提供了 Fisher 信息（任务相关的方向不确定性指标）
                    if fisher_arr != []:
                        # 将 Fisher 信息张量调整形状为 [行数, -1],在每一行上求和（即累计该行的 Fisher 信息）
                        # 将 [行数, 1] 的 Fisher 信息扩展（广播）为与参数形状相同的张量
                        fisher = fisher_arr[ii].view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                        
                    if len(mp.shape) > 1:
                        # 对于多维参数（如 weight matrix）,先计算每一行的 L2 范数,然后 扩展成与 mp 相同的 shape
                        temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
                        # # 使用标准正态分布采样,然后缩放标准差为：args.sigma × 每行范数
                        temp = torch.normal(0, args.sigma*temp).to(mp.data.device)
                    else:
                        temp = torch.empty_like(mp, device=mp.data.device)
                        temp.normal_(0, args.sigma*(mp.view(-1).norm().item() + 1e-16))
                    
                    # ---- Fisher 相关的缩放逻辑（任务感知噪声）----
                    if fisher_arr != []:
                    
                        # 使用 Fisher 信息缩放噪声项：
                        # temp ← temp / sqrt(1 + η × fisher)
                        # 使得在“确定性强”（fisher 大）的方向上抑制噪声,反之增强不确定方向上的噪声

                        temp /= torch.sqrt(1 + args.eta * fisher)
                        
                    noise.append(temp)
                    mp.data.add_(noise[-1])

            
            """


            sh = param.shape
            sh_mul = np.prod(sh[1:])
            if len(param.shape) > 1:
                # 计算每一行的 L2 范数,并广播为和 mp 相同形状
                row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)  # shape = [m, 1]
                std_matrix = row_norms.repeat(1, sh_mul).view(param.shape)
                # 构建噪声标准差张量
                scaler = std * std_matrix
                # 使用标准正态采样后乘以 std,实现与 normal(mean=0, std=...) 等效行为
                noise_temp = torch.randn_like(param) * scaler
            else:
                # 一维参数：整体 L2 范数 + ε 保证数值稳定
                scale = std * (param.view(-1).norm().item() + 1e-16)
                noise_temp = torch.randn_like(param) * scale

            # 在生成噪声的代码中添加
            # assert noise_temp.shape == fisher_param.shape, "Fisher 信息形状与噪声不匹配"
            if fisher_param is not None:
                # 按着原始实现,这里应该对 Fisher_param 按行来计算 
                # 按行 sum 得到每行一个 Fisher 值 , 广播扩展为完整参数 shape
                fisher =  fisher_param.view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                # 按照 Fisher 信息缩放噪声项： ε ← ε / sqrt(1 + η × F)
                noise_temp /= torch.sqrt(1 +  fisher_scaler* fisher )

            return noise_temp
        

        elif mode == "flatLoRA_Gauss":
            # 从 N(0, 1)进行缩放,而不是 torch.normal 
            if len(param.shape) > 1:
                sh = param.shape # 如果维度为 m * n
                sh_mul = np.prod(sh[1:]) # 这个就是 n

                # 每行的L2范数：[m, 1]
                row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
                # 将其扩展为与param相同形状
                std_matrix = row_norms.repeat(1, sh_mul).view(param.shape)
                # 使用 randn_like,然后缩放为每个元素 std = std * row_norm
                scaler = std / math.sqrt(param.shape[1]) * std_matrix 

                noise_temp = torch.randn_like(param) * scaler
            else:
                # 对于一维参数,先计算整体L2范数（加上稳定项）
                n = param.shape[0]  # 一维向量的“长度”就是输出维度
                scale = std / math.sqrt(n) * (param.view(-1).norm().item() + 1e-16)

                noise_temp = torch.randn_like(param) * scale

            return noise_temp

        else:
            raise ValueError(f"Unknown noise mode: {mode}")
            
                                                                                            
    def _gather_grad_vector(self, model):
        """
        收集所有 requires_grad 且有 grad 的参数梯度,拼接为一维向量
        """
        grad_list = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_list.append(param.grad.detach().clone().view(-1))
        if len(grad_list) == 0:
            raise ValueError("未检测到任何梯度,请检查梯度是否正确反向传播。")
        return torch.cat(grad_list)

    def _assign_grad_vector(self, model, grad_vector):
        """
        将一维梯度向量 grad_vector 拆分并赋值回模型参数的 .grad
        """
        offset = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(grad_vector[offset:offset + numel].view_as(param))
                offset += numel
        if offset != grad_vector.numel():
            raise ValueError("Grad vector size 与参数维度不匹配！")
    
    
    
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        logger.info(f"调用自定义 looping 函数进行训练,batch_size: {batch_size}")
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self._should_delay_optimizer_creation()
        if self._is_deepspeed_enabled_compat(args):

            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            # 然后交给 deepspeed,但不让它覆盖 optimizer
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )

            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine



            # ---- 原来的设置
            # deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
            #     self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            # )
            # self.model = deepspeed_engine.module
            # self.model_wrapped = deepspeed_engine
            # self.deepspeed = deepspeed_engine
            # self.optimizer = optimizer
            # self.lr_scheduler = lr_scheduler
            #$ ---- 原来的设置

        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)
        if isinstance(self.optimizer, GAM):
            self.optimizer.model = model  # 确保 maybe_no_sync 指向 DDP

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        # 在训练之前,将梯度进行归0
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        optimizer_type_name = str(getattr(self.args, "optimizer_type", "")).lower()
        sam_optimizer_active = "sam" in optimizer_type_name
        gam_optimizer_active = "gam" in optimizer_type_name
        sam_accumulated_inputs = []
        gam_accumulated_inputs = []
        if sam_optimizer_active and self.deepspeed:
            raise NotImplementedError("Accumulation-aware SAM currently supports the standard/DDP path only, not DeepSpeed.")
        if gam_optimizer_active and self.deepspeed:
            raise NotImplementedError("Accumulation-aware GAM currently supports the standard/DDP path only, not DeepSpeed.")
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if sam_optimizer_active:
                    # Keep one accumulation window so SAM can replay the exact same micro-batches on w + e(w).
                    sam_accumulated_inputs.append(inputs)

                if gam_optimizer_active:
                    # GAM needs the whole accumulation window so each closure replay sees the same effective batch.
                    gam_accumulated_inputs.append(inputs)



                if self.args.gradient_accumulation_steps == 1 and  ("rpw_single_jiou" in self.args.rwp_type):
                # 添加一个分支,专门处理 Rwp 按着论文,奇偶不同的情形,作为一起进行合并的情形

                    if step % 2 == 0:
                        pass

                    if step % 2 == 1:
                        pass

                
                else :

                    if (
                        (total_batched_samples % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                        
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            # if self.args.use_probe:
                            if self.args.train_method == 'use_probe':
                                tr_loss_step = self.training_step(model, inputs)
                            elif self.args.train_method == "lora":
                            # 修改进行梯度更新的地方,替换掉原来的函数调用
                            # -------------------
                                model.train()
                                enable_running_stats(model)
                                inputs = self._prepare_inputs(inputs)

                                if is_sagemaker_mp_enabled():
                                    loss_mb = smp_forward_backward(
                                        model, inputs, self.args.gradient_accumulation_steps)
                                    return loss_mb.reduce_mean().detach().to(self.args.device)
                                
                                # self.print_trainable_parameters(model)
                                with self.compute_loss_context_manager():
                                    loss = self.compute_loss(model, inputs)

                                logger.debug(f'no_sync:training_step compute_loss: {loss.item():.4f}')
                                

                                if self.args.n_gpu > 1:
                                    loss = loss.mean()

                                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                    loss = loss / self.args.gradient_accumulation_steps

                                ########################## Regularization ##########################
                                if self.args.lora_strategy.lower() == "olora":
                                    orthogonal_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "lora_A" in name:
                                            for name_, param_ in model.named_parameters():
                                                if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                                    orthogonal_loss += torch.abs(
                                                        torch.mm(param, param_.T)).sum()
                                                    break

                                    l2_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "loranew_" in name:
                                            l2_loss += torch.norm(param, p=2)

                                    lamda_1 = self.args.olora_lamda_1
                                    lamda_2 = self.args.olora_lamda_2

                                    logger.info(f"no_sync: orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                                    loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
                                elif self.args.lora_strategy.lower() == "nlora":
                                    l1_loss = 0.
                                    loranew_A_params = {}
                                    loranew_B_params = {}
                                    for name, param in self.model.named_parameters():
                                        if "loranew_A" in name:
                                            loranew_A_params[name.split("loranew_A")[0]] = param
                                        elif "loranew_B" in name:
                                            loranew_B_params[name.split("loranew_B")[0]] = param

                                    for key in loranew_A_params:
                                        if key in loranew_B_params:
                                            l1_loss += torch.norm(
                                                torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

                                    lamda_1 = self.args.olora_lamda_1

                                    logger.info(f"no_sync:Nlora_loss: {l1_loss.item():.4f};   accuracy_loss: {loss.item():.4f}; λ1: {lamda_1};")

                                    loss = loss + l1_loss * lamda_1
                                elif self.args.lora_strategy.lower() == "inclora":
                                    logger.debug(f"no_sync:inclora accuracy_loss: {loss.item():.4f}")
                                elif self.args.lora_strategy.lower() == "seqlora":
                                    logger.debug(f"no_sync:seqLora accuracy_loss: {loss.item():.4f}")
                                
                                elif self.args.lora_strategy.lower() == "lora_l2":
                                    l2_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "loranew_" in name:
                                            l2_loss += torch.norm(param, p=2)

                                    logger.info(f" l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f};  λ2: {lamda_2}")
                                    lamda_2 = self.args.olora_lamda_2
                                    loss = loss + l2_loss * lamda_2
                                ######################################################################
                                logger.debug(f"no_sync:sum_loss: {loss.item():.4f}")

                                if gam_optimizer_active:
                                    logger.debug("GAM active: skip immediate backward and defer to optimizer replay.")
                                elif self.do_grad_scaling:
                                    self.scaler.scale(loss).backward()
                                    logger.debug(f"梯度缩放")
                                elif self.use_apex:
                                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                        scaled_loss.backward()
                                    logger.debug(f"梯度缩放")
                                elif self.deepspeed:
                                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                                    loss = self.deepspeed.backward(loss)
                                    logger.debug(f"梯度缩放")
                                else:
                                    loss.backward()

                                tr_loss_step = loss.detach()
                            #------------------------------
                            elif self.args.train_method == "finetune":
                            # 修改进行梯度更新的地方,替换掉原来的函数调用
                            # -------------------
                                model.train()
                                enable_running_stats(model)
                                inputs = self._prepare_inputs(inputs)

                                if is_sagemaker_mp_enabled():
                                    loss_mb = smp_forward_backward(
                                        model, inputs, self.args.gradient_accumulation_steps)
                                    return loss_mb.reduce_mean().detach().to(self.args.device)
                                
                                # self.print_trainable_parameters(model)
                                with self.compute_loss_context_manager():
                                    loss = self.compute_loss(model, inputs)

                                logger.info(f'training_step compute_loss: {loss.item():.4f}')
                                

                                if self.args.n_gpu > 1:
                                    loss = loss.mean()

                                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                    loss = loss / self.args.gradient_accumulation_steps
                                

                                logger.debug(f"loss: {loss.item():.4f}")

                                if self.do_grad_scaling:
                                    self.scaler.scale(loss).backward()
                                    logger.debug(f"梯度缩放")
                                elif self.use_apex:
                                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                        scaled_loss.backward()
                                    logger.debug(f"梯度缩放")
                                elif self.deepspeed:
                                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                                    loss = self.deepspeed.backward(loss)
                                    logger.debug(f"梯度缩放")
                                else:
                                    loss.backward()

                                tr_loss_step = loss.detach()

                            



                    else:
                        # if self.args.use_probe:
                        if self.args.train_method == 'use_probe':
                            tr_loss_step = self.training_step(model, inputs)
                        elif self.args.train_method == "lora":
                            # 修改进行梯度更新的地方,替换掉原来的函数调用
                            # -------------------
                            model.train()
                            enable_running_stats(model)
                            inputs = self._prepare_inputs(inputs)

                            if is_sagemaker_mp_enabled():
                                loss_mb = smp_forward_backward(
                                    model, inputs, self.args.gradient_accumulation_steps)
                                return loss_mb.reduce_mean().detach().to(self.args.device)
                            
                            # self.print_trainable_parameters(model)
                            with self.compute_loss_context_manager():
                                loss = self.compute_loss(model, inputs)

                            logger.info(f'sync: training_step compute_loss: {loss.item():.4f}')
                            

                            if self.args.n_gpu > 1:
                                loss = loss.mean()

                            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                loss = loss / self.args.gradient_accumulation_steps
                                logger.info(f'sync: average training_step compute_loss: {loss.item():.4f}')


                            ########################### Regularization ##########################
                            if self.args.lora_strategy.lower() == "olora":
                                orthogonal_loss = 0.
                                for name, param in model.named_parameters():
                                    if "lora_A" in name:
                                        for name_, param_ in model.named_parameters():
                                            if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                                orthogonal_loss += torch.abs(
                                                    torch.mm(param, param_.T)).sum()
                                                break

                                l2_loss = 0.
                                for name, param in model.named_parameters():
                                    if "loranew_" in name:
                                        l2_loss += torch.norm(param, p=2)

                                lamda_1 = self.args.olora_lamda_1
                                lamda_2 = self.args.olora_lamda_2

                                logger.info(
                                    f"sync: orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                                loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
                            elif self.args.lora_strategy.lower() == "nlora":
                                l1_loss = 0.
                                loranew_A_params = {}
                                loranew_B_params = {}
                                for name, param in self.model.named_parameters():
                                    if "loranew_A" in name:
                                        loranew_A_params[name.split("loranew_A")[0]] = param
                                    elif "loranew_B" in name:
                                        loranew_B_params[name.split("loranew_B")[0]] = param

                                for key in loranew_A_params:
                                    if key in loranew_B_params:
                                        l1_loss += torch.norm(
                                            torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

                                lamda_1 = self.args.olora_lamda_1

                                logger.debug(f"sync:Nlora_loss: {l1_loss.item():.4f};   accuracy_loss: {loss.item():.4f}; λ1: {lamda_1};")

                                loss = loss + l1_loss * lamda_1
                            elif self.args.lora_strategy.lower() == "inclora":
                                logger.debug(f"sync:inclora accuracy_loss: {loss.item():.4f}")
                            elif self.args.lora_strategy.lower() == "seqlora":
                                    logger.debug(f"no_sync:seqLora accuracy_loss: {loss.item():.4f}")
                            elif self.args.lora_strategy.lower() == "lora_l2":
                                l2_loss = 0.
                                for name, param in model.named_parameters():
                                    if "loranew_" in name:
                                        l2_loss += torch.norm(param, p=2)

                                logger.info(f"sync: lora_l2 l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f};  λ2: {lamda_2}")
                                lamda_2 = self.args.olora_lamda_2
                                loss = loss + l2_loss * lamda_2
                            ######################################################################
                            # PyTorch中,`backward()`方法
                            # 会计算当前张量（在这里是损失值`loss`）相对于各个需要梯度的参数的梯度,
                            # 并将这些梯度累积到参数的`.grad`属性中。
                            # 这是训练神经网络时的标准步骤,
                            # 为后续优化器更新参数（如 optimizer.step()）提供梯度信息
                            
                            logger.debug(f"sync: sum_loss: {loss.item():.4f}")
                            if gam_optimizer_active:
                                logger.debug("GAM active: skip immediate backward and defer to optimizer replay.")
                            elif self.do_grad_scaling:
                                self.scaler.scale(loss).backward()
                                logger.debug(f"梯度缩放")
                            elif self.use_apex:
                                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()
                                logger.debug(f"梯度缩放")
                            elif self.deepspeed:
                                # loss gets scaled under gradient_accumulation_steps in deepspeed
                                loss = self.deepspeed.backward(loss)
                                logger.debug(f"梯度缩放")
                            else:
                                loss.backward()
                                
                            # detach()`会创建一个新的张量,与原始张量共享数据,但剥离计算图,
                            # 即新张量不再有梯度历史。这意味着对这个新张量的操作不会影响反向传播,也不会在计算梯度时被追踪。
                            # 这一步通常用于将损失值从计算图中分离出来,
                            # 以便进行后续的数值记录或日志,而不会保留不必要的计算图信息,从而节省内存。
                            tr_loss_step = loss.detach()
                            loss_cleaned_log = tr_loss_step



                            # ------------------- 
                            # -------------------
                            # 如果使用 rwp 来进行 更新梯度,在这一步要计算扰动损失 也就是g1
                            # 并且扰动损失要和原来的损失要进行混合
                            if self.args.rwp_type != "":
                                # 这里只在 RWP 分支需要 clean 梯度与参数快照。
                                g0 = self._gather_grad_vector(model)
                                p_data_g0 = {n: p.data.clone() for n, p in model.named_parameters()}

                                # pass
                                # 这里是 RWP 的损失计算过程 
                                # g0 = self._gather_grad_vector(model)
                                logger.info(f"go-loss-cleaned: {loss_cleaned_log}")
                                
                                
                                # ----------------------------
                                # Step 2: 计算扰动梯度 g1
                                # ----------------------------
                                # Step 2: g₁ - 添加扰动并计算噪声梯度
                                # === Step 2: g₁ ===
                                # 清除梯度缓冲区,确保后续计算 g1 时梯度从零开始累积
                                model.zero_grad()
                                # logger.debug("不禁用batchsize")
                                # disable_running_stats(model) # 禁用 BatchNorm 的运行统计

                                # 为参数添加扰动
                                # rwp_noise_dict = {}
                               
                                for name, param in model.named_parameters():
                                    noise = None
                                    # 对新训练的lora_new 部分添加扰动

                                    if ("seqlora"  in self.args.lora_strategy) and ("lora" in name) :

                                        if "fisher" in self.args.rwp_noise_type and ("lora" in self.args.rwp_type or "full" in self.args.rwp_type):
                                            fisher_param = self.fisher_dict.get(name, None)
                                            if fisher_param is None and name.startswith("module."):
                                                # 去掉module.前缀再查一次
                                                fisher_param = self.fisher_dict.get(name[len("module."):], None)


                                            # ======== 日志输出部分 ========
                                            if fisher_param is not None:
                                                # logger.debug(f"[Fisher-Noise] Param: {name} | Fisher found | Shape: {tuple(fisher_param.shape)} | Mean: {fisher_param.float().mean().item():.4e} | Std: {fisher_param.float().std().item():.4e}")
                                                pass
                                            else:
                                                logger.debug(f"[Fisher-Noise] Param: {name} | Fisher NOT found (None)")
                                            # ======== 日志输出结束 ========
                                            noise = self.generate_noise(
                                                param=param,
                                                mode= self.args.rwp_noise_type,  # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                                fisher_param=fisher_param,
                                                fisher_scaler= self.args.rwp_fisher_app_scaler
                                                )
                                            # logger.debug(f"1----fisher_param {name} :loranew rwp_noise_type:{self.args.rwp_noise_type},fisher_appfornoise_scaler{self.args.rwp_fisher_app_scaler},std{self.args.noise_std}")
                                            # 如果是对 lora_new 部分添加Gauss 扰动
                                        elif "Gauss" in self.args.rwp_noise_type and ("lora" in self.args.rwp_type or "full" in self.args.rwp_type):
                                            noise = self.generate_noise(
                                                param=param,
                                                mode= self.args.rwp_noise_type,  # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                                )

                                    # 只对 lora_new 层  如果是inclora
                                    elif ("seqlora" not in self.args.lora_strategy) and ("loranew_" in name) :  
                                        # 如果是对 lora_new 部分添加 Fisher信息扰动
                                        if "fisher" in self.args.rwp_noise_type and ("lora" in self.args.rwp_type or "full" in self.args.rwp_type):
                                            fisher_param = self.fisher_dict.get(name, None)
                                            if fisher_param is None and name.startswith("module."):
                                                # 去掉module.前缀再查一次
                                                fisher_param = self.fisher_dict.get(name[len("module."):], None)


                                            # ======== 日志输出部分 ========
                                            if fisher_param is not None:
                                                # logger.debug(f"[Fisher-Noise] Param: {name} | Fisher found | Shape: {tuple(fisher_param.shape)} | Mean: {fisher_param.float().mean().item():.4e} | Std: {fisher_param.float().std().item():.4e}")
                                                pass
                                            else:
                                                logger.debug(f"[Fisher-Noise] Param: {name} | Fisher NOT found (None)")
                                            # ======== 日志输出结束 ========
                                            noise = self.generate_noise(
                                                param=param,
                                                mode= self.args.rwp_noise_type,  # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                                fisher_param=fisher_param,
                                                fisher_scaler= self.args.rwp_fisher_app_scaler
                                                )
                                            # logger.debug(f"1----fisher_param {name} :loranew rwp_noise_type:{self.args.rwp_noise_type},fisher_appfornoise_scaler{self.args.rwp_fisher_app_scaler},std{self.args.noise_std}")
                                            # 如果是对 lora_new 部分添加Gauss 扰动
                                        elif "Gauss" in self.args.rwp_noise_type and ("lora" in self.args.rwp_type or "full" in self.args.rwp_type):
                                            noise = self.generate_noise(
                                                param=param,
                                                mode= self.args.rwp_noise_type,  # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                                )
                                            # logger.debug(f"2----param name :loranew rwp_noise_type:{self.args.rwp_noise_type}")
                                    # 其它层（非 lora_new）
                                    elif "full" in self.args.rwp_type:
                                        # 不管是不是seqlora
                                        # 判断 如果是all-fisher 给剩余部分添加标准高斯噪声
                                        if ("full" in self.args.rwp_type) and ("fisher" in self.args.rwp_noise_type ):
                                            # full-fisher (非lora_new部分) 用高斯
                                            noise = self.generate_noise(
                                                param=param,
                                                mode= "Gauss_standard",  # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                            )
                                            # logger.debug(f"3----fisher_param {name}  rwp_noise_type:{self.args.rwp_noise_type},fisher_appfornoise_scaler{self.args.rwp_fisher_app_scaler},std{self.args.noise_std}")
                                        elif "Gauss" in self.args.rwp_noise_type:  # full-Gaussian (非lora_new部分) 用高斯
                                            noise = self.generate_noise(
                                                param=param,
                                                mode= self.args.rwp_noise_type, # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                            )
                                            # logger.debug(f"4----param name :{name} rwp_noise_type:{self.args.rwp_noise_type}")
                                    
                                    # 若未命中任何分支，不加扰动
                                    if noise is not None:
                                        param.data.add_(noise) 
                                        # rwp_noise_dict[name] = noise # 保存为每一层参数添加的噪声,用来还原模型


                                    
                                # forward + backward with noise → 得到 g₁
                                with self.compute_loss_context_manager():
                                    loss_noisy = self.compute_loss(model, inputs)

                                #-----------
                                logger.info(f'training_step RWP model with {self.args.rwp_noise_type} g1--loss_noise : {loss_noisy.item():.4f}')

                                # 添加正则化处理,如果需要的话
                                ########################### Regularization ##########################
                                if self.args.lora_strategy.lower() == "olora":
                                    orthogonal_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "lora_A" in name:
                                            for name_, param_ in model.named_parameters():
                                                if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                                    orthogonal_loss += torch.abs(
                                                        torch.mm(param, param_.T)).sum()
                                                    break

                                    l2_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "loranew_" in name:
                                            l2_loss += torch.norm(param, p=2)

                                    lamda_1 = self.args.olora_lamda_1
                                    lamda_2 = self.args.olora_lamda_2

                                    logger.info(
                                        f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; loss_noisy: {loss_noisy.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                                    loss_noisy = loss_noisy + orthogonal_loss * lamda_1 + l2_loss * lamda_2
                                elif self.args.lora_strategy.lower() == "nlora":
                                    l1_loss = 0.
                                    loranew_A_params = {}
                                    loranew_B_params = {}
                                    for name, param in self.model.named_parameters():
                                        if "loranew_A" in name:
                                            loranew_A_params[name.split("loranew_A")[0]] = param
                                        elif "loranew_B" in name:
                                            loranew_B_params[name.split("loranew_B")[0]] = param

                                    for key in loranew_A_params:
                                        if key in loranew_B_params:
                                            l1_loss += torch.norm(
                                                torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

                                    lamda_1 = self.args.olora_lamda_1

                                    logger.info(f"Nlora_loss: {l1_loss.item():.4f};   loss_noisy: {loss_noisy.item():.4f}; λ1: {lamda_1};")

                                    loss_noisy = loss_noisy + l1_loss * lamda_1
                                elif self.args.lora_strategy.lower() == "inclora":
                                    logger.debug(f"inclora loss_noisy: {loss_noisy.item():.4f}")
                                elif self.args.lora_strategy.lower() == "seqlora":
                                    logger.debug(f"no_sync:seqLora accuracy_loss: {loss.item():.4f}")
                                elif self.args.lora_strategy.lower() == "lora_l2":
                                    l2_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "loranew_" in name:
                                            l2_loss += torch.norm(param, p=2)

                                    logger.debug(f"lora_l2 l2_loss: {l2_loss.item():.4f}; loss_noisy: {loss_noisy.item():.4f};  λ2: {lamda_2}")
                                    lamda_2 = self.args.olora_lamda_2
                                    loss_noisy = loss_noisy + l2_loss * lamda_2
                                ######################################################################
                                logger.debug(f"g1---sum_loss(with some ): {loss_noisy.item():.4f}")

                                
                                #---多GPU,平均损失---------
                                if self.args.n_gpu > 1:
                                    loss_noisy = loss_noisy.mean()
                                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                    loss_noisy = loss_noisy / self.args.gradient_accumulation_steps

                                


                                # loss 的 backward
                                if self.do_grad_scaling:
                                    self.scaler.scale(loss_noisy).backward()
                                    logger.debug("梯度缩放")
                                elif self.use_apex:
                                    with amp.scale_loss(loss_noisy, self.optimizer) as scaled_loss:
                                        scaled_loss.backward()
                                    logger.debug("梯度缩放")
                                elif self.deepspeed:
                                    logger.debug("梯度缩放")
                                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                                    loss_noisy = self.deepspeed.backward(loss_noisy)
                                else:
                                    loss_noisy.backward()

                                loss_noisy_log = loss_noisy.detach()
                                logger.info(f"g1-loss_noisy: {loss_noisy_log.item():.4f}")

                                # 保存添加噪声扰动的模型的参数的fisher信息 
                                # 这里是根据 添加噪声扰动后的梯度计算得到的fisher信息矩阵
                                if 'fisher' in self.args.rwp_noise_type:
                                    with torch.no_grad():
                                        for name, param in self.model.named_parameters():
                                            if param.grad is not None:
                                                grad_squared = param.grad.detach() ** 2
                                                # 防止爆炸
                                                grad_squared = torch.nan_to_num(grad_squared, nan=0.0, posinf=0.0, neginf=0.0)


                                                # 使用指数滑动平均更新 fisher_dict
                                                self.fisher_dict[name] = (self.fisher_dict[name] *  self.args.rwp_fisher_cal_scaler +  grad_squared  )
                                                self.fisher_dict[name] = torch.nan_to_num(
                                                        self.fisher_dict[name], nan=1e-6, posinf=1e3, neginf=0.0
                                                )

                                                #同样更新当前要保留的Fisher信息
                                                self.fisher_dict_currentTask[name] = (self.fisher_dict_currentTask[name] *  self.args.rwp_fisher_cal_scaler + grad_squared  )


                                # 收集扰动梯度 g1
                                g1 = self._gather_grad_vector(model)
                                # g1_norm = torch.norm(g1)
                                # logger.info(f"[RWP] ||g1 (noisy)|| = {g1_norm.item():.4f}")


                                # # 假设 p_data_g0 已保存了 clean 分支每个参数的快照

                                with torch.no_grad():
                                    for name, param in model.named_parameters():
                                        param.data.copy_(p_data_g0[name])

                                # === Gradient mix ===
                                # Step 3: 梯度融合
                                # g ← a·g0 + b·g1
                                # 其中 g0 是没有扰动的, g1 是有扰动的 ,其对应的权重系数分别为  rwp_a,  rwp_b
                                loss_rwp  = self.args.rwp_a * loss_cleaned_log + self.args.rwp_b* loss_noisy_log
                                # logger.info(f"loss_rwp：{loss_rwp}；g0-cleaned_loss: {loss_cleaned_log.item():.4f}; g1-noisy_loss: {loss_noisy_log.item():.4f}; a :{self.args.rwp_a}, b :{self.args.rwp_b}")
                                # logger.info(f"{loss_rwp} = {self.args.rwp_a} * {loss_cleaned_log.item():.4f} + {self.args.rwp_b} *{loss_noisy_log.item():.4f} ")

                                tr_loss_step = loss_rwp
                                
                                # 进行梯度混合
                                mixed_grad = self.args.rwp_a * g0 + self.args.rwp_b * g1 

                                self._assign_grad_vector(model, mixed_grad)

                        elif self.args.train_method == "finetune":
                            # 修改进行梯度更新的地方,替换掉原来的函数调用
                            # -------------------
                            model.train()
                            # enable_running_stats(model)
                            inputs = self._prepare_inputs(inputs)

                            if is_sagemaker_mp_enabled():
                                loss_mb = smp_forward_backward(
                                    model, inputs, self.args.gradient_accumulation_steps)
                                return loss_mb.reduce_mean().detach().to(self.args.device)
                            
                            # self.print_trainable_parameters(model)
                            with self.compute_loss_context_manager():
                                loss = self.compute_loss(model, inputs)

                            logger.info(f'training_step compute_loss: {loss.item():.4f}')
                            

                            if self.args.n_gpu > 1:
                                loss = loss.mean()

                            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                loss = loss / self.args.gradient_accumulation_steps
                            

                            logger.debug(f"loss: {loss.item():.4f}")

                            if self.do_grad_scaling:
                                self.scaler.scale(loss).backward()
                                logger.debug("梯度缩放")
                            elif self.use_apex:
                                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()
                                logger.debug("梯度缩放")
                            elif self.deepspeed:
                                # loss gets scaled under gradient_accumulation_steps in deepspeed
                                loss = self.deepspeed.backward(loss)
                                logger.debug("梯度缩放")
                            else:
                                logger.debug("梯度更新")
                                loss.backward()

                            tr_loss_step = loss.detach()

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()


                # 真正进行梯度更新的地方
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    
                    # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                    # logger.debug(f'[Before Clipping] Total Grad Norm: {total_norm:.4f}')

                    # Gradient clipping
                    # 进行梯度裁剪
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed and not gam_optimizer_active:
                        self._clip_gradients_for_step(model)

                    
                    

                    # 
                    # total_norm_after = torch.norm(
                    #     torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
                    # )
                    # logger.debug(f'[After Clipping] Total Grad Norm: {total_norm_after:.4f}')

                    
                    
                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    # elif is_torch_tpu_available():
                    #     if self.do_grad_scaling:
                    #         self.scaler.step(self.optimizer)
                    #         self.scaler.update()
                    #     else:
                    #         xm.optimizer_step(self.optimizer)
                    # elif self.do_grad_scaling:
                    #     scale_before = self.scaler.get_scale()
                    #     self.scaler.step(self.optimizer)
                    #     self.scaler.update()
                    #     scale_after = self.scaler.get_scale()
                    #     optimizer_was_run = scale_before <= scale_after
                    # else:
                    #     self.optimizer.step()
                    # elif hasattr(self.optimizer, "first_step") and hasattr(self.optimizer, "second_step"):
                    elif 'sam' in self.args.optimizer_type:
                        # === For SAM Optimizer: first_step-second_step two phases ===
                        logger.debug(f"SAM优化器训练步骤开始,参数 rho={self.args.sam_rho}")
                        if not sam_accumulated_inputs:
                            raise RuntimeError("SAM accumulation window is empty before first_step().")

                        # First pass gradients were accumulated across the full window above.
                        logger.debug("执行SAM优化器的first_step()")
                        self.optimizer.first_step(zero_grad=True)

                        second_loss = self._run_sam_second_pass(model, sam_accumulated_inputs)
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            self._clip_gradients_for_step(model)
                        if second_loss is not None:
                            logger.debug(f'SAM accumulation second_loss_sum: {second_loss.item():.4f}')

                        if self._has_nan_gradients(model):
                            logger.error("检测到NaN梯度,这可能导致训练不稳定")

                        logger.debug("执行SAM优化器的second_step()")
                        self.optimizer.second_step(zero_grad=True)
                        enable_running_stats(model)
                        sam_accumulated_inputs.clear()

                        # # 计算参数更新量
                        # update_norms = {}
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         update_norms[name] = torch.norm(param.data - original_params[name]).item()
                        
                        # 记录最大和平均更新量
                        # if update_norms:
                        #     max_update = max(update_norms.values())
                        #     avg_update = sum(update_norms.values()) / len(update_norms)
                        #     logger.debug(f"参数更新统计: 最大更新量 {max_update:.8f}, 平均更新量 {avg_update:.8f}")
                        # else:
                        #     logger.debug("没有可更新的参数")
                    
                    
                    
                    elif 'gam' in self.args.optimizer_type:
                        # === GAM Optimizer: multi-step closure-based update ===
                        logger.debug("GAM优化器训练步骤开始")
                        if not gam_accumulated_inputs:
                            raise RuntimeError("GAM accumulation window is empty before optimizer replay.")

                        def gam_closure():
                            self.optimizer.zero_grad()
                            total_gam_loss = self._run_gam_replay_pass(model, gam_accumulated_inputs)
                            return None, total_gam_loss

                        _, gam_loss_value = self.optimizer.step(
                            closure=gam_closure,
                            clip_grad_norm=(
                                args.max_grad_norm
                                if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed
                                else None
                            ),
                        )
                        logger.debug(f"GAM step finished, loss={gam_loss_value.item() if torch.is_tensor(gam_loss_value) else gam_loss_value}")
                        gam_accumulated_inputs.clear()



                    elif 'rwp' in self.args.optimizer_type:
                        # === RWP Optimizer: two-pass clean + noisy update ===
                        logger.debug("RWP优化器训练步骤开始")

                        def rwp_closure():
                            self.optimizer.zero_grad()
                            model.train()
                            local_inputs = self._prepare_inputs(inputs)
                            with self.compute_loss_context_manager():
                                loss = self.compute_loss(model, local_inputs)
                            if self.args.n_gpu > 1:
                                loss = loss.mean()
                            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                loss = loss / self.args.gradient_accumulation_steps
                            if self.args.lora_strategy.lower() == "seqlora":
                                logger.debug(f"RWP seqLora loss: {loss.item():.4f}")
                            elif self.args.lora_strategy.lower() == "inclora":
                                logger.debug(f"RWP inclora loss: {loss.item():.4f}")
                            loss.backward()
                            return loss

                        rwp_loss_val = self.optimizer.step(closure=rwp_closure)
                        tr_loss_step = rwp_loss_val if torch.is_tensor(rwp_loss_val) else torch.tensor(rwp_loss_val)
                        logger.debug(f"RWP step finished, loss={rwp_loss_val.item() if torch.is_tensor(rwp_loss_val) else rwp_loss_val:.4f}")



                    else:
                    # === Standard Optimizer ===
                     # === Standard Optimizer ===
                        # logger.debug(f"执行标准优化器更新,优化器类型: {self.args.optimizer_type}")
                        # 保存原始参数副本用于对比
                        # original_params = {name: param.data.clone() for name, param in model.named_parameters()}

                        # 计算梯度范数用于分析
                        # total_grad_norm = 0
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         param_norm = param.grad.data.norm(2)
                        #         total_grad_norm += param_norm.item() ** 2
                        # total_grad_norm = total_grad_norm ** 0.5
                        # logger.debug(f"更新前梯度L2范数: {total_grad_norm:.8f}")



                        if is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                            # logger.debug(f"梯度缩放: 更新是否执行={optimizer_was_run}")
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                            # logger.debug(f"梯度缩放:  更新是否执行={optimizer_was_run}")
                        else:
                            self.optimizer.step()
                            logger.debug("执行标准优化器step")


                        # # 计算参数更新量
                        # update_norms = {}
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         update_norms[name] = torch.norm(param.data - original_params[name]).item()
                        
                        # # 记录最大和平均更新量
                        # if update_norms:
                        #     max_update = max(update_norms.values())
                        #     avg_update = sum(update_norms.values()) / len(update_norms)
                        #     logger.debug(f"参数更新统计: 最大更新量 {max_update:.8f}, 平均更新量 {avg_update:.8f}")
                        # else:
                        #     logger.debug("没有可更新的参数")
                        
                        # 检查是否存在NaN/inf参数
                        has_nan_param = False
                        for name, param in model.named_parameters():
                            if param.grad is not None and (torch.isnan(param.data).any() or torch.isinf(param.data).any()):
                                has_nan_param = True
                                logger.error(f"发现异常参数值: {name}")
                                break
                        if has_nan_param:
                            logger.error("检测到NaN/inf参数值,这可能导致训练不稳定")




                    

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()
                        # current_lr = self.lr_scheduler.get_last_lr()[0]
                        # logger.debug(f"学习率更新为: {current_lr:.8f}")

                    # 调用optimner更新梯度,结束之后进行梯度清除,
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        
        
        
        
        if "fisher" in self.args.rwp_noise_type and total_batched_samples > 0:

            # # 进行归一化处理
            # logger.info(f"Normalize Fisher with total_batched_samples = {total_batched_samples}")
            # for name in self.fisher_dict:
            #     self.fisher_dict[name] /= total_batched_samples
            #     self.fisher_dict[name] = torch.nan_to_num(self.fisher_dict[name], nan=1e-6, posinf=1e3, neginf=0.0)
            
            # for name in self.fisher_dict_currentTask:
            #     self.fisher_dict_currentTask[name] /= total_batched_samples
            #     self.fisher_dict_currentTask[name] = torch.nan_to_num(self.fisher_dict_currentTask[name], nan=1e-6, posinf=1e3, neginf=0.0)
            
            
            # 统计 Fisher 信息（每层均值、最大值等）
            fisher_stats = {}
                
            for name, fisher in self.fisher_dict.items():
                stats = {
                    "mean": float(fisher.mean().cpu().item()),
                    "max": float(fisher.max().cpu().item()),
                    "min": float(fisher.min().cpu().item()),
                    "std": float(fisher.std().cpu().item()),
                    "numel": fisher.numel(),
                }
                fisher_stats[name] = stats
                logger.info(f"Fisher[{name}]: mean={stats['mean']:.4e}, max={stats['max']:.4e}, min={stats['min']:.4e}, std={stats['std']:.4e}, numel={stats['numel']}")

            # 全局统计
            all_fisher = torch.cat([v.flatten().cpu() for v in self.fisher_dict.values()])
            global_stats = {
                "mean": float(all_fisher.mean().item()),
                "max": float(all_fisher.max().item()),
                "min": float(all_fisher.min().item()),
                "std": float(all_fisher.std().item()),
                "numel": int(all_fisher.numel()),
            }
            logger.info(f"Global Fisher: mean={global_stats['mean']:.4e}, max={global_stats['max']:.4e}, min={global_stats['min']:.4e}, std={global_stats['std']:.4e}, total_params={global_stats['numel']}")

            # 合并写入文件
            output_data = {
                "per_layer_stats": fisher_stats,
                "global_stats": global_stats,
            }
            stats_output_path = os.path.join(self.args.output_dir, "fisher_stats.json")
            with open(stats_output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            # 原始Fisher信息
            fisher_output_path = os.path.join(self.args.output_dir, "fisher_info.pth")
            torch.save(self.fisher_dict, fisher_output_path)
            logger.info(f"Fisher info saved to {fisher_output_path}")
            logger.info(f"Fisher statistics saved to {stats_output_path}")



            #-----------同样为本次任务也保留Fisher信息
            fisher_stats_currentTask = {}
            for name, fisher in self.fisher_dict_currentTask.items():
                stats = {
                    "mean": float(fisher.mean().cpu().item()),
                    "max": float(fisher.max().cpu().item()),
                    "min": float(fisher.min().cpu().item()),
                    "std": float(fisher.std().cpu().item()),
                    "numel": fisher.numel(),
                }
                fisher_stats_currentTask[name] = stats
            # 汇总全局统计
            all_fisher_currentTask = torch.cat([v.flatten().cpu() for v in self.fisher_dict_currentTask.values()])
            global_stats_currentTask = {
                "mean": float(all_fisher_currentTask.mean().item()),
                "max": float(all_fisher_currentTask.max().item()),
                "min": float(all_fisher_currentTask.min().item()),
                "std": float(all_fisher_currentTask.std().item()),
                "numel": int(all_fisher_currentTask.numel()),
            }

            # 合并写入 JSON
            output_data_CurentTask = {
                "per_layer_stats": fisher_stats_currentTask,
                "global_stats": global_stats_currentTask,
            }

            stats_output_path_CurentTask = os.path.join(self.args.output_dir, f"fisher_stats_CurentTask.json")
            with open(stats_output_path_CurentTask, "w") as f:
                json.dump(output_data_CurentTask, f, indent=2)

            # 保存原始 Fisher 信息张量
            fisher_output_path_CurentTask = os.path.join(self.args.output_dir, f"fisher_info_CurentTask.pth")
            torch.save(self.fisher_dict_currentTask, fisher_output_path_CurentTask)

            logger.info(f"Fisher info saved to {fisher_output_path_CurentTask}")
            logger.info(f"Fisher statistics saved to {stats_output_path_CurentTask}")


        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        自定义优化器与学习率调度器的创建,兼容SAM。
        """
        if self.optimizer is None:

            # self.scaler = torch.cuda.amp.GradScaler()
            # self.do_grad_scaling = True  # 控制标志
            # optim_name = self.args.optim.lower()  # <<<<<< 改这里,统一从optim读取
            if self.args.optimizer_type:
                optim_name = self.args.optimizer_type.lower()
            logger.debug(
                f'Class ContinualTrainer def create_optimizer_and_scheduler() optim_name: {optim_name}')
            if optim_name == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.args.learning_rate)
            elif optim_name == "sam-sgd":
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=torch.optim.SGD,
                    rho=self.args.sam_rho,
                    lr=self.args.learning_rate,
                )
            elif optim_name == "adamw_hf":
                from transformers import AdamW
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.args.learning_rate)
            elif optim_name == "sam-adamw_hf":
                from transformers import AdamW
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=AdamW,
                    rho=self.args.sam_rho,
                    lr=self.args.learning_rate,
                )
            elif optim_name == "adam":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.args.learning_rate)
            elif optim_name == "sam-adam":
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=torch.optim.Adam,
                    rho=self.args.sam_rho,
                    lr=self.args.learning_rate,
                )

            elif optim_name == "gam-adamw_hf":
                # GAM wrapped SGD optimizer
                from transformers import AdamW
                base_opt = torch.optim.AdamW(
                    self.model.parameters(), lr=self.args.learning_rate)
                # Pack GAM args from training arguments
                gam_args = SimpleNamespace(
                    grad_beta_1=float(getattr(self.args, "gam_grad_beta_1", 1.0)),
                    grad_beta_2=float(getattr(self.args, "gam_grad_beta_2", 1.0)),
                    grad_beta_3=float(getattr(self.args, "gam_grad_beta_3", 1.0)),
                    grad_gamma=float(getattr(self.args, "gam_grad_gamma", 0.1)),
                )
                self.optimizer = GAM(
                    self.model.parameters(),
                    base_optimizer=base_opt,
                    model=self.model,
                    grad_rho=float(getattr(self.args, "gam_grad_rho", 0.2)),
                    grad_norm_rho=float(getattr(self.args, "gam_grad_norm_rho", 0.2)),
                    adaptive=bool(getattr(self.args, "gam_adaptive", False)),
                    perturb_eps=float(getattr(self.args, "gam_perturb_eps", 1e-12)),
                    args=gam_args,
                    grad_reduce=str(getattr(self.args, "gam_grad_reduce", "mean")),
                )
            elif optim_name == "rwp-adamw_hf":
                # RWP wrapping AdamW — single-class convenience variant
                self.optimizer = RWPAdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    std=float(getattr(self.args, "noise_std", 0.01)),
                    rwp_lambda=float(getattr(self.args, "rwp_b", 0.5)),
                    rwp_range=str(getattr(self.args, "rwp_range", "lora")),
                    pertubation_mode=str(getattr(self.args, "rwp_noise_type", "Gauss_standard")),
                    rwp_eta=float(getattr(self.args, "rwp_fisher_app_scaler", 1.0)),
                )

            elif optim_name == "rwp-base-adamw_hf":
                # RWP wrapper-class variant — same logic, base_optimizer=AdamW
                from transformers import AdamW as HFAdamW
                self.optimizer = RWP(
                    self.model.parameters(),
                    base_optimizer=torch.optim.AdamW,
                    lr=self.args.learning_rate,
                    std=float(getattr(self.args, "noise_std", 0.01)),
                    rwp_lambda=float(getattr(self.args, "rwp_b", 0.5)),
                    rwp_range=str(getattr(self.args, "rwp_range", "lora")),
                    pertubation_mode=str(getattr(self.args, "rwp_noise_type", "Gauss_standard")),
                    rwp_eta=float(getattr(self.args, "rwp_fisher_app_scaler", 1.0)),
                )

            else:
                raise ValueError(f"Unsupported optimizer type: {optim_name}")

        # if self.lr_scheduler is None:
        if 'sam' in optim_name:
            from transformers.optimization import get_constant_schedule
            self.lr_scheduler = get_constant_schedule(
                self.optimizer.base_optimizer)
        elif 'gam' in optim_name:
            from transformers.optimization import get_constant_schedule
            self.lr_scheduler = get_constant_schedule(self.optimizer.base_optimizer)
        else:
            from transformers.optimization import get_constant_schedule
            self.lr_scheduler = get_constant_schedule(self.optimizer)

            # self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
    def _apply_train_regularization(self, model, loss, log_prefix=""):
        if self.args.train_method.lower() != "lora":
            return loss

        strategy = self.args.lora_strategy.lower()
        if strategy == "olora":
            orthogonal_loss = 0.
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    for name_, param_ in model.named_parameters():
                        if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                            orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum()
                            break

            l2_loss = 0.
            for name, param in model.named_parameters():
                if "loranew_" in name:
                    l2_loss += torch.norm(param, p=2)

            lamda_1 = self.args.olora_lamda_1
            lamda_2 = self.args.olora_lamda_2
            logger.info(
                f"{log_prefix}orthogonal_loss: {orthogonal_loss.item():.4f}; "
                f"l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; "
                f"λ1: {lamda_1}; λ2: {lamda_2}"
            )
            loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
        elif strategy == "nlora":
            l1_loss = 0.
            loranew_A_params = {}
            loranew_B_params = {}
            for name, param in self.model.named_parameters():
                if "loranew_A" in name:
                    loranew_A_params[name.split("loranew_A")[0]] = param
                elif "loranew_B" in name:
                    loranew_B_params[name.split("loranew_B")[0]] = param

            for key in loranew_A_params:
                if key in loranew_B_params:
                    l1_loss += torch.norm(torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

            lamda_1 = self.args.olora_lamda_1
            logger.info(f"{log_prefix}Nlora_loss: {l1_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1};")
            loss = loss + l1_loss * lamda_1
        elif strategy == "inclora":
            logger.debug(f"{log_prefix}inclora accuracy_loss: {loss.item():.4f}")
        elif strategy == "seqlora":
            logger.debug(f"{log_prefix}seqLora accuracy_loss: {loss.item():.4f}")
        elif strategy == "lora_l2":
            l2_loss = 0.
            for name, param in model.named_parameters():
                if "loranew_" in name:
                    l2_loss += torch.norm(param, p=2)

            lamda_2 = self.args.olora_lamda_2
            logger.info(f"{log_prefix}lora_l2 l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ2: {lamda_2}")
            loss = loss + l2_loss * lamda_2
        return loss

    def _compute_micro_batch_loss(self, model, inputs, log_prefix=""):
        model.train()
        local_inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, local_inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        loss = self._apply_train_regularization(model, loss, log_prefix=log_prefix)
        return loss, local_inputs

    def _backward_loss(self, loss):
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

    def _clip_gradients_for_step(self, model):
        args = self.args
        if args.max_grad_norm is None or args.max_grad_norm <= 0 or self.deepspeed:
            return

        if self.do_grad_scaling:
            if is_torch_tpu_available():
                logger.debug('[Gradient Clipping] TPU gradient scaling and reduction applied')
                gradients = xm._fetch_gradients(self.optimizer)
                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
            self.scaler.unscale_(self.optimizer)
            logger.debug('[Gradient Clipping] AMP unscaled gradients')

        if is_sagemaker_mp_enabled() and args.fp16:
            logger.debug('[Gradient Clipping] Using Sagemaker clip_master_grads')
            self.optimizer.clip_master_grads(args.max_grad_norm)
        elif hasattr(self.optimizer, "clip_grad_norm"):
            logger.debug('[Gradient Clipping] Using optimizer.clip_grad_norm')
            self.optimizer.clip_grad_norm(args.max_grad_norm)
        elif hasattr(model, "clip_grad_norm_"):
            logger.debug('[Gradient Clipping] Using model.clip_grad_norm_')
            model.clip_grad_norm_(args.max_grad_norm)
        else:
            nn.utils.clip_grad_norm_(
                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                args.max_grad_norm,
            )

    def _has_nan_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"发现NaN梯度: {name}")
                return True
        return False

    def _run_sam_second_pass(self, model, sam_micro_batches):
        if not sam_micro_batches:
            raise RuntimeError("SAM second pass requires the cached micro-batches from the current accumulation window.")

        use_no_sync = (
            self.args.local_rank != -1
            and self.args._no_sync_in_gradient_accumulation
            and hasattr(model, "no_sync")
        )
        total_second_loss = None
        disable_running_stats(model)

        for micro_step, micro_inputs in enumerate(sam_micro_batches):
            should_sync = (not use_no_sync) or (micro_step == len(sam_micro_batches) - 1)
            sync_context = nullcontext() if should_sync else model.no_sync()
            with sync_context:
                second_loss, local_inputs = self._compute_micro_batch_loss(
                    model,
                    micro_inputs,
                    log_prefix="sam-second: ",
                )
                total_second_loss = second_loss.detach() if total_second_loss is None else total_second_loss + second_loss.detach()
                self._backward_loss(second_loss)
                self.current_flos += float(self.floating_point_ops(local_inputs))

        return total_second_loss

    def _run_gam_replay_pass(self, model, gam_micro_batches):
        if not gam_micro_batches:
            raise RuntimeError("GAM replay requires the cached micro-batches from the current accumulation window.")

        total_gam_loss = None
        for micro_inputs in gam_micro_batches:
            micro_loss, local_inputs = self._compute_micro_batch_loss(
                model,
                micro_inputs,
                log_prefix="gam-replay: ",
            )
            total_gam_loss = micro_loss.detach() if total_gam_loss is None else total_gam_loss + micro_loss.detach()
            self._backward_loss(micro_loss)
            self.current_flos += float(self.floating_point_ops(local_inputs))

        return total_gam_loss

    def print_trainable_parameters(self,model):
        total_params = 0
        trainable_params = 0
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"✅ {name}: {param.shape}")
            # else:
            #     print(f"❌ {name}: frozen")

        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%\n")


    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     自定义训练步骤,兼容SAM优化器和LoRA正则。
    #     """
    #     model.train()
    #     enable_running_stats(model)
    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(
    #             model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
        
    #     # self.print_trainable_parameters(model)
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     logger.info(f'training_step compute_loss: {loss.item():.4f}')
        

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()

    #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #         loss = loss / self.args.gradient_accumulation_steps

    #     ########################### Regularization ##########################
    #     orthogonal_loss = 0.
    #     for name, param in model.named_parameters():
    #         if "lora_A" in name:
    #             for name_, param_ in model.named_parameters():
    #                 if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
    #                     orthogonal_loss += torch.abs(
    #                         torch.mm(param, param_.T)).sum()
    #                     break

    #     l2_loss = 0.
    #     for name, param in model.named_parameters():
    #         if "loranew_" in name:
    #             l2_loss += torch.norm(param, p=2)

    #     lamda_1 = self.args.olora_lamda_1
    #     lamda_2 = self.args.olora_lamda_2

    #     logger.info(
    #         f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

    #     loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2



        

    #     # 先分支处理 backward

    #     logger.debug(f'Class ContinualTrainer def training_step_sam2step()  backward')
    #     if self.do_grad_scaling:
    #         logger.debug(
    #             f'Class ContinualTrainer def training_step_()  update optim  self.do_grad_scaling{self.do_grad_scaling}')
    #         self.scaler.scale(loss).backward()
            
    #         #然后进行梯度裁剪 和SAM 优化器更新
    #         if is_torch_tpu_available():
    #             gradients = xm._fetch_gradients(self.optimizer)
    #             xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    #         self.scaler.unscale_(self.optimizer)

           



    #     elif self.use_apex: 
    #         # 这里的 self.use_apex 是指是否使用 apex 进行混合精度训练
    #         logger.debug(
    #             f'Class ContinualTrainer def training_step_()  update optim  self.use_apex{self.use_apex}')
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         logger.debug(
    #             f'Class ContinualTrainer def training_step_sam2step()  update optim self.deepspeed')
    #         loss = self.deepspeed.backward(loss)
    #     else:

    #         logger.debug(
    #             f'Class ContinualTrainer def training_step  update optim else:  {self.args.optimizer_type.lower()}')
    #         loss.backward()

           

    #         if 'sam' in self.args.optimizer_type.lower():
    #             logger.debug( f'Class ContinualTrainer def training_step_sam2step()  update optim else:  rho :{self.args.sam_rho}')
    #             # ==SAM== 包装一步更新

    #             # def closure():
    #             #     # 注意：此处 zero_grad 是必要的,因为 optimizer.step(closure) 预期 closure 内部自行清理梯度
    #             #     self.optimizer.zero_grad()

    #             #     outputs = model(**inputs)
    #             #     loss = outputs.loss

    #             #     # 多卡时取平均
    #             #     if self.args.n_gpu > 1:
    #             #         loss = loss.mean()

    #             #     # 如果有梯度累积,除以累积步数
    #             #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #             #         loss = loss / self.args.gradient_accumulation_steps

    #             #     # === 追加正则项：orthogonal loss + L2正则 ===
    #             #     orthogonal_loss = 0.0
    #             #     for name, param in model.named_parameters():
    #             #         if "lora_A" in name:
    #             #             for name_, param_ in model.named_parameters():
    #             #                 if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
    #             #                     orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum()
    #             #                     break

    #             #     l2_loss = 0.0
    #             #     for name, param in model.named_parameters():
    #             #         if "loranew_" in name:
    #             #             l2_loss += torch.norm(param, p=2)

    #             #     lamda_1 = getattr(self.args, "lamda_1", 0.5)
    #             #     lamda_2 = getattr(self.args, "lamda_2", 0.0)

    #             #     # 加到总loss中
    #             #     loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2

    #             #     # 标准 backward
    #             #     loss.backward()

    #             #     return loss

    #             # # === 优化器执行 closure ===
    #             # # ⚡启用 BN层 running_stats（SAM要求）
    #             # enable_running_stats(model)

    #             # # ⚡先手动执行一次 closure,保证第一次 forward/backward,累积梯度
    #             # loss = closure()

    #             # # ⚡由 optimizer 完成 first_step -> second closure -> second_step
    #             # self.optimizer.step(closure=closure)

    #             # # ⚡清空梯度
    #             # self.optimizer.zero_grad()

    #             # === SAM两步更新 ===
    #             if self.args.gradient_accumulation_steps > 1 and hasattr(model, "no_sync"):
    #                 with model.no_sync():
    #                     loss.backward()
    #             else:
    #                 loss.backward()

    #             self.optimizer.first_step(zero_grad=True)

    #             # second forward-backward
    #             disable_running_stats(model)
    #             with self.compute_loss_context_manager():
    #                 second_loss = self.compute_loss(model, inputs)

    #             if self.args.n_gpu > 1:
    #                 second_loss = second_loss.mean()
    #             if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #                 second_loss = second_loss / self.args.gradient_accumulation_steps

    #             orthogonal_loss2 = 0.
    #             for name, param in model.named_parameters():
    #                 if "lora_A" in name: 
    #                     for name_, param_ in model.named_parameters(): 
    #                         if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]: 
    #                             orthogonal_loss2 += torch.abs(
    #                                 torch.mm(param, param_.T)).sum()
    #                             break

    #             l2_loss2 = 0. 
    #             for name, param in model.named_parameters(): 
    #                 if "loranew_" in name: 
    #                     l2_loss2 += torch.norm(param, p=2) 

    #             logger.info( f"SAM second loss orthogonal_loss: {orthogonal_loss2.item():.4f}; l2_loss: {l2_loss2.item():.4f}; accuracy_loss: {second_loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}") 

    #             second_loss = second_loss + orthogonal_loss2 * lamda_1 + l2_loss2 * lamda_2 

    #             logger.info(f'SAM training_step second_loss_sum: {second_loss.item():.4f}')

    #             if self.args.gradient_accumulation_steps > 1 and hasattr(model, "no_sync"): 
    #                 with model.no_sync(): 
    #                     second_loss.backward() 
    #             else: 
    #                 second_loss.backward() 
    #             self.optimizer.second_step(zero_grad=True) 
 
    #         else: 
    #             loss.backward() 
    #             self.optimizer.step() 
    #             self.optimizer.zero_grad() 
        
    #     return loss.detach()
                
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        兼容 HuggingFace Trainer 默认损失逻辑 + Linear Probe 模式下特征提取 & 分类头损失。

        Linear probe 模式通过参数 `args.use_probe` 控制：
        - False：默认走原始 T5 损失（seq2seq / causal loss）
        - True：替换为 probe_head 分类器损失（线性分类头）

        还支持多种特征抽取方式（cls/eos/mean）通过 `args.probe_feature_mode` 控制。
        """
        # --- 默认 HuggingFace 损失逻辑（非 probe 模式） ---
        if not getattr(self.args, "use_probe", False):
            # 🔁 调用父类默认实现
            # 默认路径,使用 HF 的 loss 机制（适配 label_smoother、AMP、多卡等）
            # logger.debug(f" run father loss")
            return super().compute_loss(model, inputs, return_outputs)
        
        else:

            # --- Linear Probe 模式 ---
            # real_model = unwrap_model(model)

            # 🔥 注意！！现在不需要自己取 hidden_state 手动处理了
            # 只需要 forward 返回 logits（在 base_model 的 forward 里已经加了 probe_head）
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                decoder_input_ids=inputs.get("decoder_input_ids", None),
                decoder_attention_mask=inputs.get("decoder_attention_mask", None),
                output_hidden_states=True,  # <<< ✨✨这里必须加✨✨
                return_dict=True,
            )

            logits = outputs.logits  # 🔥 直接拿 logits,不需要手动 probe_head(features)
            class_labels = inputs["class_labels"]  # 🔥 直接用 class_labels

            # CrossEntropy Loss
            loss = F.cross_entropy(logits, class_labels.view(-1))

            return (loss, {"logits": logits, "labels": class_labels}) if return_outputs else loss
                # # 🔥 核心：Linear Probe 使用 features + probe_head 分类
            # real_model =  unwrap_model(model)  # ✅ 解包,拿原始 model

            # # logger.debug(f" run Linear Probe loss")
            # # # === Linear Probe 模式 ===
            # # labels = inputs["labels"]
            # # decoder_input_ids = inputs.get("decoder_input_ids", None)
            # # --- 取输入 ---
            # class_labels = inputs["class_labels"]  # ✅ 使用整数编号
            # decoder_input_ids = inputs.get("decoder_input_ids", None)

            # # Forward 获取 encoder/decoder 表征
            # outputs = model(
            #     input_ids=inputs["input_ids"],
            #     attention_mask=inputs.get("attention_mask", None),
            #     decoder_input_ids=decoder_input_ids,
            #     output_hidden_states=True,
            #     return_dict=True
            # )

            # decoder_hidden_states = outputs.decoder_hidden_states[-1]  # (B, T, H)
            # decoder_attention_mask = inputs.get("decoder_attention_mask", None)

            # # --- 特征提取策略 ---
            # mode = getattr(self.args, "probe_feature_mode", "cls")

            # if mode == "cls":
            #     # 使用 decoder 的第一个 token 的 embedding（常用于 T5 等）
            #     features = decoder_hidden_states[:, 0]  # (B, H)

            # elif mode == "eos":
            #     # 使用 decoder 中 EOS token 的 embedding
            #     eos_token_id = getattr(model.config, "eos_token_id", 1)
            #     eos_mask = (decoder_input_ids == eos_token_id)  # (B, T)
            #     if eos_mask.sum(dim=1).min() == 0:
            #         raise ValueError("Some sequences do not contain an EOS token.")
            #     eos_idx = eos_mask.float().cumsum(dim=1).argmax(dim=1)  # (B,)
            #     features = decoder_hidden_states[torch.arange(decoder_hidden_states.size(0)), eos_idx]  # (B, H)

            # elif mode == "mean":
            #     # 使用 decoder 的非 padding token 平均
            #     if decoder_attention_mask is None:
            #         raise ValueError("decoder_attention_mask is required for mean pooling.")
            #     mask = decoder_attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            #     summed = (decoder_hidden_states * mask).sum(dim=1)  # (B, H)
            #     count = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
            #     features = summed / count  # (B, H)

            # else:
            #     raise ValueError(f"Unsupported probe_feature_mode: {mode}")

            # # --- Linear Probe 分类头 ---
            # # probe_head = getattr(model, "probe_head", None)
            # # if probe_head is None:
            # #     probe_head = getattr(model.base_model.model, "probe_head", None)
            # # if probe_head is None:
            # #     raise ValueError("Model does not have `probe_head` defined.")

            # # logits = model.probe_head(features)  # (B, num_classes)
            # # 🔥 核心改动：用 unwrapped model 的 probe_head
            # # logits = real_model.probe_head(features)
            # # loss = F.cross_entropy(logits, labels)
            # # loss = F.cross_entropy(logits, labels.view(-1))  # ✅ 处理 labels shape

            
            # # # ✅ 新增：处理 labels 是文本字符串的情况
            # # if isinstance(labels[0], str):
            # #     unique_labels = sorted(list(set(labels)))
            # #     label2id = {label: idx for idx, label in enumerate(unique_labels)}
            # #     labels = torch.tensor([label2id[label] for label in labels], dtype=torch.long, device=logits.device)

            # # elif isinstance(labels, list):
            # #     # 如果已经是 list of int
            # #     labels = torch.tensor(labels, dtype=torch.long, device=logits.device)

            # # elif isinstance(labels, torch.Tensor):
            # #     labels = labels.to(device=logits.device, dtype=torch.long)
            # #     if labels.dim() == 2:
            # #         labels = labels[:, 0]  # 取每条 label 的第一个 token

            # # else:
            # #     raise ValueError(f"Unsupported labels type: {type(labels)}")
            
            # # # 🧹 保证 labels shape 正确,cross_entropy 要求 (B,) 而不是 (B, 1)
            # # labels = labels.view(-1)
            # # loss = F.cross_entropy(logits, labels)
            # # --- CrossEntropy Loss ---

            #  # --- Probe Head 分类 ---
            # logits = real_model.probe_head(features)  # (B, num_classes)

            # # 🚀 计算 CrossEntropy Loss
            # loss = F.cross_entropy(logits, class_labels.view(-1))  # ✅ 注意 shape


            # return (loss, {"logits": logits, "labels": class_labels}) if return_outputs else loss


    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #         # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
    #         loss = loss / self.args.gradient_accumulation_steps

    #     ########################### Regularization ##########################
    #     orthogonal_loss = 0.
    #     for name, param in self.model.named_parameters():
    #         if "lora_A" in name:
    #             for name_, param_ in self.model.named_parameters():
    #                 if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
    #                     orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
    #                     break # target modules have been matched

    #     # l2-normalization for loranew_A/B
    #     l2_loss = 0.
    #     for name, param in self.model.named_parameters():
    #         if "loranew_" in name:
    #             l2_loss += torch.norm(param, p=2)

    #     lamda_1 = self.args.olora_lamda_1
    #     lamda_2 = self.args.olora_lamda_2

    #     logger.info(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
    #     loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
    #     ######################################################################

    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         # loss gets scaled under gradient_accumulation_steps in deepspeed
    #         loss = self.deepspeed.backward(loss)
    #     else:
    #         loss.backward()

    #     return loss.detach()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        logger.debug(
            f"调用自定义函数 Class ContinualTrainer def evaluation_loop self.args.predict_with_generate: {self.args.predict_with_generate},  prediction_loss_only: { prediction_loss_only},ignore_keys: {ignore_keys}"
        )


        # if eval is called w/o train init deepspeed here
        if self._is_deepspeed_enabled_compat(args) and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None,  # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        # logger.debug(
        #     f"调用自定义函数 Class ContinualTrainer def prediction_step() self.args.predict_with_generate: {self.args.predict_with_generate},  prediction_loss_only: { prediction_loss_only},ignore_keys: {ignore_keys}"
        # )

        # 🔵 处理 Linear Probe 模式
        if getattr(self.args, "use_probe", False) :
           inputs = self._prepare_inputs(inputs)

           with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    decoder_input_ids=inputs.get("decoder_input_ids", None),
                    decoder_attention_mask=inputs.get("decoder_attention_mask", None),
                    output_hidden_states=True,
                    return_dict=True,
                )

                logits = outputs.logits  # 🔥 直接拿 logits
                class_labels = inputs["class_labels"]  # 🔥 使用 class_labels

                loss = F.cross_entropy(logits, class_labels.view(-1))  # 🔥 注意 view(-1)

                if prediction_loss_only:
                    return (loss, None, None)
                else:
                    return (loss, logits, class_labels)

       
        # 若未启用生成式预测（如分类任务）,或者只需要 loss（如只计算验证损失）,就用原始 Trainer 的 prediction_step
        if not self.args.predict_with_generate or prediction_loss_only:
            # logger.debug(f"调用父类 prediction_step()")
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        # 某些模型（如 encoder-decoder）encoder.main_input_name 不一定是 'input_ids',因此进行兼容性处理。
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:

            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        
        # 生成式预测时,labels 作为 decoder 的输入 
        # 使用 .generate() 来生成预测文本序列
        generated_tokens = self.model.generate(
            input_ids=generation_inputs,
            generation_config=generation_config
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, max_length)

        with torch.no_grad():
            # 如果有标签,则正向推理一遍算 loss
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        # 若只需要 loss,就不返回预测结果。
        if self.args.prediction_loss_only:
            return (loss, None, None)
        # 否则返回三元组：
        # loss：用于评估损失
        # generated_tokens：用于 decode 出预测文本
        # labels：ground truth 标签
        

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     logger.debug(
    #         f"调用自定义函数 Class ContinualTrainer def _inner_training_loop() batch_size: {batch_size}, args: {args}, resume_from_checkpoint: {resume_from_checkpoint}, trial: {trial}, ignore_keys_for_eval: {ignore_keys_for_eval}"
    #     )
    #     self._train_batch_size = batch_size
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = args.train_batch_size * \
    #         args.gradient_accumulation_steps * args.world_size

    #     len_dataloader = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(
    #                 args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(
    #                 train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torch.distributed.launch)."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = (
    #         self.sharded_ddp is not None
    #         and self.sharded_ddp != ShardedDDPOption.SIMPLE
    #         or is_sagemaker_mp_enabled()
    #         or self.fsdp is not None
    #     )
    #     if args.deepspeed:

    #         # self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #         # # 然后交给 deepspeed,但不让它覆盖 optimizer
    #         # deepspeed_engine, _, _ = deepspeed_init(
    #         #     self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         # )

    #         # self.model = deepspeed_engine.module
    #         # self.model_wrapped = deepspeed_engine
    #         # self.deepspeed = deepspeed_engine

    #         # ---- 原来的设置
    #         deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #         # $ ---- 原来的设置

    #     elif not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable()

    #     model = self._wrap_model(self.model_wrapped)

    #     if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
    #         self._load_from_checkpoint(resume_from_checkpoint, model)

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(
    #         f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
    #     logger.info(
    #         f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(
    #         f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(
    #         f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(
    #             os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (
    #                 num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info(
    #             "  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(
    #             f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             if skip_first_batches is None:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
    #                     " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
    #                     " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
    #                     " training on data already seen by your model."
    #                 )
    #             else:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch."
    #                 )
    #             if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
    #                 steps_trained_progress_bar = tqdm(
    #                     total=steps_trained_in_current_epoch)
    #                 steps_trained_progress_bar.set_description(
    #                     "Skipping the first batches")

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(
    #         args, self.state, self.control)

    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
    #                 train_dataloader.sampler, RandomSampler
    #             )
    #             if is_torch_less_than_1_11 or not is_random_sampler:
    #                 # We just need to begin an iteration to create the randomization of the sampler.
    #                 # That was before PyTorch 1.11 however...
    #                 for _ in train_dataloader:
    #                     break
    #             else:
    #                 # Otherwise we need to call the whooooole sampler cause there is some random operation added
    #                 # AT THE VERY END!
    #                 _ = list(train_dataloader.sampler)

    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #         elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
    #             train_dataloader.dataset.set_epoch(epoch)

    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(
    #                 train_dataloader, [args.device]).per_device_loader(args.device)
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_iterator)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(
    #             args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
    #             epoch_iterator = skip_first_batches(
    #                 epoch_iterator, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #             if step == 0:
    #                 logger.info("===== Sample input batch =====")
    #                 for k, v in inputs.items():
    #                     if isinstance(v, torch.Tensor):
    #                         logger.info(f"{k}: shape={v.shape}, dtype={v.dtype}")
    #                         logger.info(f"{k} sample values: {v[0][:10]}")
    #                     else:
    #                         logger.info(f"{k}: type={type(v)}")



    #             total_batched_samples += 1
    #             if rng_to_sync:
    #                 self._load_rng_state(resume_from_checkpoint)
    #                 rng_to_sync = False

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None

    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(
    #                     args, self.state, self.control)

    #             if (
    #                 (total_batched_samples %
    #                  args.gradient_accumulation_steps != 0)
    #                 and args.local_rank != -1
    #                 and args._no_sync_in_gradient_accumulation
    #             ):
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     tr_loss_step = self.training_step(model, inputs)
    #             else:
    #                 tr_loss_step = self.training_step(model, inputs)

    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_tpu_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / \
    #                     (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 tr_loss += tr_loss_step

    #             self.current_flos += float(self.floating_point_ops(inputs))

    #             # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
    #             if self.deepspeed:
    #                 self.deepspeed.step()

    #             if total_batched_samples % args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping

    #                     if self.do_grad_scaling:
    #                         # Reduce gradients first for XLA
    #                         if is_torch_tpu_available():
    #                             gradients = xm._fetch_gradients(self.optimizer)
    #                             xm.all_reduce("sum", gradients,
    #                                           scale=1.0 / xm.xrt_world_size())
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)

    #                     if is_sagemaker_mp_enabled() and args.fp16:
    #                         self.optimizer.clip_master_grads(
    #                             args.max_grad_norm)
    #                     elif hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(args.max_grad_norm)
    #                     elif hasattr(model, "clip_grad_norm_"):
    #                         # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #                         model.clip_grad_norm_(args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         nn.utils.clip_grad_norm_(
    #                             amp.master_params(
    #                                 self.optimizer) if self.use_apex else model.parameters(),
    #                             args.max_grad_norm,
    #                         )

    #                 # Optimizer step
    #                 optimizer_was_run = True
    #                 # if self.deepspeed:
    #                 #     pass  # called outside the loop
    #                 # elif is_torch_tpu_available():
    #                 #     if self.do_grad_scaling:
    #                 #         self.scaler.step(self.optimizer)
    #                 #         self.scaler.update()
    #                 #     else:
    #                 #         xm.optimizer_step(self.optimizer)
    #                 # elif self.do_grad_scaling:
    #                 #     scale_before = self.scaler.get_scale()
    #                 #     self.scaler.step(self.optimizer)
    #                 #     self.scaler.update()
    #                 #     scale_after = self.scaler.get_scale()
    #                 #     optimizer_was_run = scale_before <= scale_after
    #                 # else:
    #                 #     self.optimizer.step()

    #                 if optimizer_was_run and not self.deepspeed:
    #                     self.lr_scheduler.step()

    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + \
    #                     (step + 1 + steps_skipped) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(
    #                     args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(
    #                     tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(
    #                     args, self.state, self.control)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(
    #             args, self.state, self.control)
    #         self._maybe_log_save_evaluate(
    #             tr_loss, model, trial, epoch, ignore_keys_for_eval)

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info(
    #         "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sur the model has been saved by process 0.
    #         if is_torch_tpu_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.local_rank != -1:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step

    #     metrics = speed_metrics(
    #         "train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(
    #         use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if checkpoint != self.state.best_model_checkpoint:
    #                 logger.info(
    #                     f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint)

    #     self.control = self.callback_handler.on_train_end(
    #         args, self.state, self.control)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)

    def _compute_stats(self, eigenvalues):
        """改进点13：统一统计计算"""
        if not eigenvalues:
            return {"min": 0., "max": 0., "median": 0., "mean": 0.}

        arr = np.array(eigenvalues)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }

    def compute_loss_landscape(
        self, eval_dataset: Dataset, output_dir, flat_eval_type = 'originModel',x_range=(-1, 1), y_range=(-1, 1), num_points=10, max_batches=1, sample_batches=False,

    ):
        """
        计算损失景观,并分别计算:
        1️⃣ **完整模型（Base Model + LoRA Adapter）** 的损失景观
        2️⃣ **仅 LoRA 适配器（LoRA Adapter）** 的损失景观
        计算损失景观,并分别计算完整模型 & LoRA Adapter 的损失表面。
        针对大模型使用多 GPU,可以并行分配 (i, j) 坐标网格。
        """
        args = self.args
        device = args.device
        # 最终损失网络的数值
        loss_grid = np.zeros((num_points, num_points))

        # 根据模型判断,这里不妨使用手动
        # 是否是训练后的模型,还是加载已经训练好的模型
        if args.do_train:
            # 如果进行训练,那么模型的new_lora就是新的当前任务添加的lora
            # Flag_newtaskLoRA = ''  # 如果只干扰训练后的新的任务的lora
            # Flag_onlyLoRA = ''  # 如果只干扰lora 部分
            # Flag_fullModel =  'fullModel'  # 如果干扰模型的所有参数
            surf_file = os.path.join(
                    output_dir, f"{args.lora_strategy}_{flat_eval_type}_T5large_testData.h5")
        else:
            # 如果不进行训练,只展示加载模型的 lossland ,则对模型的所有参数(不包括new_lora)都进行干扰,
            surf_file = os.path.join(output_dir, f"{flat_eval_type}_T5large__testData.h5")


        # 确保目标目录存在（若不存在则创建）
        if not os.path.exists(surf_file):
            with open(surf_file, 'w') as f:
                pass  # 成功创建空文件
        os.makedirs(os.path.dirname(surf_file), exist_ok=True)  # 保证父目录存在

        # ✅ 记录日志信息
        logger.info(f'***5***--5-2 **1 compute_loss_landscape  ')
        logger.info(
            f"***** Running Loss Landscape Calculati on {flat_eval_type}*****")
        logger.info(
            f"Output Dir = {output_dir},Output File :{surf_file} Num points = {num_points}x{num_points} max_batches = {max_batches}")

        # ✅ 兼容 AMP 和分布式训练
        # 复制模型避免污染
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)

        model = model.to(device=device)
        model.eval()  # 确保模型在 eval 模式

        # --------------------- 生成参数扰动阶段阶段 ---------------------
        # 确定需要扰动的参数名称
        # 根据不同的微调方法（Nlora或lora）确定需要保存原始值的参数
        original_params_to_perturb = {}
        for name, param in model.named_parameters():
            if args.do_train:
                # if distrub_name == 'originModel':
                pass
            else:# 不是训练后直接评估,加载的模型不应该包括新添加的 new_lora 部分
                # if flat_eval_type == 'originModel':
                #     if "loranew_" not in name : 
                #         original_params_to_perturb[name] = param.data.clone()
                # elif flat_eval_type == 'trainedLoRA':
                #     if "lora_" in name  and "loranew_" not in name: 
                #         original_params_to_perturb[name] = param.data.clone()
                
                if param.requires_grad: # 当扰动整个模型全部都是require ,扰动只有lora则只有训练好的lora
                    original_params_to_perturb[name] = param.data.clone()
                    param_names_to_perturb.append(name)



        # --------------------- 扰动生成优化 ---------------------
        torch.manual_seed(42)  # 固定随机种子保证可重复性
        perturb_x, perturb_y = {}, {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in original_params_to_perturb:
                    continue

                # 直接生成归一化扰动（优化点2）
                # norm_factor = torch.norm(param) + 1e-8  # 计算神经网络参数的尺度
                # torch.manual_seed(seed_x)
                # d_x_perturb = torch.randn_like(param).to(device)  # 生成随机扰动
                # perturb_x[name] = (d_x_perturb / torch.norm(d_x_perturb)) * norm_factor  # 归一化并调整尺度
                else:
                    d_x = torch.randn_like(param)
                    d_x = (d_x / d_x.norm()) * (param.norm() + 1e-8)  # 直接归一化
                    perturb_x[name] = d_x.to(device)

                    # 生成正交扰动
                    d_y = torch.randn_like(param)
                    d_y = d_y - torch.sum(d_y * d_x) * \
                        d_x / (d_x.norm()**2)  # 施密特正交化
                    d_y = (d_y / d_y.norm()) * (param.norm() + 1e-8)     # 归一化
                    perturb_y[name] = d_y.to(device)
        # 使用FP16存储扰动（优化点5）
        if args.fp16_full_eval or args.bf16_full_eval:
            perturb_x = {k: v.half() for k, v in perturb_x.items()}
            perturb_y = {k: v.half() for k, v in perturb_y.items()}

        # --------------------- 并行化网格计算（优化点3）---------------------
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        y_coords = np.linspace(y_range[0], y_range[1], num_points)

        # 分布式任务划分
        world_size = 1
        rank = 0
        start_idx, end_idx = 0, num_points
        logger.info(
            f'***5***--5-2**4 distribute--no world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')

        # --------------------- 数据准备阶段 ---------------------
        logger.info(
            f'***5***--5-2**2 begin load data---{torch.distributed.is_initialized()} ')
        dataloader = self.get_eval_dataloader(eval_dataset)
        all_batches = list(dataloader)[:max_batches]
        logger.info(f'***5***--5-2**2 begin load data---{len(all_batches)} ')

        # 合并所有批次为单个大批次（显存允许时）
        if not sample_batches and len(all_batches) > 0:
            try:
                # 尝试合并所有小批次为一个大批次
                big_batch = {
                    # 对每个特征键（如input_ids、labels）进行纵向拼接
                    k: torch.cat([b[k] for b in all_batches], dim=0)
                    for k in all_batches[0].keys()  # 假设所有批次结构相同
                }
                # 用合并后的大批次替换原始批次列表
                all_batches = [big_batch]  # 现在只包含一个合并后的批次
            except RuntimeError:
                # 显存不足时回退到原始小批次
                logger.warning("无法合并批次,保持原有批次数量")

        # --------------------- 主计算循环优化 ---------------------

        for i in tqdm(range(start_idx, end_idx), desc=f"Rank {rank} Processing"):
            xv = x_coords[i]
            for j, yv in enumerate(y_coords):
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_perturb:
                    model.state_dict()[name].copy_(
                        original_params_to_perturb[name])

                # 应用扰动
                with torch.no_grad():
                    for name in original_params_to_perturb:
                        param = model.state_dict()[name]
                        delta = xv * \
                            perturb_x[name].to(
                                param.dtype) + yv * perturb_y[name].to(param.dtype)
                        param.add_(delta)  # 原位操作减少内存分配


                # 计算损失 (统一使用 prediction_step)
                total_loss = 0.0
                for batch in all_batches:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    loss, _, _ = self.prediction_step(
                        model, 
                        inputs, 
                        prediction_loss_only=True, 
                        ignore_keys=None
                    )
                    # 兼容返回为张量/float
                    total_loss += loss.item() if hasattr(loss, "item") else float(loss)
                loss_grid[i, j] = total_loss / len(all_batches)

                # # 计算损失
                # total_loss = 0.0
                # for batch in all_batches:
                #     inputs = {k: v.to(device) for k, v in batch.items()}
                #     with torch.cuda.amp.autocast(enabled=args.fp16):  # 支持混合精度
                #         outputs = model(**inputs)
                #         loss = F.cross_entropy(
                #             outputs.logits.view(-1, outputs.logits.size(-1)),
                #             inputs["labels"].view(-1)
                #         )
                #     total_loss += loss.item()

                # loss_grid[i, j] = total_loss / len(all_batches)





                # 每5次迭代清理一次缓存（优化点6）
                if j % 5 == 0:
                    torch.cuda.empty_cache()

        # --------------------- 分布式结果收集 ---------------------
        if torch.distributed.is_initialized():
            # 收集所有进程的loss_grid
            # 将 loss_grid 转换为张量
            loss_grid_tensor = torch.tensor(loss_grid, device=device)
            # 创建一个列表,用于接收所有进程的 loss_grid
            all_loss = [torch.zeros_like(loss_grid_tensor)
                        for _ in range(world_size)]
            # 收集所有进程的 loss_grid
            torch.distributed.all_gather(all_loss, loss_grid_tensor)
            # 将收集到的结果拼接成一个完整的 loss_grid
            # loss_grid_tensor = torch.cat(all_loss, dim=0)
            loss_grid_tensor = torch.stack(all_loss, dim=0).mean(dim=0)
            loss_grid = loss_grid_tensor.cpu().numpy()  # 转回 NumPy 数组后再保存
        else:
            all_loss = [loss_grid]

        # 仅rank 0进程保存结果
        if rank == 0:
            with h5py.File(surf_file, 'w') as f:
                f.create_dataset('xcoordinates', data=x_coords)
                f.create_dataset('ycoordinates', data=y_coords)
                f.create_dataset('train_loss', data=loss_grid)

            logger.info(f"计算完成,结果保存至{surf_file}")

        # return True
    


    # def compute_fisher_information(self, eval_dataset, output_dir, name="FisherInfo", fisher_samples=1000, batch_size=32):
    #     """
    #     计算 Fisher Information 并存储到 HDF5 文件,包括统计信息
    #     :param flag_Nlora: 是否使用 Nlora 方法
    #     :param eval_dataset: 评估数据集
    #     :param output_dir: 结果存储目录
    #     :param name: 结果文件名称
    #     :param fisher_samples: 计算 Fisher 信息所用的样本数
    #     :param batch_size: 计算 Fisher 信息时的 batch 大小
    #     """
    #     args = self.args
    #     device = args.device

    #     if args.lora_strategy.lower() == 'nlora':
    #         fisher_file = os.path.join(
    #             output_dir, f"{name}_Fisher_Nlora_task.h5")
    #     elif args.lora_strategy.lower() == 'lora':
    #         fisher_file = os.path.join(output_dir, f"{name}_Fisher_LoRA.h5")

    #     os.makedirs(output_dir, exist_ok=True)

    #     # 复制模型避免污染
    #     model = copy.deepcopy(self.model)
    #     model = self._wrap_model(model, training=False)
    #     model.to(device).eval()

    #     # 随机采样训练样本
    #     sampler = RandomSampler(
    #         eval_dataset, replacement=True, num_samples=fisher_samples)
    #     dataloader = DataLoader(
    #         eval_dataset, sampler=sampler, batch_size=batch_size)

    #     fisher_vector = None
    #     total_samples = 0

    #     logger.info(
    #         f"计算 Fisher Information, 样本数: {fisher_samples}, batch_size: {batch_size}")

    #     for batch in tqdm(dataloader, desc="Computing Fisher Information"):
    #         inputs, labels = batch
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         model.zero_grad()

    #         # 计算模型输出
    #         outputs = model(inputs)
    #         log_probs = F.log_softmax(outputs.logits, dim=-1)

    #         # 选择 ground-truth 对应的 log 概率
    #         log_probs_selected = log_probs.gather(
    #             dim=-1, index=labels.unsqueeze(-1)).squeeze()

    #         # 计算对数概率的梯度
    #         gradients = torch.autograd.grad(
    #             log_probs_selected.sum(), model.parameters(), create_graph=False)

    #         # 计算 Fisher Information 的对角近似
    #         grad_vector = torch.cat([g.view(-1)
    #                                 for g in gradients]).detach() ** 2

    #         # Fisher 信息累加
    #         if fisher_vector is None:
    #             fisher_vector = grad_vector
    #         else:
    #             fisher_vector += grad_vector

    #         total_samples += len(labels)

    #         # 每 5 个 batch 释放显存
    #         if total_samples % (5 * batch_size) == 0:
    #             torch.cuda.empty_cache()

    #     fisher_vector /= total_samples  # 归一化

    #     # **计算 Fisher Information 统计信息**
    #     fisher_stats = self._compute_fisher_stats(fisher_vector.cpu().numpy())

    #     # **保存 Fisher 信息和统计信息**
    #     with h5py.File(fisher_file, 'w') as f:
    #         f.create_dataset('fisher_vector', data=fisher_vector.cpu().numpy())
    #         for k, v in fisher_stats.items():
    #             f.attrs[k] = v  # 直接将统计数据写入 HDF5 文件

    #     logger.info(f"Fisher Information 计算完成,结果保存至 {fisher_file}")
    #     logger.info(f"Fisher 统计信息: {fisher_stats}")

    # def _compute_fisher_stats(self, fisher_values):
    #     """计算 Fisher Information 的统计信息"""
    #     if fisher_values is None or len(fisher_values) == 0:
    #         return {"min": 0., "max": 0., "median": 0., "mean": 0., "std": 0.}

    #     arr = np.array(fisher_values)
    #     return {
    #         "min": float(np.min(arr)),
    #         "max": float(np.max(arr)),
    #         "median": float(np.median(arr)),
    #         "mean": float(np.mean(arr)),
    #         "std": float(np.std(arr))
    #     }

    def compute_hessian_version1(
        self,
        eval_dataset,
        output_dir,
        name="hessian",
        max_batches=10,
        sample_batches=False,
        use_gpu=True,

    ):
        """
        改进版Hessian矩阵计算函数,主要优化：
        1. 完全消除参数污染风险
        2. 支持分布式训练环境
        3. 增强数值稳定性
        4. 内存效率优化
        5. 增加特征向量分析

        参数说明：
        - max_batches: 最大计算batch数（用于大数据集采样）
        - sample_batches: 是否随机采样batch（True=随机,False=顺序取前N个）

        """
        # --------------------- 初始化阶段 ---------------------
        # 根据模型判断,这里不妨使用手动

        model_type = 'lora'
        if model_type == 'fintune':
            #如果模型是经过微调的结果,那么整个模型都要计算Hessian矩阵
            pass
        if model_type == 'lora':
            # 如果模型是经过lora微调的结果,那么只需要计算lora部分的Hessian矩阵, 
            # 因为原来的模型是冻结的,没有更新所以不需要计算 Hessian 矩阵
            # 新添加的部分是默认初始化,也不需要计算 Hessian 矩阵
            hessian_file_lora = os.path.join(output_dir, f"{name}_lora_only-predictDataset_lanczos.h5")


        logger.info(f'***5***--5-3**1 begin init   ')
        logger.info(
            f'***5***--5-3**1 use distribute ---- {torch.distributed.is_initialized()}')
        args = self.args
        device = args.device

        
        model = self.model
        model = self._wrap_model(model, training=False)

        # 混合精度处理
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=device)

        model = model.to(device=device)
        model.eval()

        logger.info(f'***5***--LORA Hessian**1 finish init ')

        # --------------------- 数据准备阶段 ---------------------
        logger.info(f'***5***--5-3**2 begin load data   ')
        # dataloader = self.get_eval_dataloader(eval_dataset)
        test_dataloader = self.get_test_dataloader(flat_dataset)
        all_batches = list(dataloader)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # all_batches = all_batches[rank::world_size]  # 数据分片
            logger.info(
                f'***5***--LORA Hessian**2 distribute yes, rank:{rank}, total_batches:{len(all_batches)}')
        else:
            logger.info(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0

        # 批量采样逻辑（改进点3：内存优化）
        # logger.info(
        #     f'***5***--5-3**2-1 if  {len(all_batches)} , {max_batches} ')
        # if len(all_batches) > max_batches:
        #     if sample_batches:
        #         indices = np.random.choice(
        #             len(all_batches), max_batches, replace=False)
        #         all_batches = [all_batches[i] for i in indices]
        #     else:
        #         all_batches = all_batches[:max_batches]

        # logger.info(f'***5***--5-3**2 finish load data ')
        logger.info(f'***5***--5-3**2 finish load data ')
        def dataloader_stream_sampler(dataloader, max_batches, sample_batches, rank=0, world_size=1):
            """
            支持分布式的数据流式 batch 采样器。
            - dataloader: torch 的数据加载器
            - max_batches: 最多取几个 batch
            - sample_batches: True 随机采样,False 顺序取
            - rank/world_size: 分布式环境下的本地索引
            """
            # 把所有batch的index组成一个list
            indices = list(range(len(dataloader)))
            if sample_batches:
                np.random.shuffle(indices)
            indices = indices[:max_batches]

            # 分布式切分
            indices = indices[rank::world_size]

            # 只取指定的batch
            for i, batch in enumerate(dataloader):
                if i in indices:
                    yield batch
                if len(indices) > 0 and i > max(indices):
                    break
        logger.info(f'***5***--5-3**2 begin load data ')
        batch_iterator = dataloader_stream_sampler(test_dataloader, max_batches, sample_batches, rank=rank, world_size=world_size)
        logger.info(f'***5***--5-3**2 finish load data ')
        # --------------------- 核心算法定义 ---------------------

        class HessianCalculator:
            """
            Hessian 计算器：
            - 计算 Hessian-Vector Product (HVP)
            - 使用 Lanczos 方法估计 Hessian 的特征值

            参数：
            - model: 计算 Hessian 的神经网络模型
            - device: 计算设备 (CPU/GPU)
            """

            def __init__(self, model, device, max_dim=10000):
                logger.info(f'***5***--5-3**3 class HessianCalculator init   ')
                self.model = model
                self.device = device
                self.criterion = torch.nn.CrossEntropyLoss()
                # self.max_dim = max_dim  # 限制 Hessian 计算的最大维度

            @staticmethod
            def _safe_normalize(v, eps=1e-12):
                """改进点4：安全归一化防止除零错误"""
                norm = torch.norm(v) + eps
                return v / norm

            def compute_hvp(self, batch, param_list=None):
                """计算Hessian-vector乘积函数 (HVP) """
                logger.info(f'***5***--5-3**5 begin compute_hvp() ')
                self.model.zero_grad()  # 清空梯度
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autograd.set_grad_enabled(True):
                    # outputs = checkpoint(self.model, **batch)  # 梯度检查点
                    outputs = self.model(**batch)
                    loss = self.criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch["labels"].view(-1)
                    )

                # 根据传入参数的类型适配：字典或列表/元组
                # 选择需要计算 Hessian 的参数
                if isinstance(param_list, dict):
                    params = [p for p in param_list.values()
                              if p.requires_grad]
                elif isinstance(param_list, (list, tuple)):
                    params = [p for p in param_list if p.requires_grad]
                else:
                    params = []  # 若传入为 None 或其它类型,则空处理

                # 计算一阶梯度
                grads = torch.autograd.grad(loss, params, create_graph=True)

                def hvp_func(v):
                    """计算 Hessian 向量积,闭包函数保持计算图"""

                    split_sizes = [p.numel()
                                   for p in params]  # 计算每个 param 对应的大小
                    v_split = torch.split(v, split_sizes)  # 按照每个参数的形状拆分 v
                    v_reshaped = [v_i.view(p.shape) for v_i, p in zip(
                        v_split, params)]  # 重新调整 v_i 形状


                    # 计算 Hessian 作用
                    Hv = torch.autograd.grad(
                        grads, params, grad_outputs=v_reshaped,
                        retain_graph=True, allow_unused=True
                    )

                    # 拼接计算结果
                    Hv_flattened = torch.cat([
                        hv.contiguous().flatten() if hv is not None else torch.zeros_like(p).flatten()
                        for hv, p in zip(Hv, params)
                    ]).to(self.device)

                    # logger.info(f"🔹 Hv computed successfully, shape: {Hv_flattened.shape}")
                    torch.cuda.empty_cache()  # 释放显存
                    return Hv_flattened

                return hvp_func

            def lanczos_algorithm(self, hvp_func, dim, order=5, num_splits=4, random_seed=0):
                """
                使用标准 Lanczos 方法计算 Hessian 特征值,并基于 v_chunks 分块计算
                - hvp_func: Hessian-Vector Product 计算函数
                - dim: 需要计算的参数维度
                - order: Lanczos 迭代阶数
                - num_splits: 将 v 拆分成 num_splits 份,逐步计算 HVP 以减少显存占用
                - random_seed: 随机种子
                """
                torch.manual_seed(random_seed)
                warnings.filterwarnings("ignore", category=UserWarning)

                # 限制 Hessian 计算的最大维度
                dim = min(dim, self.max_dim)

                # 使用 `float16` 降低显存需求
                tridiag = torch.zeros(
                    (order, order), dtype=torch.float16, device=self.device)
                vecs = torch.zeros(
                    (dim, order), dtype=torch.float16, device=self.device)

                # 生成初始随机向量并归一化
                init_vec = torch.randn(
                    dim, 1, dtype=torch.float16, device=self.device)
                init_vec /= torch.norm(init_vec)
                vecs[:, 0:1] = init_vec

                beta = 0
                v_old_chunks = [torch.zeros_like(chunk) for chunk in torch.chunk(
                    init_vec, num_splits, dim=0)]  # 存储上一次的 v_old

                for i in range(order):
                    start_time = time.time()
                    v = vecs[:, i:i+1]

                    # ✅ 分块计算 HVP
                    v_chunks = torch.chunk(v, num_splits, dim=0)
                    w_chunks = []

                    for j, v_chunk in enumerate(v_chunks):
                        w_chunk = hvp_func(v_chunk)  # 计算 HVP
                        w_chunk = w_chunk - beta * \
                            v_old_chunks[j]  # 计算 w - beta * v_old
                        w_chunks.append(w_chunk)

                    # ✅ 计算 alpha（分块）
                    alpha_chunks = [torch.matmul(
                        w_chunk.T, v_chunk) for w_chunk, v_chunk in zip(w_chunks, v_chunks)]
                    alpha = sum(alpha_chunks)  # 逐块累加
                    tridiag[i, i] = alpha

                    # ✅ 计算 w_chunks = w_chunks - alpha * v_chunks
                    for j in range(num_splits):
                        w_chunks[j] = w_chunks[j] - alpha * v_chunks[j]

                    # ✅ 重正交化（分块计算）
                    for j in range(i):
                        tau_chunks = torch.chunk(
                            vecs[:, j:j+1], num_splits, dim=0)
                        coeff_chunks = [torch.matmul(
                            w_chunk.T, tau_chunk) for w_chunk, tau_chunk in zip(w_chunks, tau_chunks)]
                        coeff = sum(coeff_chunks)

                        for k in range(num_splits):
                            w_chunks[k] = w_chunks[k] - coeff * tau_chunks[k]

                    # ✅ 重新计算 beta
                    beta = torch.norm(torch.cat(w_chunks, dim=0))
                    if beta < 1e-6:
                        warnings.warn(
                            f"数值稳定性问题: beta={beta.item()} 在迭代 {i} 时过小。")

                    # ✅ 更新 vecs（直接使用 w_chunks）
                    if i + 1 < order:
                        tridiag[i, i+1] = beta
                        tridiag[i+1, i] = beta
                        vecs[:, i+1:i+2] = torch.cat(
                            [w_chunks[j] / beta for j in range(num_splits)], dim=0)

                    # ✅ 更新 v_old_chunks（用于下轮迭代）
                    v_old_chunks = [
                        w_chunks[j].clone() / beta for j in range(num_splits)]

                    elapsed_time = time.time() - start_time
                    print(
                        f"Iter {i}/{order}: α={alpha.item():.6f}, β={beta.item():.6f}, Time={elapsed_time:.2f}s")

                    torch.cuda.empty_cache()  # 释放显存

                return tridiag

            def block_lanczos(self, hvp_func, dim, k=10, block_size=2):
                """
                改进点7：分块Lanczos算法（内存优化）
                其中超参数 k 是迭代次数,block_size 是块的大小。

                """
                logger.info(f'***5***--5-3**5 begin block_lanczos() ')
                # 初始化分块正交基
                Q = torch.zeros((k+1)*block_size, dim, device=self.device)
                T = torch.zeros(k*block_size, k*block_size, device=self.device)

                # 生成初始分块
                V = torch.randn(dim, block_size, device=self.device)
                V, _ = torch.linalg.qr(V)  # 正交化
                Q[:block_size] = V.T

                for i in range(k):
                    start_idx = i * block_size
                    # 计算Hessian作用
                    HV = torch.stack([hvp_func(Q[start_idx + j])
                                     for j in range(block_size)])

                    # 正交化过程
                    for j in range(start_idx, start_idx + block_size):
                        T[j, :j+1] = Q[:j+1] @ HV[j-start_idx]  # 计算三对角矩阵 T
                        # HV[j-start_idx] -= Q[:j+1] @ T[j, :j+1].T
                        # HV[j-start_idx] -= Q[:j+1] @ T[j, :j+1].unsqueeze(1)  # ✅ 修正形状问题
                        # HV[j-start_idx] -= Q[:j+1].T @ T[j, :j+1].unsqueeze(1)  # ✅ 修正形状问题
                        # 🔹 在计算前检查形状
                        # logger.info(f"🔹 Q[:j+1].shape: {Q[:j+1].shape}")
                        # logger.info(f"🔹 Q[:j+1].T.shape: {Q[:j+1].T.shape}")
                        # logger.info(f"🔹 T[j, :j+1].shape: {T[j, :j+1].shape}")
                        # logger.info(f"🔹 T[j, :j+1].unsqueeze(1).shape: {T[j, :j+1].unsqueeze(1).shape}")
                        # logger.info(f"🔹 HV[j-start_idx].shape: {HV[j-start_idx].shape}")

                        # ✅ 修正形状
                        HV[j-start_idx] -= (Q[:j+1].T @
                                            T[j, :j+1].unsqueeze(1)).squeeze()

                    # QR分解
                    V, R = torch.linalg.qr(HV.T)  # 正交化
                    Q[start_idx+block_size:start_idx+2*block_size] = V.T

                    # ✅ 修正错误：确保索引范围不会为空
                    if start_idx+block_size < T.shape[0]:
                        end_row = min(start_idx+2*block_size, T.shape[0])
                        end_col = min(start_idx+block_size, T.shape[1])

                        # logger.info(f"🔹 Updating T matrix at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}]")

                        T[start_idx+block_size:end_row, start_idx:end_col] = R.T
                    else:
                        logger.info(
                            f"❌ Skipping T update at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}] to prevent empty slice.")

                    # # ✅ 确保 T 不会越界
                    # end_row = min(start_idx+2*block_size, T.shape[0])
                    # end_col = min(start_idx+block_size, T.shape[1])

                    # logger.info(f"🔹 Updating T matrix at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}]")

                    # T[start_idx+block_size:end_row, start_idx:end_col] = R.T
                    # T[start_idx+block_size:start_idx+2*block_size,
                    # start_idx:start_idx+block_size] = R.T

                # 计算特征值
                T_np = T.cpu().numpy()
                eigvals = np.linalg.eigvalsh(T_np)
                return eigvals[-block_size:]  # 返回最大特征值

        # --------------------- 主计算流程 ---------------------
        logger.info(f'***5***--5-3**3 begin main loop   ')

        # 只存储初始状态（不带梯度）,用于恢复模型状态
        original_params_to_calculate_hessian = {}


        if model_type == 'lora':
            dom_eigs_lora = []
            # 获取参数集合（改进点8：动态参数处理）
            lora_params = {}
            for name, param in model.named_parameters():
                # 只需要关注哪些需要进行更新的参数
                if param.requires_grad == True:
                    lora_params[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        calculator = HessianCalculator(model, device)

        # 分布式通信初始化（改进点9：分布式支持）

        # if torch.distributed.is_initialized():
        #     logger.info(f'***5***--5-3**4 distribute--yes ')
        #     world_size = torch.distributed.get_world_size()
        #     rank = torch.distributed.get_rank()
        #     all_batches = all_batches[rank::world_size]  # 数据分片
        #     logger.info(f'***5***--5-3**4 {len(all_batches)} ')
        # else:
        #     logger.info(f'***5***--5-3**4 distribute--no ')
        #     world_size = 1
        #     rank = 0

        try:
            # for batch in tqdm(all_batches, desc=f"Rank {rank}: Processing"):
            for batch_idx, batch in enumerate(tqdm(batch_iterator, desc=f"Rank {rank}: Processing")):
                # 重置模型参数（关键改进点10：消除参数污染）
                # model.load_state_dict(original_state)
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_calculate_hessian:
                    model.state_dict()[name].copy_(
                        original_params_to_calculate_hessian[name])

                
                if model_type == 'lora':
                    logger.info(
                        f'***5***--5-3**Model device: {next(model.parameters()).device},Batch device: {next(iter(batch.values())).device} ')
                    logger.info(f"Type of lora_params: {type(lora_params)}")
                    # logger.info(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.info(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.info(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    hvp_lora = calculator.compute_hvp(batch, lora_params)
                    eigvals = calculator.block_lanczos(
                        hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    # eigvals = calculator.lanczos_algorithm(
                    #     hvp_lora, dim=sum(p.numel() for p in lora_params.values()), order=10, num_splits=4)
                    # eigvals = calculator.lanczos_algorithm(
                    #     hvp_lora, dim=sum(p.numel() for p in lora_params.values()), order=10, num_splits=4)
                    dom_eigs_lora.append(eigvals[-1])  # 或者 eigvals[0],取决于实现
                    # tridiag = calculator.lanczos_algorithm(hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    # dom_eigs_lora.extend(torch.linalg.eigvalsh(tridiag).tolist())

                
                
                
                # 内存清理
                torch.cuda.empty_cache()

        except RuntimeError as e:
            logger.error(f"Hessian计算失败: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.warning("尝试启用梯度检查点...")
                # 此处可添加fallback逻辑
            raise

        # --------------------- 结果处理与保存 ---------------------
        logger.info(f'***5***--5-3**6 begin save ')
        # 分布式结果聚合（改进点11）
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        
            if model_type == 'lora':
                dom_eigs_lora_tensor = torch.tensor(
                    dom_eigs_lora, device=device)

                # 创建接收所有进程数据的列表
                all_dom_eigs_lora = [torch.zeros_like(
                    dom_eigs_lora_tensor) for _ in range(world_size)]

                # 收集所有进程的数据
                torch.distributed.all_gather(
                    all_dom_eigs_lora, dom_eigs_lora_tensor)

                # 拼接所有收集到的数据
                dom_eigs_lora = torch.cat(
                    all_dom_eigs_lora, dim=0).cpu().numpy().tolist()

                # 计算统计指标
                stats_lora = self._compute_stats(dom_eigs_lora)

        # HDF5保存（改进点12：元数据增强）
        # 仅 rank=0 进程保存结果,避免多个进程同时写入 HDF5
        if rank == 0:
            if model_type == 'lora':
                with h5py.File(hessian_file_lora, "w") as hf:
                    hf.attrs["created_at"] = datetime.now().isoformat()
                    hf.attrs["model_type"] = type(model).__name__
                    for k, v in stats_lora.items():
                        hf.create_dataset(k, data=v)
                    hf.create_dataset(
                        "dominant_eigs", data=np.array(dom_eigs_lora))

                logger.info(f"计算完成,结果保存至 {hessian_file_lora} 文件")
        # 🚀 添加进程同步 & 关闭分布式进程
        if torch.distributed.is_initialized():
            logger.info("所有进程同步中...")
            torch.distributed.barrier()  # 确保所有进程都完成再继续

            if torch.distributed.get_rank() == 0:
                logger.info("所有进程已完成计算,开始关闭分布式进程...")

            torch.distributed.destroy_process_group()  # 释放 NCCL 资源
            logger.info("分布式进程已正确关闭")
        return True


#---------------- 计算自定义指标 ----------------   


    def compute_task_aware_expected_sharpness(self,
        flat_dataset,
        output_dir,
        flat_eval_type='Kong',
        Flag_seq = -1,
        sigma=0.004,  # 扰动标准差 ,这个是超参数,等待确定
        n_samples=30,
        max_batches=20,
        ):
        # import torch.distributed
        """
        计算 Task-Aware Expected Sharpness,支持精度切换、分布式和自适应 batch。
        """

        args = self.args
        device = args.device


        # ========== 初始化与精度设置 ==========
        # 1. 模型准备
        model = self.model
        model = self._wrap_model(model, training=False)
        model = model.to(device=device)
        model.eval()

        
        # 2. 只保存一次原始参数副本
        # ========== 确定扰动参数集合 ==========
        original_params_to_perturb = {}
        param_names_to_perturb = []
        for name, param in model.named_parameters():
            # 修改默认就是全局的结果，只评估全局的结果即可
            if flat_eval_type== "eval-full":
                original_params_to_perturb[name] = param.data.clone()
                param_names_to_perturb.append(name)
            elif flat_eval_type== "trained_lora":
                if param.requires_grad:
                    original_params_to_perturb[name] = param.data.clone()
                    param_names_to_perturb.append(name)


        # ========== 数据准备,分布式与采样 ==========
        test_dataloader = self.get_test_dataloader(flat_dataset)
        batch_size = test_dataloader.batch_size

        if has_length(test_dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(test_dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")


        # # 4. baseline loss（每次直接遍历 dataloader,限制 max_batches）
        # ======= 2. 计算 baseline loss（用 prediction_step 统一接口） =======
    
        baseline_loss_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for step, inputs in enumerate(test_dataloader):
                if step >= max_batches:
                    break 
                loss, _, _ = self.prediction_step(model, inputs, prediction_loss_only=True, ignore_keys=None)
                baseline_loss_sum += loss.item() if hasattr(loss, "item") else float(loss)
                num_batches += 1
        baseline_loss = baseline_loss_sum / max(1, num_batches)

        # 4. 多次扰动采样,每次采样只做扰动/评估/恢复
        sharpness_samples = []
        sharpness_samples_abs = []
        perturbed_loss_sumary= []
        for i in range(n_samples):
            logger.debug(f"****disturb {i}")
            # 扰动
            with torch.no_grad():
                for name in param_names_to_perturb:
                    param = model.state_dict()[name]
                    epsilon = torch.normal(mean=0.0, std=sigma, size=param.shape, device=param.device, dtype=param.dtype)
                    param.add_(epsilon)

            # 重新遍历 DataLoader
            perturbed_loss_sum = 0.0
            num_batches = 0
            with torch.no_grad():
                for step, inputs in enumerate(test_dataloader):
                    # logger.debug(f"****-----step {step}")
                    if step >= max_batches:
                        break
                    loss, _, _ = self.prediction_step(model, inputs, prediction_loss_only=True, ignore_keys=None)
                    perturbed_loss_sum += loss.item() if hasattr(loss, "item") else float(loss)
                    num_batches += 1
            perturbed_loss = perturbed_loss_sum / max(1, num_batches)

            # 恢复参数（直接复制最初保存的原始值）
            with torch.no_grad():
                for name in param_names_to_perturb:
                    model.state_dict()[name].copy_(original_params_to_perturb[name])

            this_sharpness = perturbed_loss - baseline_loss
            perturbed_loss_sumary.append(float(perturbed_loss))
            sharpness_samples.append(float(this_sharpness))
            sharpness_samples_abs.append(abs(this_sharpness))


        # 5. 多指标统计与保存
        expected_sharpness = float(np.mean(sharpness_samples))
        mean_absolute_sharpness = float(np.mean(sharpness_samples_abs))
        max_absolute_sharpness = float(np.max(sharpness_samples_abs))
        max_signed_sharpness = float(np.max(sharpness_samples))
        min_signed_sharpness = float(np.min(sharpness_samples))
        std_signed_sharpness = float(np.std(sharpness_samples))
        percentile90_abs = float(np.percentile(sharpness_samples_abs, 90))
        percentile95_abs = float(np.percentile(sharpness_samples_abs, 95))

        logger.info(
            f"Task-aware Expected Sharpness (sigma={sigma}, n={n_samples}): {expected_sharpness:.6f}, "
            f"MeanAbs: {mean_absolute_sharpness:.6f}, MaxAbs: {max_absolute_sharpness:.6f}, Std: {std_signed_sharpness:.6f}"
        )

        result = {
            "flat_eval_type": flat_eval_type,
            "sigma": sigma,
            "n_samples": n_samples,
            "max_batches": max_batches,
            "expected_sharpness": expected_sharpness,
            "mean_absolute_sharpness": mean_absolute_sharpness,
            "max_absolute_sharpness": max_absolute_sharpness,
            "baseline_loss": baseline_loss,
            "output_dir": output_dir,
            "max_signed_sharpness": max_signed_sharpness,
            "min_signed_sharpness": min_signed_sharpness,
            "std_signed_sharpness": std_signed_sharpness,
            "percentile90_abs": percentile90_abs,
            "percentile95_abs": percentile95_abs,
            # 原始采样结果以list形式保存
            "perturbed_loss_sumary": perturbed_loss_sumary,
            "sharpness_samples": sharpness_samples,
            "sharpness_samples_abs": sharpness_samples_abs,
        }
        json_path = os.path.join(output_dir, f"{flat_eval_type}/task_aware_expected_sharpness_Redo2.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)  # 保证父目录存在
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Sharpness result saved to: {json_path}")

        # 恢复参数（直接复制最初保存的原始值）original_params_to_perturb
        with torch.no_grad():
            for name in param_names_to_perturb:
                model.state_dict()[name].copy_(original_params_to_perturb[name])



#---------------- 通过 Fisher 信息计算模型参数的 Fisher 信息矩阵 ----------------   


    # def compute_fisher(self, test_dataset: Dataset, output_dir: str, fisher_type: str = 'diag'):
    #     """
    #     计算并保存模型参数的Fisher信息矩阵
    
    #     Args:
    #         dataloader: 用于计算Fisher信息的数据集加载器
    #         output_dir: 保存Fisher信息矩阵的目录
    #         fisher_type: Fisher信息矩阵的类型,支持'diag'（对角矩阵）或'full'（完整矩阵）
    #     """
    #     args = self.args
        

        

    #     os.makedirs(output_dir, exist_ok=True)

    #     # 
    #     model = self._wrap_model(self.model, training=False)
    #     model.eval()


    #     test_dataloader = self.get_test_dataloader(test_dataset)
    #     batch_size = test_dataloader.batch_size

    #     if has_length(test_dataloader.dataset):
    #         logger.info(f"  Num examples = {self.num_examples(test_dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")


    

    #     for step, inputs in enumerate(dataloader):

    #         # 将模型设置为训练模式以启用梯度计算,但我们只计算特定梯度而不进行优化
    #         model.train() # 临时设置为训练模式以启用梯度


    #         # 清除之前的梯度
    #         model.zero_grad()

    #         # 将输入数据移动到设备
    #         inputs = self._prepare_inputs(inputs)



    #         # 计算损失。为了计算梯度,我们需要在 loss.backward() 之前确保 autograd 开启。
    #         # prediction_loss_only 应该为 False,以便我们能获取到 loss 对应的 graph
    #         prediction_loss_only = False # 我们需要 loss 来计算梯度
    #         ignore_keys = None # 在计算 Fisher 信息时通常不需要忽略 keys

    #         # 使用 prediction_step 获取损失
    #         loss, logits, labels = self.prediction_step(
    #             model, inputs, prediction_loss_only, ignore_keys=ignore_keys
    #         )


    #         if loss is None:
    #             logger.warning(f"Step {step}: Loss is None, skipping this batch for Fisher Matrix calculation.")
    #             continue


    #         # 计算 log-likelihood 对参数的梯度
    #         # 这里的 loss 通常是平均损失,为了计算每个样本的梯度,我们需要调整
    #         # 对于分类任务,loss 通常是交叉熵,这可以看作是 -log p(y|x, theta)
    #         # 所以其梯度就是 -∇log p(y|x, theta)。我们想要的是 ∇log p(y|x, theta)。
    #         # 因此,我们对负损失求梯度,或者直接对损失求梯度然后取负号。
    #         # 这里我们直接对 loss 求梯度,并将其视为 score function 的负梯度。
    #         # 为了计算 E[∇log p * ∇log p^T],我们可以计算 E[(g) * (g)^T],其中 g = ∇log p。
    #         # 由于 loss = -log p(y|x,theta),那么 ∇loss = -∇log p(y|x,theta)。
    #         # 所以 ∇log p(y|x,theta) = -∇loss。
    #         # 因此,我们需要计算 (-∇loss) * (-∇loss)^T = ∇loss * ∇loss^T

    #         # 对损失进行反向传播以计算梯度
    #         loss.backward()


    #         # 收集所有参数的梯度并展平
    #         gradients = []
    #         for p in model.parameters():
    #             if p.requires_grad and p.grad is not None:
    #                 gradients.append(p.grad.view(-1))
            
    #         if not gradients:
    #             logger.warning(f"Step {step}: No gradients found for trainable parameters, skipping this batch.")
    #             continue


    #         # 将所有梯度连接成一个大的向量
    #         gradient_vector = torch.cat(gradients)


    #         fisher_matrix += torch.outer(gradient_vector, gradient_vector)

    #         # 统计当前批次的样本数
    #         current_batch_size = find_batch_size(inputs)
    #         if current_batch_size is None: # Fallback for unknown batch size
    #             current_batch_size = 1 # Assume 1 if cannot determine
    #         total_samples += current_batch_size

    #         # 恢复模型为评估模式,确保下一个迭代是正确的
    #         model.eval()

    #     # 对累加的 Fisher 矩阵进行平均,以近似期望
    #     if total_samples > 0:
    #         fisher_matrix /= total_samples
    #     else:
    #         logger.warning("No samples processed for Fisher Matrix calculation.")

    #     # 将 Fisher 矩阵保存到文件
    #     fisher_matrix_path = os.path.join(output_dir, "fisher_information_matrix.pt")
    #     torch.save(fisher_matrix.cpu(), fisher_matrix_path)
    #     logger.info(f"Fisher Information Matrix saved to {fisher_matrix_path}")

    #     return fisher_matrix



    def compute_fisher_sharpness_measure(self, test_dataset: Dataset, output_dir: str, flat_eval_type: str,fisher_type: str = 'diag',measure_type: str = "mean_diag_fisher"):
        """
        计算并保存模型参数的Fisher信息矩阵
    
        Args:
            dataloader: 用于计算Fisher信息的数据集加载器
            output_dir: 保存Fisher信息矩阵的目录
            fisher_type: Fisher信息矩阵的类型,支持'diag'（对角矩阵）或'full'（完整矩阵）
        """


        

        os.makedirs(output_dir, exist_ok=True)

        # 
        model = self._wrap_model(self.model, training=False)
        model.eval()


        test_dataloader = self.get_test_dataloader(test_dataset)
        batch_size = test_dataloader.batch_size

        if has_length(test_dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(test_dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")


    
        # 用于累加每个参数的梯度平方 (对应对角 Fisher)
        sum_of_squared_gradients = None
        total_batches = 0

        for step, inputs in enumerate(test_dataloader):



            

            # 将模型设置为训练模式以启用梯度计算,但我们只计算特定梯度而不进行优化
            model.train() # 临时设置为训练模式以启用梯度
            enable_running_stats(model)


            # 清除之前的梯度
            model.zero_grad()

            # 将输入数据移动到设备
            inputs = self._prepare_inputs(inputs)



            # 计算损失。为了计算梯度,我们需要在 loss.backward() 之前确保 autograd 开启。
            loss = self.compute_loss(model, inputs)

            logger.info(f'compute_loss: {loss.item():.4f}')
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps
                logger.info(f' average training_step compute_loss: {loss.item():.4f}')
            if loss is None:
                logger.warning(f"Step {step}: Loss is None, skipping this batch for Fisher Matrix calculation.")
                continue


            # 计算 log-likelihood 对参数的梯度
            # 这里的 loss 通常是平均损失,为了计算每个样本的梯度,我们需要调整
            # 对于分类任务,loss 通常是交叉熵,这可以看作是 -log p(y|x, theta)
            # 所以其梯度就是 -∇log p(y|x, theta)。我们想要的是 ∇log p(y|x, theta)。
            # 因此,我们对负损失求梯度,或者直接对损失求梯度然后取负号。
            # 这里我们直接对 loss 求梯度,并将其视为 score function 的负梯度。
            # 为了计算 E[∇log p * ∇log p^T],我们可以计算 E[(g) * (g)^T],其中 g = ∇log p。
            # 由于 loss = -log p(y|x,theta),那么 ∇loss = -∇log p(y|x,theta)。
            # 所以 ∇log p(y|x,theta) = -∇loss。
            # 因此,我们需要计算 (-∇loss) * (-∇loss)^T = ∇loss * ∇loss^T

            # 对损失进行反向传播以计算梯度,
            loss.backward()


            # 收集所有参数的梯度平方并展平
            current_squared_gradients = []
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    # SSAM 的经验 Fisher 实际上是 (∇log p)^2,如果 loss 是 -log p,
                    # 那么 ∇loss = -∇log p,所以 (∇loss)^2 = (∇log p)^2。
                    # 这里我们直接取梯度的平方,这对应于对角 Fisher 元素。
                    current_squared_gradients.append(p.grad.view(-1).pow(2))
            
            if not current_squared_gradients:
                logger.warning(f"Step {step}: No gradients found for trainable parameters, skipping this batch.")
                continue

            # 将所有梯度平方连接成一个大的向量
            squared_gradient_vector = torch.cat(current_squared_gradients)

            # 累加梯度平方
            if sum_of_squared_gradients is None:
                sum_of_squared_gradients = squared_gradient_vector
            else:
                sum_of_squared_gradients += squared_gradient_vector
            
            total_batches += 1

            # 恢复模型为评估模式
            model.eval()

        if total_batches == 0:
            logger.warning("No batches processed for Fisher sharpness measure calculation. Returning 0.")
            return 0.0

        # 对累加的梯度平方进行平均,得到近似的对角 Fisher 元素
        avg_diag_fisher_elements = sum_of_squared_gradients / total_batches

        # 确保 avg_diag_fisher_elements 是 1D tensor
        if avg_diag_fisher_elements is not None and avg_diag_fisher_elements.numel() > 0:
            fisher_stats = {
                "mean_diag_fisher": avg_diag_fisher_elements.mean().item(),
                "trace_diag_fisher": avg_diag_fisher_elements.sum().item(),
                "max_diag_fisher": avg_diag_fisher_elements.max().item(),
                "num_params": avg_diag_fisher_elements.numel(),
                # 增加保存所有原始数据（转为Python list以兼容JSON）
                # "diag_fisher_elements": avg_diag_fisher_elements.detach().cpu().tolist()
            }
        else:
            fisher_stats = {
                "mean_diag_fisher": 0.0,
                "trace_diag_fisher": 0.0,
                "max_diag_fisher": 0.0,
                "num_params": 0,
                "diag_fisher_elements": []
            }
            logger.warning("No trainable parameters found. Returning 0 for all Fisher stats.")

        # logger.info(f"Fisher statistics: {json.dumps(fisher_stats, indent=2)}")

        # 保存到 json 文件
        json_save_path = os.path.join(output_dir, f"{flat_eval_type}/fisher_sharpness_stats.json")
        os.makedirs(os.path.dirname(json_save_path ), exist_ok=True)  # 保证父目录存在
        with open(json_save_path, "w") as f:
            json.dump(fisher_stats, f, indent=2)

        logger.info(f"Fisher sharpness statistics saved to {json_save_path}")


        # 恢复参数（直接复制最初保存的原始值）
        # with torch.no_grad():
        #     for name in param_names_to_perturb:
        #         param = dict(model.named_parameters())[name]
        #         param.data.copy_(original_params_to_perturb[name])


        


       
        
#----------------简化版 计算Hessien 

    def compute_hessian_max_eigenvalue(self, test_dataset: Dataset, output_dir: str, flat_eval_type:str, max_batches=1, order=5, 
    ):
        """
        仅计算LoRA部分参数的Hessian最大特征值（Lanczos）。
        """
        logger.debug("compute_hessian_max_eigenvalue")

        model = self.model
        device = next(model.parameters()).device 

        # 只取需要计算梯度的参数
        # params_gradient = [p for n, p in model.named_parameters() if p.requires_grad]

        
        original_params_to_perturb = {}
        params_gradient_names = []      # 储存 name（字符串）
        params_gradient_tensors = []    # 储存 param（Tensor）

        for name, param in model.named_parameters():
            if self.args.do_train:
                pass
            else:
                if param.requires_grad:
                    original_params_to_perturb[name] = param.data.clone()
                    params_gradient_names.append(name)
                    params_gradient_tensors.append(param)

        # 用 tensor 列表计算 param_dim
        param_dim = sum(p.numel() for p in params_gradient_tensors)

        # 损失累积
        test_dataloader = self.get_test_dataloader(test_dataset)
        losses = []

        for step, inputs in enumerate(test_dataloader):
            # logger.debug(f"Calculate loss {step}")

            if step >= max_batches:
                break

            model.train()
            
            model.zero_grad()

            # 将输入数据移动到设备
            inputs = self._prepare_inputs(inputs)
            loss = self.compute_loss(model, inputs)

            # 可以加一些日志输出
            if loss is None:
                logger.warning(f"Step {step}: Loss is None, skipping this batch for Hessian calculation.")
                continue

            losses.append(loss)

        
        if not losses:
            raise ValueError("No valid loss found for Hessian calculation.")

        loss = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, device=device) for l in losses]).mean()

        # Hessian-vector product (HVP)闭包
        # def hvp_fn(v):
        #     # 将向量分割为与参数形状匹配的块
        #     split_vs = []
        #     pointer = 0

        #     for p in params_gradient:
        #         num = p.numel()
        #         split_vs.append(v[pointer:pointer+num].view_as(p))
        #         pointer += num

        #     # 计算一阶梯度
        #     grad = torch.autograd.grad(loss, params_gradient, create_graph=True)
        #     flat_grad = torch.cat([g.contiguous().view(-1) for g in grad])

        #     # 计算HVP
        #     hv = torch.autograd.grad(flat_grad, params_gradient, grad_outputs=split_vs, retain_graph=True)
        #     flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])

        #     return flat_hv
        # Hessian-vector product (HVP)闭包（优化内存版本）
        def hvp_fn(v):
            # 将v切分为与参数匹配的块
            split_vs = []
            pointer = 0
            for p in params_gradient_tensors:
                num = p.numel()
                split_vs.append(v[pointer:pointer+num].view_as(p))
                pointer += num
            # 1. 先计算一阶梯度
            grad = torch.autograd.grad(loss, params_gradient_tensors, create_graph=True)
            # 2. 对一阶梯度做二阶方向导数
            hv = torch.autograd.grad(grad, params_gradient_tensors, grad_outputs=split_vs, retain_graph=True)
            # 3. 拼成一维向量
            flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
            return flat_hv

        # Lanczos主循环
        # with torch.no_grad():
        v = torch.randn(param_dim, device=device)
        v /= v.norm()

        q = [v]
        alphas = []
        betas = []

        for i in range(order):
            # logger.debug(f"HVP: {i}")
            # 计算HVP
            # 计算HVP
            w = hvp_fn(q[-1])  

             # 正交化
            if i > 0:
                w = w - betas[-1] * q[-2]


            alpha = torch.dot(q[-1], w)
            alphas.append(alpha.item())


            w = w - alpha * q[-1]
            beta = w.norm()


            if beta < 1e-7:  # 数值收敛提前终止
                break

            betas.append(beta.item())
            # 内存优化4：只保留必要的向量,释放旧向量
            if len(q) > 2:
                del q[0]
            q.append(w / beta)
            q[-1] = q[-1]  
        # 构造三对角矩阵
        T = np.diag(alphas)
        for i, beta in enumerate(betas):
            if i+1 < T.shape[0]:
                T[i, i+1] = T[i+1, i] = beta

        # 计算特征值
        # 计算特征值
        eigvals = np.linalg.eigvalsh(T)
        eigvals_list = eigvals.tolist()  # 转为list便于json存储
        max_eig = float(np.max(eigvals))
        min_eig = float(np.min(eigvals))
        mean_eig = float(np.mean(eigvals))
        std_eig = float(np.std(eigvals))

        # 保存特征值
        # 创建输出路径
        if output_dir:
            output_path = os.path.join(output_dir, f"{flat_eval_type}/hessian_eigenvalues.npy") if output_dir else None
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, eigvals)
        # 保存json统计+原始数据
        json_path = os.path.join(output_dir, f"{flat_eval_type}/hessian_eigenvalues_stats.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)  # 保证父目录存在
        logger.debug(f"Hessien save in {json_path}")
        eig_info = {
            "max_eigenvalue": max_eig,
            "min_eigenvalue": min_eig,
            "mean_eigenvalue": mean_eig,
            "std_eigenvalue": std_eig,
            "num_eigenvalues": len(eigvals_list),
            "eigenvalues": eigvals_list
        }
        with open(json_path, "w") as f:
            json.dump(eig_info, f, indent=2)
        # logger.debug(f"Max Hessian eigenvalue (LoRA): {max_eig:.6f}, saved to {json_path} and {npy_path}")

        # return max_eig

        # 恢复参数（直接复制最初保存的原始值）
        # 用 name 列表恢复参数
        for name in params_gradient_names:
            param = dict(model.named_parameters())[name]
            param.data.copy_(original_params_to_perturb[name])
        



#----------可视化 

    def compute_loss_landscape_easy(self, test_dataset: Dataset, output_dir, flat_eval_type = 'originModel',x_range=(-1, 1), y_range=(-1, 1), num_points=20, max_batches=1, sample_batches=False):
        """
        计算损失景观,并分别计算:
        1️⃣ **完整模型（Base Model + LoRA Adapter）** 的损失景观
        2️⃣ **仅 LoRA 适配器（LoRA Adapter）** 的损失景观
        计算损失景观,并分别计算完整模型 & LoRA Adapter 的损失表面。
        针对大模型使用多 GPU,可以并行分配 (i, j) 坐标网格。
        """
        args = self.args
        device = args.device
        # 最终损失网络的数值
        loss_grid = np.zeros((num_points, num_points))

        # 根据模型判断,这里不妨使用手动
        # 是否是训练后的模型,还是加载已经训练好的模型
        if args.do_train:
           pass
        else:
            surf_file = os.path.join(output_dir, f"{flat_eval_type}_T5large__testData.h5")
        

        # 确保目标目录存在（若不存在则创建）
        if not os.path.exists(surf_file):
            with open(surf_file, 'w') as f:
                pass  # 成功创建空文件
        os.makedirs(os.path.dirname(surf_file), exist_ok=True)  # 保证父目录存在

        # ✅ 记录日志信息
        logger.info(f'***5***--5-2 **1 compute_loss_landscape ')
        logger.info(
            f"***** Running Loss Landscape Calculati on {flat_eval_type}*****")
        logger.info(
            f"Output Dir = {output_dir},Output File :{surf_file} Num points = {num_points}x{num_points} max_batches = {max_batches}")

        
        model = self.model
        model = self._wrap_model(model, training=False)
        model = model.to(device=device)
        model.eval()  # 确保模型在 eval 模式

        # --------------------- 生成参数扰动阶段阶段 ---------------------
        # 确定需要扰动的参数名称
        # 根据不同的微调方法（Nlora或lora）确定需要保存原始值的参数
        original_params_to_perturb = {}
        for name, param in model.named_parameters():
            if args.do_train:
                # if distrub_name == 'originModel':
                pass
            else: # 前面通过判断 确定了扰动范围
                if param.requires_grad: 
                    original_params_to_perturb[name] = param.data.clone()
                    

        # --------------------- 扰动生成优化 ---------------------
        torch.manual_seed(42)  # 固定随机种子保证可重复性
        perturb_x, perturb_y = {}, {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in original_params_to_perturb:
                    continue
                else:
                    d_x = torch.randn_like(param)
                    d_x = (d_x / d_x.norm()) * (param.norm() + 1e-8)  # 直接归一化
                    perturb_x[name] = d_x.to(device)

                    # 生成正交扰动
                    d_y = torch.randn_like(param)
                    d_y = d_y - torch.sum(d_y * d_x) * \
                        d_x / (d_x.norm()**2)  # 施密特正交化
                    d_y = (d_y / d_y.norm()) * (param.norm() + 1e-8)     # 归一化
                    perturb_y[name] = d_y.to(device)


        # --------------------- 并行化网格计算（优化点3）---------------------
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        y_coords = np.linspace(y_range[0], y_range[1], num_points)

        
        # --------------------- 数据准备阶段 ---------------------
        logger.info(f'***5***--5-2**2 begin load data---{torch.distributed.is_initialized()} ')
        dataloader = self.get_test_dataloader(test_dataset)
        

        # --------------------- 主计算循环优化 ---------------------

        for i in tqdm(range(num_points), desc="Loss Landscape"):
            xv = x_coords[i]
            for j, yv in enumerate(y_coords):
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_perturb:
                    model.state_dict()[name].copy_(
                        original_params_to_perturb[name])

                # 应用扰动
                with torch.no_grad():
                    for name in original_params_to_perturb:
                        param = model.state_dict()[name]
                        delta = xv * \
                            perturb_x[name].to(
                                param.dtype) + yv * perturb_y[name].to(param.dtype)
                        param.add_(delta)  # 原位操作减少内存分配

                    total_loss = 0.0
                    num_batches = 0
                    for step, inputs in enumerate(dataloader):
                        if step >= max_batches:
                            break

                        inputs = self._prepare_inputs(inputs)
                        loss, _, _ = self.prediction_step(
                            model, 
                            inputs, 
                            prediction_loss_only=True, 
                            ignore_keys=None
                        )
                        total_loss += loss.item() if hasattr(loss, "item") else float(loss)
                        num_batches += 1
                loss_grid[i, j] = total_loss / num_batches

                # 每5次迭代清理一次缓存（优化点6）
                if j % 5 == 0:
                    torch.cuda.empty_cache()
        all_loss = [loss_grid]

        # 仅rank 0进程保存结果
        
        with h5py.File(surf_file, 'w') as f:
            f.create_dataset('xcoordinates', data=x_coords)
            f.create_dataset('ycoordinates', data=y_coords)
            f.create_dataset('train_loss', data=loss_grid)
        logger.info(f"计算完成,结果保存至{surf_file}")




    
    def compute_loss_landscape_easy_1d(self, test_dataset: Dataset, output_dir, flat_eval_type='lossland_all',
                              x_range=(-0.5, 0.5), num_points=40, max_batches=100, sample_batches=False):
        """
        计算一维参数扰动下的损失曲线。
        仅在 x 方向（指定扰动方向）采样，分析损失随扰动强度的变化趋势。
        """
        args = self.args
        device = args.device

       

        # 文件路径
        surf_file = os.path.join(output_dir, f"{flat_eval_type}_T5large__testData_1d_NewSet_Try2_LongLora2.h5")
        os.makedirs(os.path.dirname(surf_file), exist_ok=True)

        # 记录日志
        logger.info(f"***** Running 1D Loss Curve Calculation: {flat_eval_type} *****")
        logger.info(f"Output Dir = {output_dir}, Output File = {surf_file}, Num points = {num_points}, max_batches = {max_batches}")

        model = self.model
        model = self._wrap_model(model, training=False)
        model = model.to(device=device)
        model.eval()


        # 保存完整原始参数（全模型保存，无论 lora 还是全模型扰动都统一还原用）
        checkpoint_state_dict = {name: param.detach().clone() for name, param in model.named_parameters()}


        # 保存原始参数，默认是对整个模型都进行可视化
        # original_params_to_perturb = {}
        # for name, param in model.named_parameters():
        #     # if param.requires_grad:
        #     #     original_params_to_perturb[name] = param.data.clone()
        #     if flat_eval_type == "lossland_all":
        #         original_params_to_perturb[name] = param.data.clone()
        #     elif flat_eval_type == "lossland_lora":
        #         if param.requires_grad and 'lora' in name:
        #             original_params_to_perturb[name] = param.data.clone()
        # 只选择要扰动的参数
        perturb_names = []
        for name, param in model.named_parameters():
            if flat_eval_type == "lossland_all":
                perturb_names.append(name)
            elif flat_eval_type == "lossland_lora":
                if "lora_" in name and param.requires_grad:
                    perturb_names.append(name)


        # 固定扰动方向 d_x
        torch.manual_seed(42)
        perturb_x = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in perturb_names:
                    continue
                d_x = torch.randn_like(param)
                d_x = (d_x / (d_x.norm() + 1e-8)) * (param.norm() + 1e-8)
                perturb_x[name] = d_x.to(device)

        # x 方向采样点
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        if 0.0 not in x_coords:
            x_coords = np.insert(x_coords, len(x_coords) // 2, 0.0)
            x_coords = np.unique(np.sort(x_coords))
        logger.info(f"x_coords: {x_coords}")


         # 最终损失向量
        loss_curve = np.zeros(len(x_coords))

        # 数据加载
        logger.info(f"Begin loading data: DDP={torch.distributed.is_initialized()}")
        dataloader = self.get_test_dataloader(test_dataset)

        # 主计算循环：只遍历 x 轴
        for i, xv in enumerate(tqdm(x_coords, desc="1D Loss Curve")):
            # 恢复参数
            # for name in original_params_to_perturb:
            #     model.state_dict()[name].copy_(original_params_to_perturb[name])
            # 确保还原所有参数到 checkpoint 状态
            for name, param in model.named_parameters():
                param.data.copy_(checkpoint_state_dict[name])

            # 应用 x 方向扰动
            # with torch.no_grad():
            #     for name in original_params_to_perturb:
            #         param = model.state_dict()[name]
            #         delta = xv * perturb_x[name].to(param.dtype)
            #         param.add_(delta)

            # 只对指定层应用扰动
            with torch.no_grad():
                for name in perturb_names:
                    param = dict(model.named_parameters())[name]
                    delta = xv * perturb_x[name].to(param.dtype)
                    param.add_(delta)

                total_loss = 0.0
                num_batches = 0
                for step, inputs in enumerate(dataloader):
                    if step >= max_batches:
                        break
                    inputs = self._prepare_inputs(inputs)
                    loss, _, _ = self.prediction_step(
                        model, inputs, prediction_loss_only=True, ignore_keys=None
                    )
                    total_loss += loss.item() if hasattr(loss, "item") else float(loss)
                    num_batches += 1
                loss_curve[i] = total_loss / num_batches

                if i % 5 == 0:
                    torch.cuda.empty_cache()

        
        
        # 恢复原始参数（直接复制最初保存的原始值）
        for name, param in model.named_parameters():
                param.data.copy_(checkpoint_state_dict[name])
        # 保存为 h5 文件
        with h5py.File(surf_file, 'w') as f:
            f.create_dataset('xcoordinates', data=x_coords)
            f.create_dataset('train_loss', data=loss_curve)

        logger.info(f"1D Loss Curve computation complete. Results saved to {surf_file}")






    # def compute_loss_landscape_easy_1d(self, test_dataset: Dataset, output_dir, flat_eval_type='lossland_all',
    #                           x_range=(-0.5, 0.5), num_points=40, max_batches=1, sample_batches=False):
    #     """
    #     计算一维参数扰动下的损失曲线。
    #     仅在 x 方向（指定扰动方向）采样，分析损失随扰动强度的变化趋势。
    #     """
    #     args = self.args
    #     device = args.device

       

    #     # 文件路径
    #     surf_file = os.path.join(output_dir, f"{flat_eval_type}_T5large__testData_1d_NewSet_Try2_LongLora2.h5")
    #     os.makedirs(os.path.dirname(surf_file), exist_ok=True)

    #     # 记录日志
    #     logger.info(f"***** Running 1D Loss Curve Calculation: {flat_eval_type} *****")
    #     logger.info(f"Output Dir = {output_dir}, Output File = {surf_file}, Num points = {num_points}, max_batches = {max_batches}")

    #     model = self.model
    #     model = self._wrap_model(model, training=False)
    #     model = model.to(device=device)
    #     model.eval()


    #     # 保存完整原始参数（全模型保存，无论 lora 还是全模型扰动都统一还原用）
    #     checkpoint_state_dict = {name: param.detach().clone() for name, param in model.named_parameters()}


    #     # 保存原始参数，默认是对整个模型都进行可视化
    #     # original_params_to_perturb = {}
    #     # for name, param in model.named_parameters():
    #     #     # if param.requires_grad:
    #     #     #     original_params_to_perturb[name] = param.data.clone()
    #     #     if flat_eval_type == "lossland_all":
    #     #         original_params_to_perturb[name] = param.data.clone()
    #     #     elif flat_eval_type == "lossland_lora":
    #     #         if param.requires_grad and 'lora' in name:
    #     #             original_params_to_perturb[name] = param.data.clone()
    #     # 只选择要扰动的参数
    #     perturb_names = []
    #     for name, param in model.named_parameters():
    #         if flat_eval_type == "lossland_all":
    #             perturb_names.append(name)
    #         elif flat_eval_type == "lossland_lora":
    #             if "lora_" in name and param.requires_grad:
    #                 perturb_names.append(name)


    #     # 固定扰动方向 d_x
    #     torch.manual_seed(42)
    #     perturb_x = {}
    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if name not in perturb_names:
    #                 continue
    #             d_x = torch.randn_like(param)
    #             d_x = (d_x / (d_x.norm() + 1e-8)) * (param.norm() + 1e-8)
    #             perturb_x[name] = d_x.to(device)

    #     # x 方向采样点
    #     x_coords = np.linspace(x_range[0], x_range[1], num_points)
    #     if 0.0 not in x_coords:
    #         x_coords = np.insert(x_coords, len(x_coords) // 2, 0.0)
    #         x_coords = np.unique(np.sort(x_coords))
    #     logger.info(f"x_coords: {x_coords}")


    #      # 最终损失向量
    #     loss_curve = np.zeros(len(x_coords))

    #     # 数据加载
    #     logger.info(f"Begin loading data: DDP={torch.distributed.is_initialized()}")
    #     dataloader = self.get_test_dataloader(test_dataset)

    #     # 主计算循环：只遍历 x 轴
    #     for i, xv in enumerate(tqdm(x_coords, desc="1D Loss Curve")):
    #         # 恢复参数
    #         # for name in original_params_to_perturb:
    #         #     model.state_dict()[name].copy_(original_params_to_perturb[name])
    #         # 确保还原所有参数到 checkpoint 状态
    #         for name, param in model.named_parameters():
    #             param.data.copy_(checkpoint_state_dict[name])

    #         # 应用 x 方向扰动
    #         # with torch.no_grad():
    #         #     for name in original_params_to_perturb:
    #         #         param = model.state_dict()[name]
    #         #         delta = xv * perturb_x[name].to(param.dtype)
    #         #         param.add_(delta)

    #         # 只对指定层应用扰动
    #         with torch.no_grad():
    #             for name in perturb_names:
    #                 param = dict(model.named_parameters())[name]
    #                 delta = xv * perturb_x[name].to(param.dtype)
    #                 param.add_(delta)

    #             total_loss = 0.0
    #             num_batches = 0
    #             for step, inputs in enumerate(dataloader):
    #                 if step >= max_batches:
    #                     break
    #                 inputs = self._prepare_inputs(inputs)
    #                 loss, _, _ = self.prediction_step(
    #                     model, inputs, prediction_loss_only=True, ignore_keys=None
    #                 )
    #                 total_loss += loss.item() if hasattr(loss, "item") else float(loss)
    #                 num_batches += 1
    #             loss_curve[i] = total_loss / num_batches

    #             if i % 5 == 0:
    #                 torch.cuda.empty_cache()

        
        
        
    #     for name, param in model.named_parameters():
    #             param.data.copy_(checkpoint_state_dict[name])
    #     # 保存为 h5 文件
    #     with h5py.File(surf_file, 'w') as f:
    #         f.create_dataset('xcoordinates', data=x_coords)
    #         f.create_dataset('train_loss', data=loss_curve)

    #     logger.info(f"1D Loss Curve computation complete. Results saved to {surf_file}")


    


