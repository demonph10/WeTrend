import os
import re
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.auto.tokenization_auto import AutoTokenizer

from swift.plugin import MeanMetric
from swift.plugin.loss import LossType, register_loss_func
from swift.utils import get_logger

logger = get_logger()

# 全局tokenizer缓存
_tokenizer_cache = None

# 数字token缓存 (0-9 对应 token_id 15-24)
DIGIT_TOKENS = torch.tensor([15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=torch.long)
DIGIT_TOKEN_SET = set(range(15, 25))
SEPARATOR_TOKENS = {11, 220, 58, 60}  # , space [ ]

# 注册新的损失类型
# 使用字符串常量以避免 linter 错误
LossType.adaptive_ce = 'adaptive_ce'

def get_qwen_tokenizer():
    """获取Qwen tokenizer，支持缓存避免重复加载"""
    global _tokenizer_cache
    
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    
    # 尝试从环境变量获取模型路径
    model_path = os.environ.get('MODEL_PATH', './Qwen3-8B-base')
    
    try:
        logger.info(f"加载Qwen tokenizer: {model_path}")
        _tokenizer_cache = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"✓ Qwen tokenizer加载成功")
        return _tokenizer_cache
    except Exception as e:
        logger.error(f"✗ 加载Qwen tokenizer失败: {e}")
        raise RuntimeError(f"无法加载Qwen tokenizer，请设置环境变量QWEN_MODEL_PATH或确保模型路径正确")


def create_digit_mask_from_tokens(token_ids: torch.Tensor) -> torch.Tensor:
    """从token序列创建数字位置mask
    
    Args:
        token_ids: token序列 [batch_size, seq_len] 或 [seq_len]
        
    Returns:
        torch.Tensor: 数字token位置的布尔mask
    """
    device = token_ids.device
    digit_tokens_device = DIGIT_TOKENS.to(device)
    
    # 使用broadcasting比较，创建数字token的mask
    is_digit = (token_ids.unsqueeze(-1) == digit_tokens_device).any(dim=-1) 
    
    return is_digit


def create_separator_mask_from_tokens(token_ids: torch.Tensor) -> torch.Tensor:
    """从token序列创建分隔符位置mask
    
    Args:
        token_ids: token序列 [batch_size, seq_len] 或 [seq_len]
        
    Returns:
        torch.Tensor: 分隔符token位置的布尔mask
    """
    device = token_ids.device
    separator_tokens_device = torch.tensor(list(SEPARATOR_TOKENS), dtype=torch.long, device=device)
    
    # 使用broadcasting比较，创建分隔符token的mask
    is_separator = (token_ids.unsqueeze(-1) == separator_tokens_device).any(dim=-1)
    
    return is_separator


def create_bracket_weighted_digit_loss_weights(token_ids: torch.Tensor, 
                                             digit_mask: torch.Tensor) -> torch.Tensor:
    """
    - 从后往前扫描token序列
    - 遇到 ] (token_id=60) 作为开始标记，开始处理方括号内的数字
    - 遇到 [ (token_id=58) 作为结束标记，停止处理方括号内的数字
    - 遇到 , (token_id=11) 时重置位数计数
    - 个位权重很小，高位权重逐渐增大，最长9位
    
    Args:
        token_ids: token序列 [batch_size, seq_len]
        digit_mask: 数字token位置mask [batch_size, seq_len]
        
    Returns:
        torch.Tensor: 权重张量 [batch_size, seq_len]
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    weights = torch.ones_like(token_ids, dtype=torch.float32, device=device)
    
    # 定义特殊token
    LEFT_BRACKET_TOKEN = 58   # [
    RIGHT_BRACKET_TOKEN = 60  # ]
    COMMA_TOKEN = 11          # ,
    
    for batch_idx in range(batch_size):
        digit_positions = digit_mask[batch_idx]
        tokens = token_ids[batch_idx]
        
        # 从后往前扫描，遇到]开始，遇到[结束
        inside_brackets = False
        position_in_number = 0  # 0=个位，1=十位，2=百位...

        # 衰减权重
        weight_list = 10**np.arange(0, 9)
        weight_list = np.log2(weight_list+1)/np.log2(10**8+1)
        
        for pos in range(seq_len - 1, -1, -1):  # 从后往前扫描
            current_token = tokens[pos].item()
            
            if current_token == RIGHT_BRACKET_TOKEN:
                # 遇到 ]，开始方括号内的处理
                inside_brackets = True
                position_in_number = 0  # 重置位数计数
                weights[batch_idx, pos] = 1  # ] 本身权重为1
                
            elif current_token == LEFT_BRACKET_TOKEN:
                # 遇到 [，结束方括号内的处理
                inside_brackets = False
                position_in_number = 0
                weights[batch_idx, pos] = 1  # [ 本身权重为1
                
            elif current_token == COMMA_TOKEN and inside_brackets:
                # 在方括号内遇到 ,，重置位数计数
                position_in_number = 0
                weights[batch_idx, pos] = 1  # , 本身权重为1
                
            elif digit_positions[pos] and inside_brackets:
                # 方括号内的数字token，从低位开始递增权重
                weight = weight_list[position_in_number]
                
                weights[batch_idx, pos] = weight
                position_in_number += 1  # 下一位（更高位）
                
            elif digit_positions[pos] and not inside_brackets:
                # 方括号外的数字，作为文本处理（权重为1）
                weights[batch_idx, pos] = 1.0
                
            else:
                # 其他token，保持权重为1（正常文本损失）
                weights[batch_idx, pos] = 1.0
    
    return weights

@register_loss_func(LossType.adaptive_ce)
def adaptive_ce_loss(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """    
    Args:
        outputs: 模型输出，包含logits
        labels: 标签tensor
        num_items_in_batch: 批次中的项目数量
        
    Returns:
        torch.Tensor: 组合损失值
    """
    logits = outputs.logits
    device = logits.device

    # 计算基础的交叉熵损失（标准的shift操作）
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    
    # 创建有效token mask
    valid_mask = shift_labels != -100
    
    # 计算所有token的CE损失
    flat_shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_shift_labels = shift_labels.reshape(-1)
    flat_valid_mask = valid_mask.reshape(-1)
    
    loss_fct = CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
    
    # 创建数字mask
    digit_mask = create_digit_mask_from_tokens(shift_labels)
        
    # 使用方括号权重函数（已包含所有权重：方括号外=1，方括号内=递增）
    digit_weights = create_bracket_weighted_digit_loss_weights(
        shift_labels, digit_mask
    )
        
    # 应用权重到所有有效token
    flat_digit_weights = digit_weights.reshape(-1)
    weighted_token_losses = token_losses * flat_valid_mask.float() * flat_digit_weights
        
    # 计算加权损失
    if num_items_in_batch is None:
        total_loss = weighted_token_losses.sum() / (flat_valid_mask.sum().float() + 1e-8)
    else:
        total_loss = weighted_token_losses.sum() / num_items_in_batch
    
    return total_loss