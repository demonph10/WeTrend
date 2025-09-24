import os
import re
from typing import List, Optional

import numpy as np

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


class Reward(ORM):
    
    def __init__(self):
        import importlib.util
        
        # 检查scipy是否可用（用于皮尔逊相关系数计算）
        if importlib.util.find_spec('scipy') is None:
            logger.warning("scipy包未安装，将使用简化的相关系数计算")
            self.has_scipy = False
        else:
            self.has_scipy = True
            from scipy.stats import pearsonr
            self.pearsonr = pearsonr
        
        # 获取权重配置
        self.accuracy_weight = float(os.environ.get('TS_REWARD_ACCURACY_WEIGHT', '0.6'))

    def extract_timeseries_from_text(self, text: str) -> List[float]:
        """从生成的文本中提取48小时的时间序列数据
        
        Args:
            text: 模型生成的文本，包含格式：[数值1, 数值2, ..., 数值48] 或 [数值1 数值2 ... 数值48]
            
        Returns:
            List[float]: 包含48个浮点数的列表
        """
        try:
            # 正则表达式匹配方括号内的数值序列，支持数字、空格、逗号
            pattern = r'\[([\d\s,]+)\]'
            match = re.search(pattern, text)
            
            if match:
                # 提取数值字符串
                numbers_str = match.group(1)
                
                # 使用正则表达式提取所有数字，支持空格和逗号分隔
                number_pattern = r'\d+'
                number_matches = re.findall(number_pattern, numbers_str)
                
                # 转换为浮点数
                numbers = [float(x) for x in number_matches if x.strip()]
                return numbers
            
            # 如果没有匹配到，返回全零序列
            return []
            
        except Exception as e:
            logger.warning(f"时间序列解析失败: {e}")
            return []

    def calculate_accuracy_reward(self, predicted_series: List[float], actual_series: List[float]) -> float:
        """计算准确度奖励 = 1 - 平均mse
        
        Args:
            predicted_series: 预测的48小时时间序列
            actual_series: 实际的48小时时间序列
            
        Returns:
            float: 准确度奖励 (0-1)
        """
        try:
            if len(predicted_series) < len(actual_series):
                predicted_series = predicted_series + [0.0] * (len(actual_series) - len(predicted_series))
            elif len(predicted_series) > len(actual_series):
                predicted_series = predicted_series[:len(actual_series)]
            
            # 转换为numpy数组进行计算
            pred_array = np.array(predicted_series, dtype=np.float32)
            actual_array = np.array(actual_series, dtype=np.float32)

            # 取对数
            pred_array = np.log10(pred_array + 1)
            actual_array = np.log10(actual_array + 1)
            
            # 计算逐点的MSE
            mse_value = (pred_array - actual_array) ** 2

            mse_value = np.clip(mse_value, 0.0, 1.0)  # 限制在0-1范围内

            mse_value = np.mean(mse_value)  # 计算平均MSE
            
            # 准确度奖励 = 1 - MSE
            accuracy_reward = 1.0 - mse_value
            
            return float(np.clip(accuracy_reward, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"准确度奖励计算失败: {e}")
            return 0.0
    
    def calculate_trend_reward(self, predicted_series: List[float], actual_series: List[float]) -> float:
        """计算趋势一致性奖励（基于皮尔逊相关系数）
        
        Args:
            predicted_series: 预测的48小时时间序列
            actual_series: 实际的48小时时间序列
            
        Returns:
            float: 趋势一致性奖励 (0-1)
        """
        try:
            if len(predicted_series) < len(actual_series):
                predicted_series = predicted_series + [0.0] * (len(actual_series) - len(predicted_series))
            elif len(predicted_series) > len(actual_series):
                predicted_series = predicted_series[:len(actual_series)]
            
            # 计算皮尔逊相关系数
            if self.has_scipy:
                correlation, _ = self.pearsonr(predicted_series, actual_series)
            else:
                # 简化的皮尔逊相关系数计算
                predicted_series = np.array(predicted_series)
                actual_series = np.array(actual_series)
                
                pred_mean = np.mean(predicted_series)
                actual_mean = np.mean(actual_series)
                
                numerator = np.sum((predicted_series - pred_mean) * (actual_series - actual_mean))
                denominator = np.sqrt(np.sum((predicted_series - pred_mean)**2) * np.sum((actual_series - actual_mean)**2))
                
                if denominator == 0:
                    correlation = 0.0
                else:
                    correlation = numerator / denominator
            
            # 处理NaN情况
            if np.isnan(correlation):
                correlation = 0.0
            
            # 将相关系数从 [-1, 1] 转换为 [0, 1]
            trend_reward = (correlation + 1.0) / 2.0
            
            return float(np.clip(trend_reward, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"预测序列: {predicted_series}")
            logger.warning(f"实际序列: {actual_series}")
            logger.warning(f"趋势奖励计算失败: {e}")
            exit(0)
    

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        计算时间序列综合奖励
        
        Args:
            completions: 模型输出文本列表
            solution: 标准答案文本列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 奖励分数列表
        """
        rewards = []
        
        accuracy_reward_all, trend_reward_all = [], []

        for completion, sol in zip(completions, solution):
            try:
                # 提取预测和实际的时间序列
                predicted_series = self.extract_timeseries_from_text(completion)
                actual_series = self.extract_timeseries_from_text(sol)
                
                # 计算奖励
                accuracy_reward = self.calculate_accuracy_reward(predicted_series, actual_series)
                trend_reward = self.calculate_trend_reward(predicted_series, actual_series)

                accuracy_reward = accuracy_reward * self.accuracy_weight
                trend_reward = trend_reward * (1 - self.accuracy_weight)
                
                accuracy_reward_all.append(accuracy_reward)
                trend_reward_all.append(trend_reward)
                
                # 加权组合奖励
                total_reward = (
                    accuracy_reward + trend_reward
                )
                
                # 确保奖励在 [0, 1] 范围内
                total_reward = max(0.0, min(1.0, total_reward))
                
                rewards.append(total_reward)
                
                
            except Exception as e:
                logger.warning(f"时间序列奖励计算失败: {e}")
                rewards.append(0.0)
        
        accuracy_reward_all = np.mean(accuracy_reward_all) if accuracy_reward_all else 0.0
        trend_reward_all = np.mean(trend_reward_all) if trend_reward_all else 0.0
        
        logger.info(f"【奖励】准确度: {accuracy_reward_all:.4f}, 趋势一致性: {trend_reward_all:.4f}")
        
        return rewards


# 注册新的奖励函数
orms['reward'] = Reward 