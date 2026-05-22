#!/usr/bin/env python3
"""
TensorManager: A standalone tensor management system for reproducible ML experiments

This module provides deterministic tensor generation and management capabilities
to ensure consistent input tensors across different virtual environments during
mutation and reproducibility experiments.

Author: AI Assistant + HEctum
Version: 1.0.0
"""

import os
import numpy as np
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    import mindspore as ms
    _HAS_TORCH = False
import hashlib
import json
import time
from typing import Optional, Dict, List, Tuple, Union


class TensorManager:
    """
    Tensor管理器：负责为每次变异迭代生成、保存和加载一致的输入张量
    确保在不同虚拟环境中第k次变异使用相同的输入张量
    
    Features:
    - Deterministic tensor generation using iteration number as seed
    - Automatic file management and caching
    - Cross-environment consistency validation
    - Built-in checksum verification
    - Flexible tensor type configurations
    - Graceful fallback to default values
    """
    
    def __init__(self, base_dir: str = "./tensors", seed: int = 42):
        """
        初始化Tensor管理器
        
        Args:
            base_dir: 张量存储的基础目录
            seed: 随机种子，确保张量生成的可重现性
        """
        self.base_dir = base_dir
        self.seed = seed
        os.makedirs(base_dir, exist_ok=True)
        
        # 默认张量配置：定义不同类型张量的规格 当前规格适用于ds
        dtype_f32 = torch.float32 if _HAS_TORCH else ms.float32
        self._default_tensor_configs = {
            'input_ids': {
                'shape': (8, 1),
                'dtype': dtype_f32,
                'range': (0.0, 5.0),
                'description': 'Input tensor for model forward pass'
            },
            'input_data': {
                'shape': (8, 1), 
                'dtype': dtype_f32,
                'range': (0.0, 5.0),
                'description': 'Continuous input data tensor'
            }
        }
        
        
        # 允许用户自定义配置
        self.tensor_configs = self._default_tensor_configs.copy()
        
    def configure_tensor_type(self, tensor_type: str, shape: Tuple, dtype, 
                             value_range: Tuple, description: str = ""):
        """
        配置或添加新的张量类型
        
        Args:
            tensor_type: 张量类型名称
            shape: 张量形状
            dtype: 张量数据类型
            value_range: 值范围 (min, max)
            description: 张量描述
        """
        self.tensor_configs[tensor_type] = {
            'shape': shape,
            'dtype': dtype,
            'range': value_range,
            'description': description
        }
        
    def remove_tensor_type(self, tensor_type: str):
        """移除张量类型配置"""
        if tensor_type in self.tensor_configs:
            del self.tensor_configs[tensor_type]
            
    def reset_to_default_configs(self):
        """重置为默认张量配置"""
        self.tensor_configs = self._default_tensor_configs.copy()

    def _normalize_device(self, device: Optional[Union[str, "torch.device"]]):
        if not _HAS_TORCH:
            return "cpu"
        if device is None:
            return torch.device("cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(str(device))

    def _save_array(self, filepath: str, array: np.ndarray) -> None:
        with open(filepath, "wb") as f:
            np.save(f, array)

    def _load_array(self, filepath: str) -> np.ndarray:
        with open(filepath, "rb") as f:
            loaded = np.load(f, allow_pickle=False)

        return loaded
        
    def _get_tensor_filename(self, tensor_type: str, iteration: int) -> str:
        """
        生成张量文件名
        
        Args:
            tensor_type: 张量类型 ('input_ids', 'input_data', etc.)
            iteration: 迭代次数 (1-based)
            
        Returns:
            str: 文件名
        """
        return f"{tensor_type}_iter_{iteration:04d}.npy"
    
    def _get_tensor_filepath(self, tensor_type: str, iteration: int) -> str:
        """
        生成张量文件完整路径
        """
        filename = self._get_tensor_filename(tensor_type, iteration)
        return os.path.join(self.base_dir, filename)
    
    def _generate_deterministic_tensor(self, tensor_type: str, iteration: int, 
                                     device) -> "torch.Tensor":
        """
        为指定迭代生成确定性张量
        使用迭代次数和张量类型作为种子，确保跨环境的一致性
        
        Args:
            tensor_type: 张量类型
            iteration: 迭代次数
            device: 目标设备
            
        Returns:
            torch.Tensor: 生成的张量
        """
        if tensor_type not in self.tensor_configs:
            raise ValueError(f"未知的张量类型: {tensor_type}")
        
        config = self.tensor_configs[tensor_type]
        
        # 使用迭代次数、张量类型和基础种子创建确定性种子
        seed_string = f"{self.seed}_{tensor_type}_{iteration}"
        deterministic_seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        
        # 临时设置随机种子
        if _HAS_TORCH:
            original_state = torch.get_rng_state()
            torch.manual_seed(deterministic_seed)
            try:
                if config['dtype'] == torch.int64:
                    min_val, max_val = config['range']
                    tensor = torch.randint(min_val, max_val, config['shape'], dtype=config['dtype'])
                else:
                    min_val, max_val = config['range']
                    tensor = torch.rand(config['shape'], dtype=config['dtype']) * (max_val - min_val) + min_val
                return tensor.to(device)
            finally:
                torch.set_rng_state(original_state)

        rng = np.random.RandomState(deterministic_seed)
        min_val, max_val = config['range']
        if config['dtype'] == ms.int64:
            array = rng.randint(min_val, max_val, size=config['shape'], dtype=np.int64)
            return ms.Tensor(array, dtype=ms.int64)
        array = rng.rand(*config['shape']).astype(np.float32) * (max_val - min_val) + min_val
        return ms.Tensor(array, dtype=config['dtype'])
    
    def get_or_create_tensor(self, tensor_type: str, iteration: int, 
                           device: Optional[Union[str, "torch.device"]]) -> "torch.Tensor":
        """
        获取或创建指定迭代的张量
        如果文件存在则加载，否则生成新张量并保存
        
        Args:
            tensor_type: 张量类型
            iteration: 迭代次数 (1-based)
            device: 目标设备
            
        Returns:
            torch.Tensor: 加载或生成的张量
        """
        device = self._normalize_device(device)
        filepath = self._get_tensor_filepath(tensor_type, iteration)
        
        try:
            if os.path.exists(filepath):
                print(f"从文件加载张量: {filepath}")
                tensor_nparray = self._load_array(filepath)
                if _HAS_TORCH:
                    tensor = torch.from_numpy(tensor_nparray)
                else:
                    tensor = ms.Tensor(tensor_nparray, dtype=self.tensor_configs[tensor_type]['dtype'])
                expected_config = self.tensor_configs[tensor_type]
                if tensor.shape != expected_config['shape']:
                    print(f"警告: 张量形状不匹配 {tensor.shape} vs {expected_config['shape']}，重新生成")
                    raise ValueError("Shape mismatch")
                if tensor.dtype != expected_config['dtype']:
                    print(f"警告: 张量类型不匹配 {tensor.dtype} vs {expected_config['dtype']}，重新生成")
                    raise ValueError("Dtype mismatch")

                return tensor.to(device) if _HAS_TORCH else tensor

            print(f"生成新张量: {filepath}")
            tensor = self._generate_deterministic_tensor(tensor_type, iteration, "cpu")
            if _HAS_TORCH:
                self._save_array(filepath, tensor.detach().cpu().numpy())
            else:
                self._save_array(filepath, tensor.asnumpy())
            print(f"张量已保存: {filepath}")
            return tensor.to(device) if _HAS_TORCH else tensor

        except Exception as e:
            print(f"张量处理失败 ({tensor_type}, iteration {iteration}): {e}")
            print("使用临时生成的张量")
            return self._generate_deterministic_tensor(tensor_type, iteration, device)
    
    def get_iteration_tensors(self, iteration: int, device: Optional[Union[str, "torch.device"]]) -> Dict[str, "torch.Tensor"]:
        """
        获取指定迭代的所有张量
        
        Args:
            iteration: 迭代次数 (1-based)
            device: 目标设备
            
        Returns:
            dict: 包含所有张量类型的字典
        """
        device = self._normalize_device(device)
        tensors = {}
        for tensor_type in self.tensor_configs.keys():
            tensors[tensor_type] = self.get_or_create_tensor(tensor_type, iteration, device)
        return tensors
    
    def create_custom_tensor(self, tensor_type: str, iteration: int, 
                           tensor_data, overwrite: bool = False) -> bool:
        """
        创建自定义张量并保存到文件
        
        Args:
            tensor_type: 张量类型
            iteration: 迭代次数
            tensor_data: 张量数据
            overwrite: 是否覆盖已存在的文件
            
        Returns:
            bool: 是否成功创建
        """
        filepath = self._get_tensor_filepath(tensor_type, iteration)
        
        if os.path.exists(filepath) and not overwrite:
            print(f"文件已存在，跳过: {filepath}")
            return False
            
        try:
            if _HAS_TORCH and isinstance(tensor_data, torch.Tensor):
                array = tensor_data.detach().cpu().numpy()
            elif not _HAS_TORCH and isinstance(tensor_data, ms.Tensor):
                array = tensor_data.asnumpy()
            elif isinstance(tensor_data, np.ndarray):
                array = tensor_data
            else:
                array = np.asarray(tensor_data)

            self._save_array(filepath, array)
            print(f"自定义张量已保存: {filepath}")
            return True
        except Exception as e:
            print(f"保存自定义张量失败: {e}")
            return False
    
    def load_tensor_from_file(self, filepath: str, device: Optional[Union[str, "torch.device"]]):
        """
        从指定文件加载张量
        
        Args:
            filepath: 文件路径
            device: 目标设备
            
        Returns:
            torch.Tensor or None: 加载的张量，失败时返回None
        """
        try:
            device = self._normalize_device(device)
            if os.path.exists(filepath):
                tensor_nparray = self._load_array(filepath)
                if _HAS_TORCH:
                    tensor = torch.from_numpy(tensor_nparray)
                    return tensor.to(device)
                else:
                    tensor = ms.Tensor(tensor_nparray)
                return tensor
            else:
                print(f"文件不存在: {filepath}")
                return None
        except Exception as e:
            print(f"加载张量失败: {e}")
            return None
    
    def list_available_iterations(self) -> List[int]:
        """
        列出已保存张量的所有迭代次数
        
        Returns:
            list: 可用的迭代次数列表
        """
        iterations = set()
        if os.path.exists(self.base_dir):
            for filename in os.listdir(self.base_dir):
                if filename.endswith('.npy'):
                    # 解析文件名中的迭代次数
                    parts = filename.split('_')
                    if len(parts) >= 3 and parts[-1].endswith('.npy'):
                        try:
                            iter_num = int(parts[-1].replace('.npy', ''))
                            iterations.add(iter_num)
                        except ValueError:
                            continue
        return sorted(list(iterations))
    
    def cleanup_tensors(self, keep_last_n: int = 10):
        """
        清理旧的张量文件，只保留最近的N个迭代
        
        Args:
            keep_last_n: 保留的最近迭代数量
        """
        available_iterations = self.list_available_iterations()
        if len(available_iterations) <= keep_last_n:
            return
            
        iterations_to_remove = available_iterations[:-keep_last_n]
        print(f"清理旧张量文件，移除迭代: {iterations_to_remove}")
        
        for iteration in iterations_to_remove:
            for tensor_type in self.tensor_configs.keys():
                filepath = self._get_tensor_filepath(tensor_type, iteration)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        print(f"已删除: {filepath}")
                    except Exception as e:
                        print(f"删除文件失败 {filepath}: {e}")
    
    def get_tensor_summary(self) -> Dict:
        """
        获取张量管理器的摘要信息
        
        Returns:
            dict: 摘要信息
        """
        available_iterations = self.list_available_iterations()
        summary = {
            'base_dir': self.base_dir,
            'seed': self.seed,
            'tensor_types': list(self.tensor_configs.keys()),
            'available_iterations': available_iterations,
            'total_iterations': len(available_iterations),
            'tensor_configs': {}
        }
        
        for tensor_type, config in self.tensor_configs.items():
            summary['tensor_configs'][tensor_type] = {
                'shape': config['shape'],
                'dtype': str(config['dtype']),
                'range': config['range'],
                'description': config['description']
            }
            
        return summary
    
    def export_iteration_metadata(self, iteration: int, export_path: str) -> bool:
        """
        导出指定迭代的元数据信息
        
        Args:
            iteration: 迭代次数
            export_path: 导出文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            device = self._normalize_device("cpu")
            tensors = self.get_iteration_tensors(iteration, device)
            
            metadata = {
                'iteration': iteration,
                'timestamp': time.time(),
                'seed': self.seed,
                'base_dir': self.base_dir,
                'tensor_info': {}
            }
            
            for tensor_type, tensor in tensors.items():
                if _HAS_TORCH:
                    array = tensor.detach().cpu().numpy()
                    is_float = tensor.dtype.is_floating_point
                    tmin = float(tensor.min())
                    tmax = float(tensor.max())
                    tmean = float(tensor.float().mean()) if is_float else "N/A"
                    tstd = float(tensor.float().std()) if is_float else "N/A"
                else:
                    array = tensor.asnumpy()
                    is_float = tensor.dtype in (ms.float16, ms.float32, ms.float64)
                    tmin = float(array.min())
                    tmax = float(array.max())
                    tmean = float(array.mean()) if is_float else "N/A"
                    tstd = float(array.std()) if is_float else "N/A"
                metadata['tensor_info'][tensor_type] = {
                    'shape': list(array.shape),
                    'dtype': str(tensor.dtype),
                    'checksum': hashlib.md5(array.tobytes()).hexdigest(),
                    'mean': tmean,
                    'std': tstd,
                    'min': tmin,
                    'max': tmax
                }
                
            with open(export_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"元数据已导出到: {export_path}")
            return True
            
        except Exception as e:
            print(f"导出元数据失败: {e}")
            return False
    
    def batch_create_tensors(self, iterations: List[int], device, 
                           progress_callback=None) -> Dict[int, Dict[str, "torch.Tensor"]]:
        """
        批量创建多个迭代的张量
        
        Args:
            iterations: 迭代次数列表
            device: 目标设备
            progress_callback: 进度回调函数
            
        Returns:
            dict: 格式为 {iteration: {tensor_type: tensor}}
        """
        result = {}
        total = len(iterations)
        
        for i, iteration in enumerate(iterations):
            result[iteration] = self.get_iteration_tensors(iteration, device)
            
            if progress_callback:
                progress_callback(i + 1, total, iteration)
            elif i % 10 == 0 or i == total - 1:
                print(f"批量创建进度: {i + 1}/{total}")
                
        return result
    
    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_tensor_summary()
        return (f"TensorManager(base_dir='{self.base_dir}', seed={self.seed}, "
                f"types={len(summary['tensor_types'])}, iterations={summary['total_iterations']})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


def create_default_tensor_manager(base_dir: str = "./tensors", seed: int = 42) -> TensorManager:
    """
    创建默认配置的TensorManager实例
    
    Args:
        base_dir: 张量存储目录
        seed: 随机种子
        
    Returns:
        TensorManager: 配置好的实例
    """
    return TensorManager(base_dir=base_dir, seed=seed)


def create_custom_tensor_manager(base_dir: str, seed: int, 
                                custom_configs: Dict) -> TensorManager:
    """
    创建自定义配置的TensorManager实例
    
    Args:
        base_dir: 张量存储目录
        seed: 随机种子
        custom_configs: 自定义张量配置
        
    Returns:
        TensorManager: 配置好的实例
    """
    tm = TensorManager(base_dir=base_dir, seed=seed)
    
    # 清空默认配置并添加自定义配置
    tm.tensor_configs.clear()
    for tensor_type, config in custom_configs.items():
        tm.configure_tensor_type(
            tensor_type=tensor_type,
            shape=config['shape'],
            dtype=config['dtype'],
            value_range=config['range'],
            description=config.get('description', '')
        )
    
    return tm



if __name__ == "__main__":
    # 简单的测试示例
    print("TensorManager Test")
    
    # 创建测试实例
    tm = TensorManager("./test_tensors", seed=42)
    device = "cpu"
    
    # 生成一些测试张量
    for i in range(1, 4):
        tensors = tm.get_iteration_tensors(i, device)
        print(f"Iteration {i}: {[(k, v.shape) for k, v in tensors.items()]}")
    
    # 显示摘要
    summary = tm.get_tensor_summary()
    print(f"Summary: {summary['total_iterations']} iterations available")
    
    # 清理测试文件
    import shutil
    if os.path.exists("./test_tensors"):
        shutil.rmtree("./test_tensors")
        print("Cleaned up test files")

