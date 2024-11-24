import numpy as np
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

@dataclass
class CodeFeatures:
    basic_block_count: int = 0
    instruction_count: int = 0
    memory_ops: int = 0
    float_ops: int = 0
    integer_ops: int = 0
    branch_count: int = 0
    function_calls: int = 0
    loop_count: int = 0

class FeatureExtractor:
    """Extract and process features from LLVM IR for ML model input."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_cache = {}
        self.feature_means = None
        self.feature_stds = None
    
    def extract_static_features(self, ir_code: str) -> CodeFeatures:
        """Extract static features from LLVM IR code."""
        try:
            features = CodeFeatures()
            lines = ir_code.split('\n')
            
            # Count basic blocks by looking for labels ending with ':'
            current_block = None
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith(';'):
                    continue
                
                # Check for basic block labels
                if line.endswith(':') and not line.startswith('define'):
                    current_block = line[:-1]  # Remove the colon
                    features.basic_block_count += 1
                    continue
                
                # Count instructions
                if current_block is not None:
                    features.instruction_count += 1
                    
                    # Count specific instruction types
                    if 'alloca' in line or 'load' in line or 'store' in line:
                        features.memory_ops += 1
                    elif any(op in line for op in ['fadd', 'fmul', 'fdiv', 'fsub']):
                        features.float_ops += 1
                    elif any(op in line for op in ['add', 'mul', 'div', 'sub', 'shl', 'lshr']):
                        features.integer_ops += 1
                    elif 'br' in line or 'switch' in line:
                        features.branch_count += 1
                    elif 'call' in line:
                        features.function_calls += 1
                    
                # Count loops
                if 'for.cond' in line or 'while.cond' in line or 'do.body' in line:
                    features.loop_count += 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in extract_static_features: {str(e)}")
            raise
    
    def extract_runtime_features(self, runtime_data: Dict[str, Any]) -> np.ndarray:
        """Extract runtime features from execution data."""
        try:
            # Extract runtime features from the provided data
            features = np.array([
                float(runtime_data.get('execution_time', 0.0)),
                float(runtime_data.get('memory_usage', 0.0)),
                float(runtime_data.get('cache_misses', 0.0)),
                float(runtime_data.get('branch_mispredictions', 0.0))
            ], dtype=np.float32)
            return features
            
        except Exception as e:
            self.logger.error(f"Error in extract_runtime_features: {str(e)}")
            raise
    
    def combine_features(self, static_features: CodeFeatures, runtime_features: np.ndarray) -> np.ndarray:
        """Combine static and runtime features into a single vector."""
        static_vector = np.array([
            static_features.basic_block_count,
            static_features.instruction_count,
            static_features.memory_ops,
            static_features.float_ops,
            static_features.integer_ops,
            static_features.branch_count,
            static_features.function_calls,
            static_features.loop_count
        ], dtype=np.float32)
        
        return np.concatenate([static_vector, runtime_features])
    
    def normalize_features(self, features) -> np.ndarray:
        """Normalize features using mean and standard deviation."""
        try:
            # Convert input to numpy array if it's a list of CodeFeatures
            if isinstance(features, (list, tuple)) and len(features) > 0 and isinstance(features[0], CodeFeatures):
                features_array = np.array([
                    [
                        f.basic_block_count,
                        f.instruction_count,
                        f.memory_ops,
                        f.float_ops,
                        f.integer_ops,
                        f.branch_count,
                        f.function_calls,
                        f.loop_count
                    ] for f in features
                ], dtype=np.float32)
            else:
                features_array = np.array(features, dtype=np.float32)
                if features_array.ndim == 1:
                    features_array = features_array.reshape(1, -1)

            # Calculate mean and std if not already computed
            if self.feature_means is None or self.feature_stds is None:
                self.feature_means = np.mean(features_array, axis=0)
                self.feature_stds = np.std(features_array, axis=0)
                # Replace zero std with 1 to avoid division by zero
                self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)

            # Normalize features
            normalized_features = (features_array - self.feature_means) / self.feature_stds
            
            # If input was 1D, return 1D
            if len(features_array) == 1 and not isinstance(features, (list, tuple)):
                normalized_features = normalized_features.flatten()
                
            return normalized_features

        except Exception as e:
            self.logger.error(f"Error in normalize_features: {str(e)}")
            raise
