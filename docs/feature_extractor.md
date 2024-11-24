# Feature Extractor Documentation

The Feature Extractor is a core component of the adaptive JIT compiler that extracts both static and runtime features from code for machine learning-based optimization decisions.

## Overview

The Feature Extractor processes LLVM IR code and runtime performance data to generate feature vectors that capture important characteristics of the code being optimized.

## Components

### CodeFeatures Class

A dataclass that holds various metrics extracted from LLVM IR:

```python
@dataclass
class CodeFeatures:
    basic_block_count: int      # Number of basic blocks in the function
    instruction_count: int      # Total number of instructions
    memory_ops: int            # Memory operations (load, store, alloca)
    float_ops: int            # Floating-point arithmetic operations
    integer_ops: int          # Integer arithmetic operations
    branch_count: int         # Number of branch instructions
    function_calls: int       # Number of function calls
    loop_count: int          # Number of loops detected
```

### FeatureExtractor Class

The main class responsible for feature extraction and normalization:

#### Static Feature Extraction

```python
def extract_static_features(self, ir_code: str) -> CodeFeatures:
```

Extracts static features from LLVM IR code:
- Identifies basic blocks by labels ending with ':'
- Counts different types of instructions
- Detects loops in the code
- Returns a CodeFeatures object

#### Runtime Feature Extraction

```python
def extract_runtime_features(self, runtime_data: Dict[str, Any]) -> np.ndarray:
```

Extracts runtime features from execution data:
- Execution time
- Memory usage
- Cache misses
- Branch mispredictions

#### Feature Normalization

```python
def normalize_features(self, features) -> np.ndarray:
```

Normalizes features using mean and standard deviation:
- Handles both numpy arrays and CodeFeatures objects
- Automatically computes mean and standard deviation
- Handles zero standard deviations
- Supports both batch and single-sample normalization

## Usage Example

```python
from src.ml_engine.feature_extractor import FeatureExtractor, CodeFeatures

# Initialize the feature extractor
extractor = FeatureExtractor()

# Extract static features from LLVM IR
ir_code = """
define i32 @test_function(i32 %a, i32 %b) {
entry:
  %result = alloca i32
  store i32 0, i32* %result
  ret i32 %ret
}
"""
static_features = extractor.extract_static_features(ir_code)

# Extract runtime features
runtime_data = {
    'execution_time': 0.1,
    'memory_usage': 1024,
    'cache_misses': 10,
    'branch_mispredictions': 2
}
runtime_features = extractor.extract_runtime_features(runtime_data)

# Normalize features
normalized_features = extractor.normalize_features(static_features)
```

## Error Handling

The Feature Extractor includes comprehensive error handling:
- Logs errors using Python's logging framework
- Provides descriptive error messages
- Gracefully handles edge cases (empty IR, missing runtime data)

## Testing

The Feature Extractor comes with a comprehensive test suite:
- Unit tests for all major functions
- Integration tests with sample LLVM IR
- Edge case testing
- Floating-point comparison handling

Run the tests using:
```bash
python run_tests.py
```
