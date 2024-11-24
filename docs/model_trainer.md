# Model Trainer Documentation

The Model Trainer is responsible for training and managing the machine learning models used by the adaptive JIT compiler for optimization decisions.

## Overview

The Model Trainer takes features extracted from code and runtime performance data to train a neural network that predicts optimal optimization strategies.

## Components

### ModelTrainer Class

The main class for training and managing ML models:

```python
class ModelTrainer:
    def __init__(self, config_path: str):
        """Initialize the model trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_extractor = FeatureExtractor()
```

### Configuration

The trainer uses a JSON configuration file to specify model architecture and training parameters:

```json
{
    "model": {
        "input_dim": 12,
        "hidden_layers": [64, 32],
        "output_dim": 5,
        "activation": "relu",
        "output_activation": "softmax"
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "validation_split": 0.2,
        "learning_rate": 0.001,
        "early_stopping_patience": 10
    }
}
```

### Key Methods

#### Model Creation

```python
def create_model(self, input_dim: int) -> None:
```

Creates a neural network model with:
- Configurable input dimension
- Multiple hidden layers
- Customizable activation functions
- Output layer for optimization choices

#### Training Data Preparation

```python
def prepare_training_data(self,
    ir_samples: List[str],
    runtime_data: List[Dict],
    optimization_results: List[int]) -> Tuple[np.ndarray, np.ndarray]:
```

Prepares training data by:
- Extracting features from IR code
- Processing runtime metrics
- Normalizing features
- Creating one-hot encoded labels

#### Model Training

```python
def train(self,
    training_data: Tuple[np.ndarray, np.ndarray],
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
```

Trains the model with:
- Early stopping for overfitting prevention
- Learning rate scheduling
- Validation data monitoring
- Progress tracking

#### Model Persistence

```python
def save_model(self, model_path: str) -> None:
def load_model(self, model_path: str) -> None:
```

Handles model saving and loading:
- Saves model architecture and weights
- Stores normalization parameters
- Preserves feature extraction configuration

## Usage Example

```python
from src.ml_engine.trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer('config/model_config.json')

# Prepare training data
X_train, y_train = trainer.prepare_training_data(
    ir_samples=ir_code_samples,
    runtime_data=performance_metrics,
    optimization_results=optimization_choices
)

# Create and train model
trainer.create_model(input_dim=X_train.shape[1])
trainer.train(
    training_data=(X_train, y_train),
    validation_data=(X_val, y_val)
)

# Save trained model
trainer.save_model('models/jit_optimizer.h5')
```

## Training Process

1. **Data Collection**:
   - Gather LLVM IR code samples
   - Collect runtime performance metrics
   - Record optimization decisions and outcomes

2. **Feature Extraction**:
   - Extract static features from IR
   - Process runtime metrics
   - Combine and normalize features

3. **Model Training**:
   - Initialize model architecture
   - Train with early stopping
   - Validate performance
   - Save best model

4. **Evaluation**:
   - Test model accuracy
   - Validate optimization decisions
   - Monitor runtime improvements

## Best Practices

1. **Data Preparation**:
   - Use diverse code samples
   - Balance optimization choices
   - Normalize features appropriately

2. **Model Configuration**:
   - Adjust architecture based on data size
   - Tune hyperparameters
   - Use appropriate learning rates

3. **Training**:
   - Monitor validation metrics
   - Prevent overfitting
   - Save model checkpoints

4. **Deployment**:
   - Version control models
   - Track performance metrics
   - Update models periodically

## Testing

The Model Trainer includes tests for:
- Model creation and architecture
- Data preparation and normalization
- Training workflow
- Model persistence

Run the tests using:
```bash
python run_tests.py
```
