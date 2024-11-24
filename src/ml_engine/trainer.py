import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import json
from datetime import datetime
from .feature_extractor import FeatureExtractor

class ModelTrainer:
    """Training infrastructure for the JIT optimization model."""
    
    def __init__(self, config_path: str = None):
        """Initialize the model trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        self.logger = logging.getLogger('ModelTrainer')
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.training_history = []
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load training configuration from file.
        
        Args:
            config_path: Path to configuration JSON file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def create_model(self, input_dim: int):
        """Create the neural network model.
        
        Args:
            input_dim: Dimension of input features
        """
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.config.get('num_optimization_choices', 5),
                                   activation='softmax')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def prepare_training_data(self, 
                            ir_samples: List[str],
                            runtime_data: List[Dict],
                            optimization_results: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from IR samples and runtime data.
        
        Args:
            ir_samples: List of LLVM IR text samples
            runtime_data: List of runtime metric dictionaries
            optimization_results: List of optimization choices that worked well
            
        Returns:
            Tuple of (features, labels) arrays
        """
        features_list = []
        labels = []
        
        try:
            for ir, runtime, opt in zip(ir_samples, runtime_data, optimization_results):
                # Extract features
                static_features = self.feature_extractor.extract_static_features(ir)
                runtime_features = self.feature_extractor.extract_runtime_features(runtime)
                combined = self.feature_extractor.combine_features(
                    static_features, runtime_features)
                
                features_list.append(combined)
                labels.append(opt)
            
            # Convert to numpy arrays
            X = np.array(features_list)
            y = tf.keras.utils.to_categorical(labels, 
                num_classes=self.config.get('num_optimization_choices', 5))
            
            # Normalize features
            X = self.feature_extractor.normalize_features(X)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {str(e)}")
            raise
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray = None,
             y_val: np.ndarray = None,
             epochs: int = 100,
             batch_size: int = 32):
        """Train the model on prepared data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            if self.model is None:
                self.create_model(X_train.shape[1])
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f'models/model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history.append(history.history)
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            results = self.model.evaluate(X_test, y_test, verbose=0)
            metrics = dict(zip(self.model.metrics_names, results))
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def save_model(self, path: str):
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        try:
            self.model.save(path)
            
            # Save feature normalization parameters
            normalization_params = {
                'feature_means': self.feature_extractor.feature_means.tolist(),
                'feature_stds': self.feature_extractor.feature_stds.tolist()
            }
            
            params_path = Path(path).parent / 'normalization_params.json'
            with open(params_path, 'w') as f:
                json.dump(normalization_params, f)
                
            self.logger.info(f"Model and parameters saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
