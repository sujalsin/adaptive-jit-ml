import unittest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml_engine.feature_extractor import FeatureExtractor, CodeFeatures
from src.ml_engine.trainer import ModelTrainer

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = FeatureExtractor()
        
        # Sample LLVM IR for testing
        self.sample_ir = """
define i32 @test_function(i32 %a, i32 %b) {
entry:
  %result = alloca i32
  store i32 0, i32* %result
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %sum = add i32 %a, %b
  store i32 %sum, i32* %result
  br label %return

if.else:
  %diff = sub i32 %b, %a
  store i32 %diff, i32* %result
  br label %return

return:
  %ret = load i32, i32* %result
  ret i32 %ret
}
"""
    
    def test_extract_static_features(self):
        features = self.feature_extractor.extract_static_features(self.sample_ir)
        
        # Verify extracted features
        self.assertIsInstance(features, CodeFeatures)
        self.assertEqual(features.basic_block_count, 4)  # entry, if.then, if.else, return
        self.assertTrue(features.instruction_count > 0)
        self.assertTrue(features.branch_count > 0)
    
    def test_extract_runtime_features(self):
        runtime_data = {
            'execution_time': 0.1,
            'memory_usage': 1024,
            'cache_misses': 10,
            'branch_mispredictions': 2
        }
        
        features = self.feature_extractor.extract_runtime_features(runtime_data)
        
        self.assertEqual(len(features), 4)
        self.assertAlmostEqual(features[0], 0.1, places=7)
        self.assertAlmostEqual(features[1], 1024.0, places=7)
    
    def test_normalize_features(self):
        features = self.feature_extractor.normalize_features(
            self.feature_extractor.extract_runtime_features({
                'execution_time': 100,
                'memory_usage': 2048,
                'cache_misses': 20,
                'branch_mispredictions': 5
            })
        )
        
        # Check if features are normalized to [0, 1]
        self.assertTrue(all(0 <= f <= 1 for f in features))

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        config_path = project_root / 'src' / 'ml_engine' / 'config' / 'model_config.json'
        self.trainer = ModelTrainer(str(config_path))
    
    def test_model_creation(self):
        # Test if model is created with correct input dimensions
        self.trainer.create_model(input_dim=12)
        
        self.assertIsNotNone(self.trainer.model)
        # Check input shape
        self.assertEqual(self.trainer.model.layers[0].input_shape, (None, 12))
        # Check output shape (num_optimization_choices = 5 from config)
        self.assertEqual(self.trainer.model.layers[-1].output_shape, (None, 5))
    
    def test_data_preparation(self):
        # Create sample data
        ir_samples = [self.generate_sample_ir() for _ in range(5)]
        runtime_data = [self.generate_sample_runtime_data() for _ in range(5)]
        optimization_results = [2, 3, 1, 4, 2]  # Sample optimization choices
        
        X, y = self.trainer.prepare_training_data(
            ir_samples, runtime_data, optimization_results)
        
        self.assertEqual(len(X), 5)  # Check number of samples
        self.assertEqual(y.shape[1], 5)  # Check one-hot encoding dimension
    
    def generate_sample_ir(self):
        return """
define i32 @sample(i32 %x) {
entry:
  %result = add i32 %x, 1
  ret i32 %result
}
"""
    
    def generate_sample_runtime_data(self):
        return {
            'execution_time': 0.05,
            'memory_usage': 512,
            'cache_misses': 5,
            'branch_mispredictions': 1
        }

if __name__ == '__main__':
    unittest.main()
