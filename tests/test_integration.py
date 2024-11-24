import unittest
import sys
import os
from pathlib import Path
import tempfile
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml_engine.feature_extractor import FeatureExtractor
from src.ml_engine.trainer import ModelTrainer
from tests.benchmarks.benchmark_suite import MatrixMultiplication, RecursiveFibonacci

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory for test artifacts
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = Path(cls.temp_dir) / 'test_model.h5'
        
        # Initialize components
        cls.feature_extractor = FeatureExtractor()
        cls.trainer = ModelTrainer(
            str(project_root / 'src' / 'ml_engine' / 'config' / 'model_config.json'))
        
        # Create benchmarks
        cls.matrix_mult = MatrixMultiplication(sizes=[64, 128])  # Smaller sizes for testing
        cls.fibonacci = RecursiveFibonacci(values=[10, 15])  # Smaller values for testing
    
    def test_end_to_end_workflow(self):
        # 1. Generate training data from benchmarks
        training_data = self.collect_training_data()
        
        # 2. Train model
        X, y = self.trainer.prepare_training_data(
            training_data['ir_samples'],
            training_data['runtime_data'],
            training_data['optimization_choices']
        )
        
        self.trainer.train(X, y, epochs=2)  # Just 2 epochs for testing
        
        # 3. Save and reload model
        self.trainer.save_model(str(self.model_path))
        self.assertTrue(self.model_path.exists())
        
        # 4. Make predictions
        test_features = self.feature_extractor.extract_static_features(
            training_data['ir_samples'][0])
        test_runtime = self.feature_extractor.extract_runtime_features(
            training_data['runtime_data'][0])
        
        combined_features = self.feature_extractor.combine_features(
            test_features, test_runtime)
        
        predictions = self.trainer.model.predict(combined_features.reshape(1, -1))
        
        # Verify predictions
        self.assertEqual(predictions.shape, (1, 5))  # 5 optimization choices
        self.assertTrue(all(0 <= p <= 1 for p in predictions[0]))  # Probabilities
    
    def collect_training_data(self):
        """Collect training data from benchmarks."""
        ir_samples = []
        runtime_data = []
        optimization_choices = []
        
        # Run matrix multiplication benchmark
        matrix_results = self.matrix_mult.run()
        for size, metrics in matrix_results.items():
            ir_samples.append(self.matrix_mult.generate_code(int(size.split('_')[1])))
            runtime_data.append({
                'execution_time': metrics['execution_time'],
                'memory_usage': size.split('_')[1],  # Use size as proxy for memory
                'cache_misses': 0,  # Placeholder
                'branch_mispredictions': 0  # Placeholder
            })
            # Assign optimization choice based on size
            optimization_choices.append(min(4, int(int(size.split('_')[1]) / 64)))
        
        # Run Fibonacci benchmark
        fib_results = self.fibonacci.run()
        for n, metrics in fib_results.items():
            ir_samples.append(self.fibonacci.generate_code(int(n.split('_')[1])))
            runtime_data.append({
                'execution_time': metrics['execution_time'],
                'memory_usage': 1024,  # Placeholder
                'cache_misses': 0,  # Placeholder
                'branch_mispredictions': 0  # Placeholder
            })
            # Assign optimization choice based on input size
            optimization_choices.append(min(4, int(int(n.split('_')[1]) / 5)))
        
        return {
            'ir_samples': ir_samples,
            'runtime_data': runtime_data,
            'optimization_choices': optimization_choices
        }
    
    def test_benchmark_execution(self):
        # Test matrix multiplication benchmark
        matrix_results = self.matrix_mult.run()
        self.assertTrue(len(matrix_results) > 0)
        for size, metrics in matrix_results.items():
            self.assertIn('compilation_time', metrics)
            self.assertIn('execution_time', metrics)
            self.assertIn('matrix_size', metrics)
        
        # Test Fibonacci benchmark
        fib_results = self.fibonacci.run()
        self.assertTrue(len(fib_results) > 0)
        for n, metrics in fib_results.items():
            self.assertIn('compilation_time', metrics)
            self.assertIn('execution_time', metrics)
            self.assertIn('input_size', metrics)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        for file in Path(cls.temp_dir).glob('*'):
            file.unlink()
        os.rmdir(cls.temp_dir)

if __name__ == '__main__':
    unittest.main()
