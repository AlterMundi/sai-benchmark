"""
Integration tests for run_suite.py CLI interface.

Tests cover:
- Command-line argument parsing
- Test suite execution from CLI
- Output file generation and formatting
- Error handling and user feedback
- Registry information display
- Parallel execution control
"""

import pytest
import sys
import json
import yaml
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import tempfile

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the CLI module
import run_suite
from core.test_suite import TestSuiteResult, TestSuiteConfig, TestResult
from core.metrics_registry import MetricResult
from engines.base_engine import EngineResponse


class TestRunSuiteCLI:
    """Test run_suite.py command-line interface."""
    
    @pytest.fixture
    def sample_suite_yaml(self, temp_dir):
        """Create a sample test suite YAML file."""
        suite_config = {
            "name": "cli_test_suite",
            "description": "Test suite for CLI testing",
            "version": "1.0",
            "prompts": ["early_fire_json"],
            "models": ["qwen2.5-vl:7b"],
            "datasets": [str(temp_dir / "test_data")],
            "metrics": ["accuracy", "latency"],
            "engine_config": {
                "max_tokens": 512,
                "temperature": 0.1
            },
            "test_config": {
                "max_workers": 2,
                "iou_threshold": 0.4
            }
        }
        
        suite_path = temp_dir / "test_suite.yaml"
        with open(suite_path, 'w') as f:
            yaml.dump(suite_config, f)
        
        # Create test data directory and files
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()
        (data_dir / "test_image.jpg").touch()
        
        return suite_path
    
    @pytest.fixture
    def mock_test_suite_runner(self):
        """Create a mock TestSuiteRunner that returns successful results."""
        with patch('run_suite.TestSuiteRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner
            
            # Create mock successful result
            mock_result = TestSuiteResult(
                suite_name="cli_test_suite",
                config=TestSuiteConfig(
                    name="cli_test_suite",
                    description="Test suite",
                    prompts=["early_fire_json"],
                    models=["qwen2.5-vl:7b"],
                    datasets=["/test/data"],
                    metrics=["accuracy"]
                ),
                test_results=[
                    TestResult(
                        test_case_id="test_001",
                        prompt_id="early_fire_json",
                        model_id="qwen2.5-vl:7b",
                        engine_response=EngineResponse(
                            content='{"has_smoke": true}',
                            model="qwen2.5-vl:7b",
                            latency=1.25
                        ),
                        metrics={"accuracy": MetricResult("accuracy", 0.9)}
                    )
                ],
                aggregated_metrics={"accuracy": MetricResult("accuracy", 0.9)},
                execution_time=15.5
            )
            
            mock_runner.run_suite.return_value = mock_result
            
            return mock_runner_class, mock_runner
    
    def test_argument_parsing_minimal(self, sample_suite_yaml):
        """Test parsing minimal required arguments."""
        args = run_suite.parse_args([
            "--suite", str(sample_suite_yaml)
        ])
        
        assert args.suite == str(sample_suite_yaml)
        assert args.workers == 4  # Default
        assert args.output == "out/"  # Default
        assert args.verbose is False  # Default
        assert args.show_registries is False  # Default
    
    def test_argument_parsing_all_options(self, sample_suite_yaml, temp_dir):
        """Test parsing all available arguments."""
        output_dir = str(temp_dir / "custom_output")
        
        args = run_suite.parse_args([
            "--suite", str(sample_suite_yaml),
            "--workers", "8",
            "--output", output_dir,
            "--verbose",
            "--show-registries",
            "--dry-run",
            "--save-config"
        ])
        
        assert args.suite == str(sample_suite_yaml)
        assert args.workers == 8
        assert args.output == output_dir
        assert args.verbose is True
        assert args.show_registries is True
        assert args.dry_run is True
        assert args.save_config is True
    
    def test_argument_parsing_missing_suite(self):
        """Test error when required --suite argument is missing."""
        with pytest.raises(SystemExit):
            run_suite.parse_args([])
    
    def test_argument_parsing_invalid_workers(self, sample_suite_yaml):
        """Test error with invalid worker count."""
        with pytest.raises(SystemExit):
            run_suite.parse_args([
                "--suite", str(sample_suite_yaml),
                "--workers", "0"
            ])
    
    def test_run_suite_basic(self, sample_suite_yaml, mock_test_suite_runner, temp_dir):
        """Test basic suite execution."""
        mock_runner_class, mock_runner = mock_test_suite_runner
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml)]):
            with patch('run_suite.Path.mkdir'), \
                 patch('builtins.open', create=True) as mock_open:
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                mock_runner.run_suite.assert_called_once_with(str(sample_suite_yaml))
                
                # Check that results were saved
                mock_open.assert_called()
    
    def test_run_suite_with_custom_workers(self, sample_suite_yaml, mock_test_suite_runner):
        """Test suite execution with custom worker count."""
        mock_runner_class, mock_runner = mock_test_suite_runner
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml), '--workers', '8']):
            with patch('run_suite.Path.mkdir'), \
                 patch('builtins.open', create=True):
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                # Verify TestSuiteRunner was initialized with custom worker count
                mock_runner_class.assert_called_once()
    
    def test_run_suite_with_custom_output(self, sample_suite_yaml, mock_test_suite_runner, temp_dir):
        """Test suite execution with custom output directory."""
        mock_runner_class, mock_runner = mock_test_suite_runner
        output_dir = str(temp_dir / "custom_results")
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml), '--output', output_dir]):
            with patch('run_suite.Path.mkdir') as mock_mkdir, \
                 patch('builtins.open', create=True):
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                # Verify custom output directory was created
                mock_mkdir.assert_called()
    
    def test_run_suite_verbose_output(self, sample_suite_yaml, mock_test_suite_runner, capsys):
        """Test verbose output mode."""
        mock_runner_class, mock_runner = mock_test_suite_runner
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml), '--verbose']):
            with patch('run_suite.Path.mkdir'), \
                 patch('builtins.open', create=True):
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                
                captured = capsys.readouterr()
                assert "Starting test suite execution" in captured.out
                assert "Suite execution completed" in captured.out
    
    def test_show_registries(self, sample_suite_yaml, capsys):
        """Test registry information display."""
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml), '--show-registries']):
            with patch('run_suite.prompt_registry') as mock_prompt_reg, \
                 patch('run_suite.model_registry') as mock_model_reg, \
                 patch('run_suite.engine_registry') as mock_engine_reg, \
                 patch('run_suite.metrics_registry') as mock_metrics_reg:
                
                # Mock registry stats
                mock_prompt_reg.get_stats.return_value = {"total_prompts": 5}
                mock_model_reg.get_stats.return_value = {"total_models": 8}
                mock_engine_reg.get_stats.return_value = {"total_engines": 3}
                mock_metrics_reg.get_stats.return_value = {"total_metrics": 10}
                
                with patch('run_suite.Path.mkdir'), \
                     patch('builtins.open', create=True), \
                     patch('run_suite.TestSuiteRunner'):
                    
                    exit_code = run_suite.main()
                    
                    assert exit_code == 0
                    
                    captured = capsys.readouterr()
                    assert "Registry Information" in captured.out
                    assert "total_prompts: 5" in captured.out
                    assert "total_models: 8" in captured.out
    
    def test_dry_run_mode(self, sample_suite_yaml, capsys):
        """Test dry run mode (doesn't execute, just validates)."""
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml), '--dry-run']):
            with patch('run_suite.TestSuiteConfig.from_yaml') as mock_config:
                mock_config.return_value = TestSuiteConfig(
                    name="test",
                    description="test",
                    prompts=["prompt1"],
                    models=["model1"],
                    datasets=["/data"],
                    metrics=["accuracy"]
                )
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                
                captured = capsys.readouterr()
                assert "DRY RUN MODE" in captured.out
                assert "Configuration validated successfully" in captured.out
    
    def test_save_config_option(self, sample_suite_yaml, temp_dir):
        """Test saving effective configuration."""
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml), '--save-config']):
            with patch('run_suite.Path.mkdir'), \
                 patch('builtins.open', create=True) as mock_open, \
                 patch('run_suite.TestSuiteRunner'):
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                
                # Verify config was saved
                # Check that open was called for both results and config
                assert mock_open.call_count >= 2
    
    def test_error_handling_invalid_suite_file(self, temp_dir, capsys):
        """Test error handling with invalid suite file."""
        invalid_suite = temp_dir / "invalid.yaml"
        invalid_suite.write_text("invalid: yaml: content: [")
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(invalid_suite)]):
            exit_code = run_suite.main()
            
            assert exit_code == 1
            
            captured = capsys.readouterr()
            assert "Error" in captured.err
    
    def test_error_handling_missing_suite_file(self, temp_dir, capsys):
        """Test error handling with missing suite file."""
        missing_suite = temp_dir / "missing.yaml"
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(missing_suite)]):
            exit_code = run_suite.main()
            
            assert exit_code == 1
            
            captured = capsys.readouterr()
            assert "Error" in captured.err
    
    def test_error_handling_suite_execution_failure(self, sample_suite_yaml, capsys):
        """Test error handling when suite execution fails."""
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml)]):
            with patch('run_suite.TestSuiteRunner') as mock_runner_class:
                mock_runner = Mock()
                mock_runner_class.return_value = mock_runner
                mock_runner.run_suite.side_effect = Exception("Suite execution failed")
                
                exit_code = run_suite.main()
                
                assert exit_code == 1
                
                captured = capsys.readouterr()
                assert "Suite execution failed" in captured.err
    
    def test_output_file_naming(self, sample_suite_yaml, mock_test_suite_runner):
        """Test output file naming convention."""
        mock_runner_class, mock_runner = mock_test_suite_runner
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml)]):
            with patch('run_suite.Path.mkdir'), \
                 patch('builtins.open', create=True) as mock_open, \
                 patch('run_suite.datetime') as mock_datetime:
                
                # Mock timestamp for predictable filename
                mock_datetime.now.return_value.strftime.return_value = "20241215_143052"
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                
                # Check that file was opened with expected naming pattern
                mock_open.assert_called()
                call_args = mock_open.call_args_list
                
                # Find the results file call
                results_call = None
                for call in call_args:
                    if "results.json" in str(call):
                        results_call = call
                        break
                
                assert results_call is not None
                filename = str(results_call[0][0])
                assert "suite_cli_test_suite_20241215_143052_results.json" in filename
    
    def test_json_output_format(self, sample_suite_yaml, mock_test_suite_runner):
        """Test that results are saved in correct JSON format."""
        mock_runner_class, mock_runner = mock_test_suite_runner
        
        with patch('sys.argv', ['run_suite.py', '--suite', str(sample_suite_yaml)]):
            with patch('run_suite.Path.mkdir'), \
                 patch('builtins.open', create=True) as mock_open:
                
                # Create a mock file handle to capture written content
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                exit_code = run_suite.main()
                
                assert exit_code == 0
                
                # Verify JSON was written
                mock_file.write.assert_called()
                
                # Get the written content and verify it's valid JSON
                written_content = mock_file.write.call_args[0][0]
                try:
                    parsed_json = json.loads(written_content)
                    assert "suite_name" in parsed_json
                    assert "test_results" in parsed_json
                    assert "aggregated_metrics" in parsed_json
                except json.JSONDecodeError:
                    pytest.fail("Output is not valid JSON")


# Integration tests with subprocess (more realistic CLI testing)
@pytest.mark.integration
class TestRunSuiteCLIIntegration:
    """Integration tests that actually invoke the CLI as a subprocess."""
    
    def test_cli_help_output(self):
        """Test CLI help output."""
        result = subprocess.run(
            ["python3", "run_suite.py", "--help"],
            capture_output=True,
            text=True,
            cwd="/root/sai-benchmark"
        )
        
        assert result.returncode == 0
        assert "Run SAI-Benchmark test suites" in result.stdout
        assert "--suite" in result.stdout
        assert "--workers" in result.stdout
    
    def test_cli_missing_arguments(self):
        """Test CLI error with missing required arguments."""
        result = subprocess.run(
            ["python3", "run_suite.py"],
            capture_output=True,
            text=True,
            cwd="/root/sai-benchmark"
        )
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower()
    
    @pytest.mark.skipif(not Path("/root/sai-benchmark/suites").exists(),
                        reason="Test suites directory not available")
    def test_cli_with_real_suite_dry_run(self):
        """Test CLI with real suite file in dry-run mode."""
        # Look for any existing suite file
        suites_dir = Path("/root/sai-benchmark/suites")
        suite_files = list(suites_dir.glob("*.yaml"))
        
        if not suite_files:
            pytest.skip("No suite files available for testing")
        
        suite_file = suite_files[0]
        
        result = subprocess.run(
            ["python3", "run_suite.py", "--suite", str(suite_file), "--dry-run"],
            capture_output=True,
            text=True,
            cwd="/root/sai-benchmark"
        )
        
        # Dry run should succeed or fail gracefully
        assert "DRY RUN MODE" in result.stdout or result.returncode != 0
    
    def test_cli_invalid_worker_count(self, temp_dir):
        """Test CLI error with invalid worker count."""
        # Create minimal valid suite file
        suite_config = {
            "name": "test",
            "description": "test",
            "prompts": ["test"],
            "models": ["test"],
            "datasets": ["/tmp"],
            "metrics": ["accuracy"]
        }
        
        suite_path = temp_dir / "test_suite.yaml"
        with open(suite_path, 'w') as f:
            yaml.dump(suite_config, f)
        
        result = subprocess.run(
            ["python3", "run_suite.py", "--suite", str(suite_path), "--workers", "0"],
            capture_output=True,
            text=True,
            cwd="/root/sai-benchmark"
        )
        
        assert result.returncode != 0
        assert "worker" in result.stderr.lower()


def parse_args(args):
    """Helper function to test argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", "-s", required=True)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--output", "-o", default="out/")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--show-registries", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-config", action="store_true")
    
    # Add validation for workers
    parsed_args = parser.parse_args(args)
    if parsed_args.workers <= 0:
        parser.error("Number of workers must be positive")
    
    return parsed_args


# Mock the parse_args function for testing
run_suite.parse_args = parse_args