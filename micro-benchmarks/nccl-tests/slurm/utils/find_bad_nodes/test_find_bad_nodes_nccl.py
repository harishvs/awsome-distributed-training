#!/usr/bin/env python3

"""
Unit tests for NCCL Bad Node Finder
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock, call
import tempfile
import os
import sys
from pathlib import Path
import subprocess

# Add the current directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from find_bad_nodes_nccl import NCCLBadNodeFinder


class TestNCCLBadNodeFinder(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.finder = NCCLBadNodeFinder(
            min_nodes=2,
            max_nodes=4,
            test_duration=10,
            nccl_test_path="/test/nccl/path"
        )
        
    @patch('find_bad_nodes_nccl.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_init(self, mock_print, mock_file, mock_mkdir):
        """Test NCCLBadNodeFinder initialization"""
        finder = NCCLBadNodeFinder()
        
        # Check default values
        self.assertEqual(finder.min_nodes, 2)
        self.assertEqual(finder.max_nodes, 8)
        self.assertEqual(finder.test_duration, 30)
        self.assertEqual(finder.nccl_test_path, "/opt/nccl-tests/build/all_reduce_perf")
        
        # Check that log directory is created
        mock_mkdir.assert_called_once_with(exist_ok=True)
        
        # Check that initial log message is written
        mock_file.assert_called()
        mock_print.assert_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_log_to_summary(self, mock_print, mock_file):
        """Test logging to summary file"""
        message = "Test message"
        self.finder._log_to_summary(message)
        
        mock_print.assert_called_with(message)
        mock_file.assert_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_log_to_combinations(self, mock_print, mock_file):
        """Test logging to combinations file"""
        message = "Test combination"
        self.finder._log_to_combinations(message)
        
        mock_print.assert_called_with(message)
        mock_file.assert_called()

    @patch('subprocess.run')
    def test_get_available_nodes_success(self, mock_run):
        """Test successful retrieval of available nodes"""
        mock_result = Mock()
        mock_result.stdout = "node1\nnode2\nnode3\n"
        mock_run.return_value = mock_result
        
        nodes = self.finder.get_available_nodes()
        
        expected_nodes = ['node1', 'node2', 'node3']
        self.assertEqual(nodes, expected_nodes)
        
        mock_run.assert_called_once_with(
            ["sinfo", "-N", "-h", "-t", "idle,alloc", "-o", "%N"],
            capture_output=True, text=True, check=True
        )

    @patch('subprocess.run')
    def test_get_available_nodes_with_empty_lines(self, mock_run):
        """Test node retrieval with empty lines in output"""
        mock_result = Mock()
        mock_result.stdout = "node1\n\nnode2\n\nnode3\n"
        mock_run.return_value = mock_result
        
        nodes = self.finder.get_available_nodes()
        
        expected_nodes = ['node1', 'node2', 'node3']
        self.assertEqual(nodes, expected_nodes)

    @patch('subprocess.run')
    @patch('sys.exit')
    def test_get_available_nodes_slurm_not_available(self, mock_exit, mock_run):
        """Test handling when Slurm is not available"""
        mock_run.side_effect = FileNotFoundError()
        
        self.finder.get_available_nodes()
        
        mock_exit.assert_called_once_with(1)

    @patch('subprocess.run')
    @patch('sys.exit')
    def test_get_available_nodes_slurm_error(self, mock_exit, mock_run):
        """Test handling Slurm command errors"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'sinfo')
        
        self.finder.get_available_nodes()
        
        mock_exit.assert_called_once_with(1)

    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_run_nccl_test_success(self, mock_unlink, mock_file, mock_subprocess, mock_tempfile):
        """Test successful NCCL test execution"""
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = '/tmp/test_hostfile'
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp
        
        # Mock successful subprocess run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Mock bandwidth extraction
        with patch.object(self.finder, '_extract_bandwidth', return_value='50.5'):
            with patch.object(self.finder, '_log_to_combinations'):
                result = self.finder.run_nccl_test(['node1', 'node2'])
        
        self.assertTrue(result)
        mock_unlink.assert_called_once_with('/tmp/test_hostfile')

    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_run_nccl_test_failure(self, mock_unlink, mock_file, mock_subprocess, mock_tempfile):
        """Test failed NCCL test execution"""
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = '/tmp/test_hostfile'
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp
        
        # Mock failed subprocess run
        mock_result = Mock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result
        
        # Mock error analysis
        with patch.object(self.finder, '_analyze_errors'):
            with patch.object(self.finder, '_log_to_combinations'):
                result = self.finder.run_nccl_test(['node1'])
        
        self.assertFalse(result)
        mock_unlink.assert_called_once_with('/tmp/test_hostfile')

    def test_extract_bandwidth_success(self):
        """Test successful bandwidth extraction"""
        test_content = """
        Some output
        Avg bus bandwidth: 45.2 GB/s
        More output
        """
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            bandwidth = self.finder._extract_bandwidth(Path('/fake/path'))
        
        self.assertEqual(bandwidth, 'GB/s')  # 6th element (0-indexed: 5)

    def test_extract_bandwidth_not_found(self):
        """Test bandwidth extraction when pattern not found"""
        test_content = "No bandwidth info here"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            bandwidth = self.finder._extract_bandwidth(Path('/fake/path'))
        
        self.assertEqual(bandwidth, 'N/A')

    def test_extract_bandwidth_file_error(self):
        """Test bandwidth extraction with file read error"""
        with patch('builtins.open', side_effect=IOError()):
            bandwidth = self.finder._extract_bandwidth(Path('/fake/path'))
        
        self.assertEqual(bandwidth, 'N/A')

    def test_analyze_errors_nccl_warnings(self):
        """Test error analysis with NCCL warnings"""
        test_content = "Some output\nNCCL WARN: Something went wrong\nMore output"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            with patch.object(self.finder, '_log_to_combinations') as mock_log:
                self.finder._analyze_errors(Path('/fake/path'))
        
        mock_log.assert_called_with("    NCCL warnings detected")

    def test_analyze_errors_timeout(self):
        """Test error analysis with timeout"""
        test_content = "Some output\nTimeout occurred\nMore output"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            with patch.object(self.finder, '_log_to_combinations') as mock_log:
                self.finder._analyze_errors(Path('/fake/path'))
        
        mock_log.assert_called_with("    Timeout detected")

    def test_analyze_errors_file_error(self):
        """Test error analysis with file read error"""
        with patch('builtins.open', side_effect=IOError()):
            # Should not raise exception
            self.finder._analyze_errors(Path('/fake/path'))

    def test_find_bad_nodes_all_working(self):
        """Test find_bad_nodes when all nodes are working"""
        nodes = ['node1', 'node2', 'node3']
        
        with patch.object(self.finder, 'run_nccl_test', return_value=True):
            with patch.object(self.finder, '_log_to_summary'):
                bad_nodes = self.finder.find_bad_nodes(nodes)
        
        self.assertEqual(bad_nodes, [])

    def test_find_bad_nodes_binary_search_first_half(self):
        """Test binary search finding issue in first half"""
        nodes = ['node1', 'node2', 'node3', 'node4']
        
        # Mock test results: all fail, first half fails, individual node fails
        test_results = [False, False, False]  # all nodes, first half, individual
        
        with patch.object(self.finder, 'run_nccl_test', side_effect=test_results):
            with patch.object(self.finder, '_log_to_summary'):
                bad_nodes = self.finder.find_bad_nodes(nodes)
        
        self.assertEqual(bad_nodes, ['node1', 'node2'])

    def test_find_bad_nodes_binary_search_second_half(self):
        """Test binary search finding issue in second half"""
        nodes = ['node1', 'node2', 'node3', 'node4']
        
        # Mock test results: all fail, first half passes, second half fails, individual fails
        def mock_test_side_effect(test_nodes):
            if len(test_nodes) == 4:  # All nodes
                return False
            elif test_nodes == ['node1', 'node2']:  # First half
                return True
            elif test_nodes == ['node3', 'node4']:  # Second half
                return False
            elif test_nodes == ['node3']:  # Individual
                return False
            return True
        
        with patch.object(self.finder, 'run_nccl_test', side_effect=mock_test_side_effect):
            with patch.object(self.finder, '_log_to_summary'):
                bad_nodes = self.finder.find_bad_nodes(nodes)
        
        self.assertEqual(bad_nodes, ['node3'])

    def test_find_bad_nodes_scaling_issue(self):
        """Test detection of scaling issues (both halves work individually)"""
        nodes = ['node1', 'node2', 'node3', 'node4']
        
        def mock_test_side_effect(test_nodes):
            if len(test_nodes) == 4:  # All nodes fail
                return False
            else:  # Individual halves pass
                return True
        
        with patch.object(self.finder, 'run_nccl_test', side_effect=mock_test_side_effect):
            with patch.object(self.finder, '_log_to_summary'):
                bad_nodes = self.finder.find_bad_nodes(nodes)
        
        self.assertEqual(bad_nodes, [])

    @patch.object(NCCLBadNodeFinder, 'get_available_nodes')
    @patch.object(NCCLBadNodeFinder, 'find_bad_nodes')
    def test_run_with_node_limit(self, mock_find_bad, mock_get_nodes):
        """Test run method with node limiting"""
        mock_get_nodes.return_value = ['node1', 'node2', 'node3', 'node4', 'node5']
        mock_find_bad.return_value = []
        
        with patch.object(self.finder, '_log_to_summary'):
            result = self.finder.run()
        
        # Should limit to max_nodes (4)
        mock_find_bad.assert_called_once_with(['node1', 'node2', 'node3', 'node4'])
        self.assertEqual(result, 0)

    @patch.object(NCCLBadNodeFinder, 'get_available_nodes')
    @patch.object(NCCLBadNodeFinder, 'find_bad_nodes')
    def test_run_with_bad_nodes_found(self, mock_find_bad, mock_get_nodes):
        """Test run method when bad nodes are found"""
        mock_get_nodes.return_value = ['node1', 'node2']
        mock_find_bad.return_value = ['node1']
        
        with patch.object(self.finder, '_log_to_summary'):
            result = self.finder.run()
        
        self.assertEqual(result, 1)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.getenv')
    def test_main_with_args(self, mock_getenv, mock_parse_args):
        """Test main function with command line arguments"""
        # Mock command line arguments
        mock_args = Mock()
        mock_args.min_nodes = 3
        mock_args.max_nodes = 6
        mock_args.test_duration = 45
        mock_args.nccl_test_path = '/custom/path'
        mock_parse_args.return_value = mock_args
        
        # Mock environment variables (return defaults)
        mock_getenv.side_effect = lambda key, default: default
        
        with patch.object(NCCLBadNodeFinder, 'run', return_value=0) as mock_run:
            from find_bad_nodes_nccl import main
            result = main()
        
        self.assertEqual(result, 0)
        mock_run.assert_called_once()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.getenv')
    def test_main_with_env_vars(self, mock_getenv, mock_parse_args):
        """Test main function with environment variable overrides"""
        # Mock command line arguments
        mock_args = Mock()
        mock_args.min_nodes = 2
        mock_args.max_nodes = 8
        mock_args.test_duration = 30
        mock_args.nccl_test_path = '/default/path'
        mock_parse_args.return_value = mock_args
        
        # Mock environment variables
        env_vars = {
            'MIN_NODES': '4',
            'MAX_NODES': '12',
            'TEST_DURATION': '60',
            'NCCL_TEST_PATH': '/env/path'
        }
        mock_getenv.side_effect = lambda key, default: env_vars.get(key, default)
        
        with patch('find_bad_nodes_nccl.NCCLBadNodeFinder') as mock_finder_class:
            mock_finder = Mock()
            mock_finder.run.return_value = 0
            mock_finder_class.return_value = mock_finder
            
            from find_bad_nodes_nccl import main
            result = main()
        
        # Verify NCCLBadNodeFinder was created with env var values
        mock_finder_class.assert_called_once_with(4, 12, 60, '/env/path')
        self.assertEqual(result, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests that test multiple components together"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    @patch('builtins.open', new_callable=mock_open)
    def test_full_workflow_success(self, mock_file, mock_tempfile, mock_subprocess):
        """Test complete workflow with successful node detection"""
        # Mock sinfo command
        sinfo_result = Mock()
        sinfo_result.stdout = "node1\nnode2\n"
        
        # Mock NCCL test command
        nccl_result = Mock()
        nccl_result.returncode = 0
        
        mock_subprocess.side_effect = [sinfo_result, nccl_result]
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = '/tmp/test_hostfile'
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp
        
        # Mock bandwidth in file content
        mock_file.return_value.readlines.return_value = [
            "Some output\n",
            "Avg bus bandwidth: 45.2 GB/s\n"
        ]
        
        finder = NCCLBadNodeFinder(max_nodes=2, test_duration=5)
        
        with patch('os.unlink'):
            with patch.object(finder, '_log_to_summary'):
                with patch.object(finder, '_log_to_combinations'):
                    result = finder.run()
        
        self.assertEqual(result, 0)  # No bad nodes found


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)