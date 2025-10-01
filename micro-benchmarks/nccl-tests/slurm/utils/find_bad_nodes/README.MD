# NCCL Bad Node Finder

A Python tool for identifying problematic nodes in a Slurm cluster using NCCL (NVIDIA Collective Communication Library) all-reduce tests.

## Overview

The `find_bad_nodes_nccl.py` script uses a binary search approach to efficiently identify nodes that cause NCCL communication failures in multi-GPU distributed training environments. This is particularly useful for maintaining cluster health in HPC and ML training environments.

## What it does

### Core Functionality

1. **Node Discovery**: Automatically discovers available nodes using Slurm's `sinfo` command
2. **NCCL Testing**: Runs NCCL all-reduce performance tests across different node combinations
3. **Binary Search**: Uses an efficient binary search algorithm to isolate problematic nodes
4. **Comprehensive Logging**: Creates detailed logs of all test runs and results
5. **Error Analysis**: Analyzes NCCL output for specific error patterns and warnings

### How it works

1. **Initial Test**: Tests all available nodes together to see if there are any issues
2. **Binary Search**: If issues are found, splits the node set in half and tests each half
3. **Isolation**: Continues dividing until individual problematic nodes are identified
4. **Reporting**: Provides actionable recommendations for draining bad nodes

### Key Features

- **Configurable Parameters**: Test duration, node limits, and NCCL binary path
- **Environment Variable Support**: Can be configured via environment variables
- **Timestamped Logging**: All logs include timestamps for tracking
- **Bandwidth Analysis**: Extracts and reports NCCL bandwidth performance
- **Error Pattern Detection**: Identifies common NCCL failure patterns

## Usage

### Basic Usage

```bash
python3 find_bad_nodes_nccl.py
```

### With Custom Parameters

```bash
python3 find_bad_nodes_nccl.py --max-nodes 16 --test-duration 60 --nccl-test-path /custom/path/to/all_reduce_perf
```

### Using Environment Variables

```bash
export MAX_NODES=16
export TEST_DURATION=60
export NCCL_TEST_PATH=/custom/path/to/all_reduce_perf
python3 find_bad_nodes_nccl.py
```

### Command Line Options

- `--min-nodes`: Minimum number of nodes to test (default: 2)
- `--max-nodes`: Maximum number of nodes to test (default: 8)
- `--test-duration`: Test duration in seconds (default: 30)
- `--nccl-test-path`: Path to NCCL test binary (default: `/opt/nccl-tests/build/all_reduce_perf`)

## Output

The tool creates several output files in the `find_bad_nodes_logs/` directory:

- `analysis_summary_TIMESTAMP.txt`: High-level summary of the analysis
- `node_combinations_TIMESTAMP.txt`: Detailed results for each node combination tested
- `nccl_test_*nodes_TIMESTAMP.log`: Raw NCCL test output for each test run

## Prerequisites

- Python 3.6+
- Slurm workload manager
- MPI (OpenMPI or similar)
- NCCL tests binary
- Access to GPU nodes in the cluster

## Test Suite

### What `test_find_bad_nodes_nccl.py` Tests

The test suite provides comprehensive unit testing for the NCCL bad node finder:

#### Test Categories

1. **Initialization Tests**
   - Verifies proper class initialization with default and custom parameters
   - Tests log directory creation and file setup

2. **Node Discovery Tests**
   - Mocks Slurm `sinfo` command responses
   - Tests handling of various node availability scenarios
   - Verifies error handling when Slurm is unavailable

3. **NCCL Test Execution Tests**
   - Mocks MPI command execution
   - Tests successful and failed NCCL test scenarios
   - Verifies hostfile creation and cleanup
   - Tests bandwidth extraction from NCCL output

4. **Binary Search Algorithm Tests**
   - Tests the core binary search logic for isolating bad nodes
   - Verifies handling of different failure scenarios:
     - All nodes working
     - Single bad node
     - Multiple bad nodes
     - Scaling issues (nodes work individually but not together)

5. **Error Analysis Tests**
   - Tests detection of NCCL warnings
   - Tests timeout detection
   - Verifies error pattern recognition

6. **Logging Tests**
   - Verifies proper logging to summary and combinations files
   - Tests log file creation and content

7. **Integration Tests**
   - Tests the complete workflow from start to finish
   - Verifies proper exit codes and return values

#### Test Features

- **Comprehensive Mocking**: Uses `unittest.mock` to simulate external dependencies
- **Temporary Files**: Uses `tempfile` for safe test file operations
- **Parameterized Tests**: Tests multiple scenarios with different inputs
- **Error Simulation**: Tests various failure modes and edge cases

## Running the Tests

### Prerequisites for Testing

The tests require additional Python packages. It's recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install test dependencies
pip install -r test_requirements.txt
```

#### Test Dependencies

- `pytest>=7.0`: Modern testing framework
- `pytest-cov>=4.0`: Coverage reporting for pytest
- `pytest-mock>=3.0`: Enhanced mocking capabilities
- `coverage>=6.0`: Code coverage analysis

### Setting Up Test Environment

```bash
# Navigate to the find_bad_nodes directory
cd micro-benchmarks/nccl-tests/slurm/utils/find_bad_nodes

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install test dependencies
pip install -r test_requirements.txt
```

### Running All Tests

```bash
# Using pytest (recommended)
python3 -m pytest

# Or using unittest directly (no additional dependencies needed)
python3 -m unittest test_find_bad_nodes_nccl.py
```

### Running Specific Test Classes

```bash
# Test only initialization
python3 -m pytest test_find_bad_nodes_nccl.py::TestNCCLBadNodeFinderInit

# Test only NCCL execution
python3 -m pytest test_find_bad_nodes_nccl.py::TestNCCLTestExecution

# Test only binary search algorithm
python3 -m pytest test_find_bad_nodes_nccl.py::TestBinarySearchAlgorithm
```

### Running with Verbose Output

```bash
python3 -m pytest -v
```

### Running Tests with Coverage (Optional)

If you want to see test coverage (requires test dependencies):

```bash
# Run tests with coverage report
python3 -m pytest --cov=find_bad_nodes_nccl --cov-report=term-missing

# Generate HTML coverage report
python3 -m pytest --cov=find_bad_nodes_nccl --cov-report=html
```

### Test Configuration

The `pytest.ini` file configures:
- Test discovery patterns
- Verbose output
- Short traceback format

### Alternative: Running Without Virtual Environment

If you prefer not to use a virtual environment, you can run the basic tests using Python's built-in unittest:

```bash
python3 -m unittest test_find_bad_nodes_nccl.py
```

This approach uses only standard library modules but won't have the enhanced features of pytest.

## Example Workflow

1. **Run the tool**:
   ```bash
   python3 find_bad_nodes_nccl.py --max-nodes 8
   ```

2. **Check results**:
   ```bash
   cat find_bad_nodes_logs/analysis_summary_*.txt
   ```

3. **If bad nodes found, drain them**:
   ```bash
   scontrol update NodeName=bad-node-01,bad-node-02 State=drain Reason='NCCL test failed'
   ```

## Troubleshooting

### Common Issues

1. **Slurm not available**: Ensure you're running on a Slurm cluster with proper access
2. **NCCL binary not found**: Verify the NCCL test path is correct
3. **MPI issues**: Ensure MPI is properly configured for multi-node execution
4. **Permission issues**: Ensure you have access to create log files and run MPI jobs

### Debug Mode

For debugging, you can examine the detailed logs in the `find_bad_nodes_logs/` directory, particularly the raw NCCL output files.

## Contributing

When contributing to this tool:

1. Run the test suite to ensure no regressions
2. Add tests for new functionality
3. Update this README for any new features or changes
4. Follow Python best practices and type hints