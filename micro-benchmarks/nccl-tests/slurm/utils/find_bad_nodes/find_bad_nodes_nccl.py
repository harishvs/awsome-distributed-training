#!/usr/bin/env python3

"""
NCCL-based Bad Node Finder
This script uses NCCL all-reduce tests to identify problematic nodes
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
import argparse


class NCCLBadNodeFinder:
    def __init__(self, min_nodes: int = 2, max_nodes: int = 8, 
                 test_duration: int = 30, nccl_test_path: str = "/opt/nccl-tests/build/all_reduce_perf"):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.test_duration = test_duration
        self.nccl_test_path = nccl_test_path
        
        # Setup logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logdir = Path("find_bad_nodes_logs")
        self.logdir.mkdir(exist_ok=True)
        
        self.summary_file = self.logdir / f"analysis_summary_{self.timestamp}.txt"
        self.combinations_file = self.logdir / f"node_combinations_{self.timestamp}.txt"
        
        # Initialize log files
        self._log_to_summary(f"NCCL Bad Node Finder - Started at {datetime.now()}")

    def _log_to_summary(self, message: str) -> None:
        """Log message to summary file and print to stdout"""
        print(message)
        with open(self.summary_file, 'a') as f:
            f.write(f"{message}\n")

    def _log_to_combinations(self, message: str) -> None:
        """Log message to combinations file and print to stdout"""
        print(message)
        with open(self.combinations_file, 'a') as f:
            f.write(f"{message}\n")

    def get_available_nodes(self) -> List[str]:
        """Get available nodes from Slurm"""
        try:
            result = subprocess.run(
                ["sinfo", "-N", "-h", "-t", "idle,alloc", "-o", "%N"],
                capture_output=True, text=True, check=True
            )
            nodes = sorted(set(result.stdout.strip().split('\n')))
            nodes = [node for node in nodes if node.strip()]  # Remove empty strings
            return nodes
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Slurm not available. Please set AVAILABLE_NODES manually.")
            sys.exit(1)

    def run_nccl_test(self, nodes: List[str]) -> bool:
        """Run NCCL test on specific nodes"""
        node_list = ','.join(nodes)
        num_nodes = len(nodes)
        gpus_per_node = 8  # Adjust based on your instance type
        
        self._log_to_combinations(f"Testing nodes: {node_list}")
        
        # Create temporary hostfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.hostfile') as hostfile:
            for node in nodes:
                for i in range(gpus_per_node):
                    hostfile.write(f"{node} slots=1\n")
            hostfile_path = hostfile.name
        
        try:
            # Prepare output file
            output_file = self.logdir / f"nccl_test_{num_nodes}nodes_{self.timestamp}.log"
            
            # Build mpirun command
            cmd = [
                "timeout", str(self.test_duration),
                "mpirun",
                "--hostfile", hostfile_path,
                "--map-by", f"ppr:{gpus_per_node}:node",
                "--bind-to", "none",
                "-x", "NCCL_DEBUG=INFO",
                "-x", "NCCL_TREE_THRESHOLD=0",
                "-x", "NCCL_IB_DISABLE=1",
                "-x", "NCCL_SOCKET_IFNAME=^docker0,lo",
                self.nccl_test_path,
                "-b", "1G", "-e", "1G", "-f", "2", "-g", "1"
            ]
            
            # Run NCCL test
            with open(output_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            
            exit_code = result.returncode
            
            # Analyze results
            if exit_code == 0:
                bandwidth = self._extract_bandwidth(output_file)
                self._log_to_combinations(f"  ✅ SUCCESS - Bandwidth: {bandwidth} GB/s")
                return True
            else:
                self._log_to_combinations(f"  ❌ FAILED - Exit code: {exit_code}")
                self._analyze_errors(output_file)
                return False
                
        finally:
            # Clean up hostfile
            try:
                os.unlink(hostfile_path)
            except OSError:
                pass

    def _extract_bandwidth(self, output_file: Path) -> str:
        """Extract bandwidth from NCCL test output"""
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                if "Avg bus bandwidth" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        return parts[5]
            return "N/A"
        except Exception:
            return "N/A"

    def _analyze_errors(self, output_file: Path) -> None:
        """Analyze error patterns in output file"""
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            if "NCCL WARN" in content:
                self._log_to_combinations("    NCCL warnings detected")
            if "timeout" in content.lower():
                self._log_to_combinations("    Timeout detected")
        except Exception:
            pass

    def find_bad_nodes(self, all_nodes: List[str]) -> List[str]:
        """Find bad nodes using binary search approach"""
        bad_nodes = []
        
        self._log_to_summary("")
        self._log_to_summary("=== BINARY SEARCH FOR BAD NODES ===")
        
        # Test all nodes together first
        self._log_to_summary(f"Testing all {len(all_nodes)} nodes together...")
        if self.run_nccl_test(all_nodes):
            self._log_to_summary("✅ All nodes working together")
            return []
        
        self._log_to_summary("❌ Issues detected with full node set. Starting binary search...")
        
        # Binary search to isolate bad nodes
        nodes_to_test = all_nodes.copy()
        
        while len(nodes_to_test) > 1:
            mid = len(nodes_to_test) // 2
            first_half = nodes_to_test[:mid]
            second_half = nodes_to_test[mid:]
            
            self._log_to_summary(f"Testing first half: {first_half}")
            if not self.run_nccl_test(first_half):
                nodes_to_test = first_half
                self._log_to_summary("  Issue in first half")
            else:
                self._log_to_summary(f"Testing second half: {second_half}")
                if not self.run_nccl_test(second_half):
                    nodes_to_test = second_half
                    self._log_to_summary("  Issue in second half")
                else:
                    self._log_to_summary("  Both halves work individually - may be a scaling issue")
                    break
        
        # Test individual nodes if we narrowed it down
        if len(nodes_to_test) == 1:
            self._log_to_summary(f"Testing individual node: {nodes_to_test[0]}")
            if not self.run_nccl_test(nodes_to_test):
                bad_nodes.extend(nodes_to_test)
        
        # Report bad nodes
        if bad_nodes:
            self._log_to_summary("")
            self._log_to_summary(f"❌ BAD NODES IDENTIFIED: {bad_nodes}")
            bad_nodes_str = ','.join(bad_nodes)
            print(f"Recommended action: scontrol update NodeName={bad_nodes_str} State=drain Reason='NCCL test failed'")
        else:
            self._log_to_summary("✅ No individual bad nodes found - may be a configuration or scaling issue")
        
        return bad_nodes

    def run(self) -> int:
        """Main execution method"""
        self._log_to_summary("Starting NCCL-based bad node detection...")
        
        # Get available nodes
        available_nodes = self.get_available_nodes()
        self._log_to_summary(f"Found {len(available_nodes)} available nodes: {available_nodes}")
        
        # Limit to reasonable number of nodes for testing
        if len(available_nodes) > self.max_nodes:
            test_nodes = available_nodes[:self.max_nodes]
            self._log_to_summary(f"Limiting test to first {self.max_nodes} nodes: {test_nodes}")
        else:
            test_nodes = available_nodes
        
        # Find bad nodes
        bad_nodes = self.find_bad_nodes(test_nodes)
        
        self._log_to_summary("")
        self._log_to_summary(f"Analysis complete at {datetime.now()}")
        self._log_to_summary(f"Logs saved in: {self.logdir}")
        self._log_to_summary(f"Summary: {self.summary_file}")
        
        return 1 if bad_nodes else 0


def main():
    parser = argparse.ArgumentParser(description="NCCL-based Bad Node Finder")
    parser.add_argument("--min-nodes", type=int, default=2, help="Minimum nodes to test")
    parser.add_argument("--max-nodes", type=int, default=8, help="Maximum nodes to test")
    parser.add_argument("--test-duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--nccl-test-path", type=str, default="/opt/nccl-tests/build/all_reduce_perf",
                       help="Path to NCCL test binary")
    
    args = parser.parse_args()
    
    # Allow environment variable overrides
    min_nodes = int(os.getenv("MIN_NODES", args.min_nodes))
    max_nodes = int(os.getenv("MAX_NODES", args.max_nodes))
    test_duration = int(os.getenv("TEST_DURATION", args.test_duration))
    nccl_test_path = os.getenv("NCCL_TEST_PATH", args.nccl_test_path)
    
    finder = NCCLBadNodeFinder(min_nodes, max_nodes, test_duration, nccl_test_path)
    return finder.run()


if __name__ == "__main__":
    sys.exit(main())