"""
Property-based tests for CentOS 9 EFA AMI configuration.

These tests verify that the Ansible/Packer configuration files contain
the expected patterns for environment configuration.

**Validates: Requirements 13.2, 13.3, 13.4**
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pytest
from hypothesis import given, strategies as st, settings


# Get the base path for the centtos9_machine_image directory
BASE_PATH = Path(__file__).parent.parent


@dataclass
class ComponentPaths:
    """Represents the expected paths for a component in the EFA stack."""
    name: str
    library_path: str
    binary_path: Optional[str] = None
    header_path: Optional[str] = None


# Define all components with their expected paths based on the design document
# and environment_config role defaults
COMPONENTS = [
    ComponentPaths(
        name="CUDA",
        library_path="env_cuda_lib",  # {{ env_cuda_home }}/lib64
        binary_path="CUDA_HOME",  # $CUDA_HOME/bin
        header_path="env_cuda_include",  # {{ env_cuda_home }}/targets/x86_64-linux/include
    ),
    ComponentPaths(
        name="GDRCopy",
        library_path="env_gdrcopy_lib",  # {{ env_gdrcopy_prefix }}/lib64
        binary_path="env_gdrcopy_bin",  # {{ env_gdrcopy_prefix }}/bin
        header_path="env_gdrcopy_include",  # {{ env_gdrcopy_prefix }}/include
    ),
    ComponentPaths(
        name="Hwloc",
        library_path="env_hwloc_lib",  # {{ env_hwloc_prefix }}/lib
        binary_path="env_hwloc_bin",  # {{ env_hwloc_prefix }}/bin
        header_path=None,  # Hwloc doesn't configure CPATH in the template
    ),
    ComponentPaths(
        name="EFA/Libfabric",
        library_path="env_efa_lib",  # {{ env_efa_prefix }}/lib
        binary_path="env_efa_bin",  # {{ env_efa_prefix }}/bin
        header_path=None,  # EFA doesn't configure CPATH in the template
    ),
    ComponentPaths(
        name="NCCL",
        library_path="env_nccl_lib",  # {{ env_nccl_prefix }}/build/lib
        binary_path=None,  # NCCL doesn't have binaries in PATH
        header_path=None,  # NCCL doesn't configure CPATH in the template
    ),
    ComponentPaths(
        name="AWS OFI NCCL",
        library_path="env_aws_ofi_nccl_lib",  # {{ env_aws_ofi_nccl_prefix }}/lib
        binary_path=None,  # aws-ofi-nccl doesn't have binaries
        header_path=None,  # aws-ofi-nccl doesn't configure CPATH
    ),
    ComponentPaths(
        name="NCCL Tests",
        library_path=None,  # NCCL tests don't add to LD_LIBRARY_PATH
        binary_path="env_nccl_tests_bin",  # {{ env_nccl_tests_prefix }}/build
        header_path=None,  # NCCL tests don't configure CPATH
    ),
]


def get_template_content() -> str:
    """Read the environment configuration template file."""
    template_path = BASE_PATH / "roles" / "environment_config" / "templates" / "efa-stack.sh.j2"
    if not template_path.exists():
        pytest.skip(f"Template file not found: {template_path}")
    return template_path.read_text()


def get_defaults_content() -> str:
    """Read the environment_config defaults file."""
    defaults_path = BASE_PATH / "roles" / "environment_config" / "defaults" / "main.yml"
    if not defaults_path.exists():
        pytest.skip(f"Defaults file not found: {defaults_path}")
    return defaults_path.read_text()


class TestEnvironmentConfigurationCompleteness:
    """
    Property 1: Environment Configuration Completeness
    
    **Validates: Requirements 13.2, 13.3, 13.4**
    
    For any installed component (CUDA, GDRCopy, EFA, libfabric, NCCL, hwloc, 
    aws-ofi-nccl), the environment configuration in /etc/profile.d/ SHALL include 
    the component's library path in LD_LIBRARY_PATH, binary path in PATH 
    (if applicable), and header path in CPATH (if applicable).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load template and defaults content once per test class."""
        self.template_content = get_template_content()
        self.defaults_content = get_defaults_content()

    # Strategy to generate any component from our list
    component_strategy = st.sampled_from(COMPONENTS)

    @given(component=component_strategy)
    @settings(max_examples=len(COMPONENTS))
    def test_library_path_in_ld_library_path(self, component: ComponentPaths):
        """
        Property: For any component with a library path, that path SHALL be 
        included in LD_LIBRARY_PATH exports.
        
        **Validates: Requirements 13.2**
        """
        if component.library_path is None:
            # Component doesn't have a library path requirement
            return
        
        # Check that the library path variable is used in LD_LIBRARY_PATH export
        ld_library_path_pattern = re.compile(
            r'export\s+LD_LIBRARY_PATH=.*\{\{\s*' + re.escape(component.library_path) + r'\s*\}\}',
            re.MULTILINE
        )
        
        assert ld_library_path_pattern.search(self.template_content), (
            f"Component '{component.name}' library path variable '{component.library_path}' "
            f"is not included in LD_LIBRARY_PATH export in the template"
        )

    @given(component=component_strategy)
    @settings(max_examples=len(COMPONENTS))
    def test_binary_path_in_path(self, component: ComponentPaths):
        """
        Property: For any component with a binary path, that path SHALL be 
        included in PATH exports.
        
        **Validates: Requirements 13.3**
        """
        if component.binary_path is None:
            # Component doesn't have a binary path requirement
            return
        
        # Check that the binary path variable is used in PATH export
        # Handle both Jinja2 variable syntax and shell variable syntax
        if component.binary_path.startswith("env_"):
            # Jinja2 variable
            path_pattern = re.compile(
                r'export\s+PATH=.*\{\{\s*' + re.escape(component.binary_path) + r'\s*\}\}',
                re.MULTILINE
            )
        else:
            # Shell variable like $CUDA_HOME/bin
            path_pattern = re.compile(
                r'export\s+PATH=.*\$' + re.escape(component.binary_path) + r'/bin',
                re.MULTILINE
            )
        
        assert path_pattern.search(self.template_content), (
            f"Component '{component.name}' binary path variable '{component.binary_path}' "
            f"is not included in PATH export in the template"
        )

    @given(component=component_strategy)
    @settings(max_examples=len(COMPONENTS))
    def test_header_path_in_cpath(self, component: ComponentPaths):
        """
        Property: For any component with a header path, that path SHALL be 
        included in CPATH exports.
        
        **Validates: Requirements 13.4**
        """
        if component.header_path is None:
            # Component doesn't have a header path requirement
            return
        
        # Check that the header path variable is used in CPATH export
        cpath_pattern = re.compile(
            r'export\s+CPATH=.*\{\{\s*' + re.escape(component.header_path) + r'\s*\}\}',
            re.MULTILINE
        )
        
        assert cpath_pattern.search(self.template_content), (
            f"Component '{component.name}' header path variable '{component.header_path}' "
            f"is not included in CPATH export in the template"
        )

    def test_all_components_have_library_or_binary_path(self):
        """
        Verify that every component in the EFA stack has at least a library 
        or binary path configured.
        
        This is a sanity check to ensure our component definitions are complete.
        """
        for component in COMPONENTS:
            assert component.library_path is not None or component.binary_path is not None, (
                f"Component '{component.name}' has neither library_path nor binary_path defined"
            )

    def test_template_exports_ld_library_path(self):
        """Verify the template contains LD_LIBRARY_PATH exports."""
        assert "export LD_LIBRARY_PATH=" in self.template_content, (
            "Template does not contain any LD_LIBRARY_PATH exports"
        )

    def test_template_exports_path(self):
        """Verify the template contains PATH exports."""
        assert "export PATH=" in self.template_content, (
            "Template does not contain any PATH exports"
        )

    def test_template_exports_cpath(self):
        """Verify the template contains CPATH exports."""
        assert "export CPATH=" in self.template_content, (
            "Template does not contain any CPATH exports"
        )

    def test_defaults_define_all_path_variables(self):
        """
        Verify that all path variables used in the template are defined 
        in the defaults file.
        """
        # Extract all Jinja2 variables from template
        jinja_vars = re.findall(r'\{\{\s*(env_\w+)\s*\}\}', self.template_content)
        
        for var in jinja_vars:
            assert var in self.defaults_content, (
                f"Variable '{var}' used in template but not defined in defaults"
            )


def get_all_roles() -> list[str]:
    """
    Get all role directories in the roles/ directory.
    
    Excludes .gitkeep and any non-directory entries.
    """
    roles_path = BASE_PATH / "roles"
    if not roles_path.exists():
        pytest.skip(f"Roles directory not found: {roles_path}")
    
    roles = []
    for item in roles_path.iterdir():
        # Skip .gitkeep and any hidden files
        if item.name.startswith('.'):
            continue
        # Only include directories
        if item.is_dir():
            roles.append(item.name)
    
    return sorted(roles)


class TestRoleStructureConsistency:
    """
    Property 2: Role Structure Consistency
    
    **Validates: Requirements 14.2, 14.5**
    
    For any Ansible role in the roles/ directory, the role SHALL contain 
    at minimum a `tasks/main.yml` file and optionally a `defaults/main.yml` 
    file for configurable variables.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the roles path for tests."""
        self.roles_path = BASE_PATH / "roles"
        self.all_roles = get_all_roles()

    # Strategy to generate any role from the roles directory
    @property
    def role_strategy(self):
        """Create a strategy that samples from all available roles."""
        roles = get_all_roles()
        if not roles:
            pytest.skip("No roles found in roles/ directory")
        return st.sampled_from(roles)

    @given(role=st.data())
    @settings(max_examples=50)
    def test_role_has_tasks_main_yml(self, role):
        """
        Property: For any role in the roles/ directory, the role SHALL 
        contain a tasks/main.yml file.
        
        **Validates: Requirements 14.2**
        """
        roles = get_all_roles()
        if not roles:
            pytest.skip("No roles found in roles/ directory")
        
        role_name = role.draw(st.sampled_from(roles))
        role_path = self.roles_path / role_name
        tasks_main = role_path / "tasks" / "main.yml"
        
        assert tasks_main.exists(), (
            f"Role '{role_name}' is missing required tasks/main.yml file. "
            f"Expected path: {tasks_main}"
        )

    @given(role=st.data())
    @settings(max_examples=50)
    def test_role_defaults_consistency(self, role):
        """
        Property: For any role in the roles/ directory, IF a defaults/ 
        directory exists, THEN it SHALL contain a main.yml file.
        
        **Validates: Requirements 14.5**
        """
        roles = get_all_roles()
        if not roles:
            pytest.skip("No roles found in roles/ directory")
        
        role_name = role.draw(st.sampled_from(roles))
        role_path = self.roles_path / role_name
        defaults_dir = role_path / "defaults"
        defaults_main = defaults_dir / "main.yml"
        
        # If defaults/ directory exists, main.yml must exist
        if defaults_dir.exists() and defaults_dir.is_dir():
            assert defaults_main.exists(), (
                f"Role '{role_name}' has a defaults/ directory but is missing "
                f"defaults/main.yml file. Expected path: {defaults_main}"
            )

    def test_all_roles_enumerated(self):
        """
        Sanity check: Verify we can enumerate all roles in the directory.
        """
        roles = get_all_roles()
        assert len(roles) > 0, "No roles found in roles/ directory"
        
        # Verify expected roles exist based on design document
        expected_roles = [
            "base_centos9",
            "packages_centos9",
            "nvidia_driver_centos9",
            "nvidia_cuda_centos9",
            "nvidia_gdrcopy_centos9",
            "efa_driver_src",
            "rdma_core_src",
            "hwloc_src",
            "libfabric_src",
            "nccl_src",
            "jemalloc_src",
            "aws_ofi_nccl_src",
            "nccl_tests_src",
            "environment_config",
        ]
        
        for expected_role in expected_roles:
            assert expected_role in roles, (
                f"Expected role '{expected_role}' not found in roles/ directory"
            )

    def test_all_roles_have_tasks_main(self):
        """
        Exhaustive check: Verify ALL roles have tasks/main.yml.
        
        This complements the property test by ensuring complete coverage.
        """
        roles = get_all_roles()
        missing_tasks = []
        
        for role_name in roles:
            role_path = self.roles_path / role_name
            tasks_main = role_path / "tasks" / "main.yml"
            if not tasks_main.exists():
                missing_tasks.append(role_name)
        
        assert not missing_tasks, (
            f"The following roles are missing tasks/main.yml: {missing_tasks}"
        )

    def test_all_roles_defaults_consistency(self):
        """
        Exhaustive check: Verify ALL roles with defaults/ have defaults/main.yml.
        
        This complements the property test by ensuring complete coverage.
        """
        roles = get_all_roles()
        inconsistent_roles = []
        
        for role_name in roles:
            role_path = self.roles_path / role_name
            defaults_dir = role_path / "defaults"
            defaults_main = defaults_dir / "main.yml"
            
            if defaults_dir.exists() and defaults_dir.is_dir():
                if not defaults_main.exists():
                    inconsistent_roles.append(role_name)
        
        assert not inconsistent_roles, (
            f"The following roles have defaults/ directory but missing "
            f"defaults/main.yml: {inconsistent_roles}"
        )


# Source-build roles that must have all four build phases
SOURCE_BUILD_ROLES = [
    "efa_driver_src",
    "rdma_core_src",
    "hwloc_src",
    "libfabric_src",
    "nccl_src",
    "jemalloc_src",
    "aws_ofi_nccl_src",
    "nccl_tests_src",
]


@dataclass
class BuildPhasePatterns:
    """Patterns to detect each build phase in Ansible task files."""
    name: str
    patterns: list[str]
    description: str


# Define patterns for each build phase
BUILD_PHASES = [
    BuildPhasePatterns(
        name="source_acquisition",
        patterns=[
            r"ansible\.builtin\.git:",  # git clone
            r"ansible\.builtin\.get_url:",  # download tarball
            r"git:",  # short form git module
            r"get_url:",  # short form get_url module
            r"Clone.*repository",  # task name pattern
            r"Download.*source",  # task name pattern
            r"Download.*tarball",  # task name pattern
        ],
        description="Source acquisition (git clone, get_url, or similar)",
    ),
    BuildPhasePatterns(
        name="build_configuration",
        patterns=[
            r"\./configure",  # autotools configure
            r"cmake\s",  # cmake configuration
            r"autogen\.sh",  # autogen before configure
            r"Configure\s",  # task name pattern
            r"configure\s+--",  # configure with flags
            r"make\s+.*[A-Z_]+=",  # Makefile-based config (e.g., make MPI=1 CUDA_HOME=...)
            r"CUDA_HOME=",  # CUDA configuration variable
            r"MPI_HOME=",  # MPI configuration variable
            r"NCCL_HOME=",  # NCCL configuration variable
        ],
        description="Build configuration (configure, cmake, make variables, or similar)",
    ),
    BuildPhasePatterns(
        name="build_execution",
        patterns=[
            r"make\s+-j",  # make with parallel jobs
            r"make\s+.*build",  # make with build target (e.g., src.build)
            r"make\s*$",  # plain make
            r"ninja\s",  # ninja build
            r"Build\s",  # task name pattern
        ],
        description="Build execution (make, ninja, or similar)",
    ),
    BuildPhasePatterns(
        name="installation",
        patterns=[
            r"make\s+install",  # make install
            r"ninja\s+install",  # ninja install
            r"cp\s+-r",  # copy recursively
            r"cp\s+.*\s+/",  # copy to absolute path
            r"Install\s",  # task name pattern
            r"Copy.*to.*prefix",  # task name pattern
            r"depmod\s+-a",  # kernel module installation
        ],
        description="Installation (make install, cp, or similar)",
    ),
]


def get_source_build_roles() -> list[str]:
    """
    Get the list of source-build roles that exist in the roles/ directory.
    
    Returns only roles from SOURCE_BUILD_ROLES that actually exist.
    """
    roles_path = BASE_PATH / "roles"
    if not roles_path.exists():
        pytest.skip(f"Roles directory not found: {roles_path}")
    
    existing_roles = []
    for role_name in SOURCE_BUILD_ROLES:
        role_path = roles_path / role_name
        if role_path.exists() and role_path.is_dir():
            existing_roles.append(role_name)
    
    return existing_roles


def get_role_tasks_content(role_name: str) -> str:
    """Read the tasks/main.yml content for a given role."""
    tasks_path = BASE_PATH / "roles" / role_name / "tasks" / "main.yml"
    if not tasks_path.exists():
        pytest.skip(f"Tasks file not found: {tasks_path}")
    return tasks_path.read_text()


def check_build_phase_present(content: str, phase: BuildPhasePatterns) -> bool:
    """
    Check if a build phase is present in the task file content.
    
    Returns True if any of the phase's patterns match the content.
    """
    for pattern in phase.patterns:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return True
    return False


class TestSourceBuildConfigurationCompleteness:
    """
    Property 3: Source Build Configuration Completeness
    
    **Validates: Requirements 5.1-5.4, 6.1-6.4, 7.1-7.4, 8.1-8.10, 9.1-9.4, 10.1-10.4, 11.1-11.7, 12.1-12.4**
    
    For any component built from source (EFA driver, rdma-core, hwloc, libfabric, 
    NCCL, jemalloc, aws-ofi-nccl, NCCL tests), the Ansible role SHALL include 
    tasks for: (1) cloning/downloading source, (2) configuring build, 
    (3) building, and (4) installing.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the roles path for tests."""
        self.roles_path = BASE_PATH / "roles"
        self.source_build_roles = get_source_build_roles()

    @given(role=st.data())
    @settings(max_examples=len(SOURCE_BUILD_ROLES) * 2)
    def test_source_build_role_has_source_acquisition(self, role):
        """
        Property: For any source-build role, the tasks file SHALL contain 
        source acquisition tasks (git clone, get_url, or similar).
        
        **Validates: Requirements 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1**
        """
        roles = get_source_build_roles()
        if not roles:
            pytest.skip("No source-build roles found")
        
        role_name = role.draw(st.sampled_from(roles))
        content = get_role_tasks_content(role_name)
        phase = BUILD_PHASES[0]  # source_acquisition
        
        assert check_build_phase_present(content, phase), (
            f"Role '{role_name}' is missing {phase.description}. "
            f"Expected patterns: {phase.patterns}"
        )

    @given(role=st.data())
    @settings(max_examples=len(SOURCE_BUILD_ROLES) * 2)
    def test_source_build_role_has_build_configuration(self, role):
        """
        Property: For any source-build role, the tasks file SHALL contain 
        build configuration tasks (configure, cmake, or similar).
        
        **Validates: Requirements 5.2, 6.2, 7.2, 8.2-8.8, 9.2, 10.2, 11.2-11.5, 12.2**
        """
        roles = get_source_build_roles()
        if not roles:
            pytest.skip("No source-build roles found")
        
        role_name = role.draw(st.sampled_from(roles))
        content = get_role_tasks_content(role_name)
        phase = BUILD_PHASES[1]  # build_configuration
        
        assert check_build_phase_present(content, phase), (
            f"Role '{role_name}' is missing {phase.description}. "
            f"Expected patterns: {phase.patterns}"
        )

    @given(role=st.data())
    @settings(max_examples=len(SOURCE_BUILD_ROLES) * 2)
    def test_source_build_role_has_build_execution(self, role):
        """
        Property: For any source-build role, the tasks file SHALL contain 
        build execution tasks (make, ninja, or similar).
        
        **Validates: Requirements 5.2, 6.2, 7.2, 8.9, 9.2, 10.2, 11.6, 12.3**
        """
        roles = get_source_build_roles()
        if not roles:
            pytest.skip("No source-build roles found")
        
        role_name = role.draw(st.sampled_from(roles))
        content = get_role_tasks_content(role_name)
        phase = BUILD_PHASES[2]  # build_execution
        
        assert check_build_phase_present(content, phase), (
            f"Role '{role_name}' is missing {phase.description}. "
            f"Expected patterns: {phase.patterns}"
        )

    @given(role=st.data())
    @settings(max_examples=len(SOURCE_BUILD_ROLES) * 2)
    def test_source_build_role_has_installation(self, role):
        """
        Property: For any source-build role, the tasks file SHALL contain 
        installation tasks (make install, cp, or similar).
        
        **Validates: Requirements 5.3, 6.3, 7.3, 8.10, 9.3, 10.3, 11.6, 12.4**
        """
        roles = get_source_build_roles()
        if not roles:
            pytest.skip("No source-build roles found")
        
        role_name = role.draw(st.sampled_from(roles))
        content = get_role_tasks_content(role_name)
        phase = BUILD_PHASES[3]  # installation
        
        assert check_build_phase_present(content, phase), (
            f"Role '{role_name}' is missing {phase.description}. "
            f"Expected patterns: {phase.patterns}"
        )

    def test_all_source_build_roles_exist(self):
        """
        Sanity check: Verify all expected source-build roles exist.
        """
        existing_roles = get_source_build_roles()
        missing_roles = set(SOURCE_BUILD_ROLES) - set(existing_roles)
        
        assert not missing_roles, (
            f"Expected source-build roles not found: {missing_roles}"
        )

    def test_all_source_build_roles_have_all_phases(self):
        """
        Exhaustive check: Verify ALL source-build roles have all four build phases.
        
        This complements the property tests by ensuring complete coverage.
        """
        roles = get_source_build_roles()
        failures = []
        
        for role_name in roles:
            content = get_role_tasks_content(role_name)
            missing_phases = []
            
            for phase in BUILD_PHASES:
                if not check_build_phase_present(content, phase):
                    missing_phases.append(phase.name)
            
            if missing_phases:
                failures.append(f"{role_name}: missing {missing_phases}")
        
        assert not failures, (
            f"Source-build roles with missing build phases:\n" +
            "\n".join(failures)
        )

    def test_build_phase_patterns_are_valid_regex(self):
        """
        Sanity check: Verify all build phase patterns are valid regex.
        """
        for phase in BUILD_PHASES:
            for pattern in phase.patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    pytest.fail(f"Invalid regex pattern '{pattern}' in phase '{phase.name}': {e}")
