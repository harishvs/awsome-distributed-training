# Solution Log Book

Problems encountered and solutions during the CentOS 9 EFA AMI + ParallelCluster build process.

---

## 1. NCCL Tests — `mpi.h: No such file or directory`

**Phase**: Packer AMI build → `nccl_tests_src` role
**Error**: `common.h:17:10: fatal error: mpi.h: No such file or directory`
**Root cause**: CentOS 9's `openmpi-devel` installs headers to `/usr/include/openmpi-x86_64/` instead of `/usr/lib64/openmpi/include/`.
**Fix**: Added `export CPATH=/usr/include/openmpi-x86_64:$CPATH` to the NCCL tests build step.

---

## 2. ParallelCluster — `Operating System 'centos.9' is not supported`

**Phase**: `pcluster build-image` → OS detection
**Error**: `Operating System 'centos.9' is not supported. Failing build.`
**Root cause**: ParallelCluster reads `ID` from `/etc/os-release`. CentOS Stream 9 reports `ID="centos"` which isn't in the supported list (rhel, rocky, amzn, ubuntu).
**Fix**: Created `pcluster_compat` Ansible role that patches `/etc/os-release` to set `ID="rhel"`. Original backed up to `/etc/os-release.centos9.bak`.

---

## 3. ParallelCluster — `No matching repo: codeready-builder-for-rhel-9-rhui-rpms`

**Phase**: `pcluster build-image` → Chef cookbook `package_repos`
**Error**: `Error: No matching repo to modify: codeready-builder-for-rhel-9-rhui-rpms.`
**Root cause**: ParallelCluster's Chef cookbook expects RHEL 9 RHUI repos. CentOS Stream 9 uses `crb` (CodeReady Builder) with different repo names.
**Fix**: Added a RHUI repo alias in `pcluster_compat` role at `/etc/yum.repos.d/pcluster-rhui-compat.repo` that maps `codeready-builder-for-rhel-9-rhui-rpms` to CentOS CRB mirrors.

---

## 4. ParallelCluster — `kmod-lustre-client` kernel symbol mismatch

**Phase**: `pcluster build-image` → Chef cookbook `lustre[Install FSx options]`
**Error**: `nothing provides kernel(generic_error_remove_page) needed by kmod-lustre-client-2.15.6-1.fsx26.el9.x86_64`
**Root cause**: The pre-built `kmod-lustre-client` RPMs from the `aws-fsx` repo are compiled against RHEL 9 kernels. CentOS Stream 9 kernel (`5.14.0-677.el9`) has different symbol versions.
**Fix**: Created `lustre_client` Ansible role that installs a dummy `kmod-lustre-client` RPM matching the expected version so ParallelCluster's Chef cookbook sees it as already satisfied.

---

## 5. Lustre role — Source RPM not available

**Phase**: Packer AMI build → `lustre_client` role
**Error**: `ls: cannot access 'kmod-lustre-client-*.src.rpm': No such file or directory`
**Root cause**: The `aws-fsx` yum repo does not publish source RPMs, so `yumdownloader --source` found nothing.
**Fix**: Switched from source RPM rebuild approach to dummy RPM approach using `printf` + `rpmbuild`.

---

## 6. Lustre role — YAML parsing error with heredoc

**Phase**: Packer AMI build → `lustre_client` role
**Error**: `YAML parsing failed: Colons in unquoted values must be followed by a non-space character.`
**Root cause**: The shell heredoc in the Ansible task contained `Key: Value` lines (e.g., `Group: System Environment/Kernel`) that Ansible's YAML parser interpreted as YAML mappings.
**Fix**: Replaced `cat << SPEC ... SPEC` heredoc with `printf` to generate the RPM spec file, avoiding colons in the YAML block.

---

## 7. ParallelCluster — `Module lnet not found`

**Phase**: `pcluster build-image` → Chef cookbook `lustre[Install FSx options]`
**Error**: `modprobe: FATAL: Module lnet not found in directory /lib/modules/5.14.0-681.el9.x86_64`
**Root cause**: The dummy `kmod-lustre-client` RPM satisfied the package dependency, but ParallelCluster's cookbook also runs `modprobe lnet` to load the Lustre networking kernel module. Since no actual kernel module was built for the CentOS kernel, modprobe fails.
**Fix**: Added stub kernel modules (`lnet`, `ksocklnd`, `ko2iblnd`, `lustre`) compiled as minimal `.ko` files against the running kernel, then ran `depmod -a` so modprobe can find them.

---

## 8. ParallelCluster — `Module lnet not found` (kernel version mismatch)

**Phase**: `pcluster build-image` → Chef cookbook `lustre[Install FSx options]`
**Error**: `modprobe: FATAL: Module lnet not found in directory /lib/modules/5.14.0-681.el9.x86_64`
**Root cause**: Stub modules were built only for the running kernel during Packer build (`5.14.0-677`), but the ParallelCluster Image Builder instance booted with a newer kernel (`5.14.0-681`) that was installed but not active during the Packer build.
**Fix**: Changed stub module build to iterate over ALL installed kernels in `/lib/modules/` instead of just `$(uname -r)`. Also install `kernel-devel` (latest) in addition to `kernel-devel-$(uname -r)`.

---

## 9. ParallelCluster — `Module lnet not found` (missing kernel-devel)

**Phase**: `pcluster build-image` → Chef cookbook `lustre[Install FSx options]`
**Error**: `modprobe: FATAL: Module lnet not found in directory /lib/modules/5.14.0-681.el9.x86_64`
**Root cause**: The NVIDIA DKMS driver install pulled in kernel `5.14.0-681` but not its `kernel-devel` package. The stub module loop skipped this kernel because `/lib/modules/5.14.0-681.el9.x86_64/build` didn't exist.
**Fix**: Added explicit `yum install -y kernel-devel-${KVER}` for every kernel found in `/lib/modules/` before building stubs. Also added fallback to `/usr/src/kernels/${KVER}` as alternate build directory.

---

## 10. Lustre role — YAML parsing error with heredoc (again)

**Phase**: Packer AMI build → `lustre_client` role
**Error**: `YAML parsing failed: While scanning a simple key could not find expected ':'.`
**Root cause**: Used `cat << 'EOF'` heredoc inside Ansible shell task to create C source and Makefile. Ansible's YAML parser chokes on heredoc markers.
**Fix**: Replaced all heredocs with `printf` statements, same pattern as issue #6.

---

## 11. Cluster creation — `ModuleNotFoundError: No module named 'pkg_resources'`

**Phase**: Cluster creation → head node bootstrap (`cfn-init` / `cfn-signal`)
**Error**: `ModuleNotFoundError: No module named 'pkg_resources'` in ParallelCluster's `cfn_bootstrap_virtualenv`
**Root cause**: `setuptools` 82.0.0 was installed in ParallelCluster's Python virtualenv, but setuptools >= 82 removed `pkg_resources` as a separate module. ParallelCluster's `cfn-bootstrap` depends on `pkg_resources`.
**Fix**: Added task in `pcluster_compat` role to pin `setuptools<82` in all ParallelCluster virtualenvs during the Packer build.

---

## 12. Cluster creation — `pkg_resources` fix not persisting

**Phase**: Cluster creation → head node bootstrap
**Error**: Same as #11 — `ModuleNotFoundError: No module named 'pkg_resources'`
**Root cause**: The `setuptools<82` pin in the `pcluster_compat` role runs during the Packer build, but `pcluster build-image` creates/overwrites the ParallelCluster virtualenvs afterward with setuptools 82.0.0. The fix gets lost.
**Fix**: Created a systemd oneshot service (`fix-setuptools.service`) that runs `Before=cloud-init.service` at boot time. It pins `setuptools<82` in all ParallelCluster virtualenvs before `cfn-init` runs, then disables itself.

---

## 13. Cluster creation — systemd fix-setuptools service didn't run before cloud-init

**Phase**: Cluster creation → head node bootstrap
**Error**: Same WaitCondition timeout — `pkg_resources` still missing
**Root cause**: The systemd `Before=cloud-init.service` ordering didn't guarantee the fix ran before cloud-init's user scripts executed `cfn-init`. Systemd ordering is complex and cloud-init has its own internal ordering.
**Fix**: Switched from systemd service to cloud-init `bootcmd` via `/etc/cloud/cloud.cfg.d/00-fix-setuptools.cfg`. `bootcmd` runs very early in cloud-init, before any user scripts or `runcmd`.

---

## 14. Cluster creation — bootcmd glob not expanding + wrong setuptools version

**Phase**: Cluster creation → head node bootstrap
**Error**: Same `pkg_resources` missing — bootcmd ran but setuptools stayed at 82.0.0
**Root cause**: Two issues: (1) inline glob `*` in cloud-init YAML bootcmd wasn't expanding in the shell, (2) `setuptools==81.1.0` doesn't exist as a published version.
**Fix**: Changed bootcmd to call an external script `/usr/local/sbin/fix-setuptools.sh` which properly expands globs and uses `setuptools<82` range instead of exact version.
