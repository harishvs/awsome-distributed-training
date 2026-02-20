#!/bin/bash
# Fix setuptools in ParallelCluster virtualenvs
# setuptools >= 82 removed pkg_resources which cfn-bootstrap needs
for venv in /opt/parallelcluster/pyenv/versions/*/envs/*/bin/pip; do
  [ -x "${venv}" ] && ${venv} install 'setuptools<82' 2>/dev/null || true
done
