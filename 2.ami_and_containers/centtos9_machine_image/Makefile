# Makefile for CentOS 9 EFA AMI Build
# Provides targets for Packer validation and AMI building

PACKER_FILE := packer-centos9.pkr.hcl

.PHONY: init validate ami_centos9_gpu clean help

# Initialize Packer plugins
init:
	packer init $(PACKER_FILE)

# Validate Packer configuration
validate:
	packer validate $(PACKER_FILE)

# Build the CentOS 9 GPU AMI with EFA stack
ami_centos9_gpu: init validate
	packer build $(PACKER_FILE)

# Clean up build artifacts
clean:
	rm -rf packer_cache/
	rm -f *.log

# Display help information
help:
	@echo "CentOS 9 EFA AMI Build Targets:"
	@echo "  init            - Initialize Packer plugins"
	@echo "  validate        - Validate Packer configuration"
	@echo "  ami_centos9_gpu - Build the CentOS 9 GPU AMI (runs init and validate first)"
	@echo "  clean           - Remove build artifacts and cache"
	@echo "  help            - Display this help message"
