packer {
  required_plugins {
    amazon = {
      version = ">= 1.0.9"
      source  = "github.com/hashicorp/amazon"
    }
    ansible = {
      version = ">= 1.0.1"
      source  = "github.com/hashicorp/ansible"
    }
  }
}

variable "ami_name" {
  type        = string
  default     = "centos9-efa-ml"
  description = "Name prefix for the AMI"
}

variable "ami_version" {
  type        = string
  default     = "1.0.0"
  description = "Version string for the AMI"
}

variable "base_ami" {
  type        = string
  default     = "ami-04331ec57720ee626"
  description = "CentOS Stream 9 base AMI ID"
}

variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for AMI creation"
}

variable "instance_type" {
  type        = string
  default     = "g5.16xlarge"
  description = "Instance type for build (must have GPU)"
}

variable "ssh_username" {
  type        = string
  default     = "ec2-user"
  description = "SSH username for CentOS 9"
}

variable "volume_size" {
  type        = number
  default     = 100
  description = "Root volume size in GB"
}

variable "inventory_directory" {
  type        = string
  default     = "inventory"
  description = "Ansible inventory directory"
}

variable "playbook_file" {
  type        = string
  default     = "playbook-centos9-gpu.yml"
  description = "Ansible playbook file"
}

# Timestamp for unique AMI naming
locals {
  timestamp = regex_replace(timestamp(), "[- TZ:]", "")
}

source "amazon-ebs" "centos9-efa" {
  ami_name      = "${var.ami_name}-${var.ami_version}-${local.timestamp}"
  instance_type = var.instance_type
  region        = var.aws_region
  source_ami    = var.base_ami
  ssh_username  = var.ssh_username

  # Cloud-init to configure passwordless sudo
  user_data_file = "cloud-init-sudo.yaml"

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = var.volume_size
    volume_type           = "gp3"
    throughput            = 1000
    iops                  = 10000
    delete_on_termination = true
  }

  tags = {
    "Name"        = "${var.ami_name}-${var.ami_version}"
    "OS"          = "CentOS 9"
    "BuildDate"   = local.timestamp
    "Description" = "CentOS 9 AMI with EFA stack built from source for ML workloads"
  }

  run_tags = {
    "Name" = "packer-builder-centos9-efa"
  }
}

build {
  name    = "centos9-efa-gpu"
  sources = ["source.amazon-ebs.centos9-efa"]

  # Run Ansible playbook
  provisioner "ansible" {
    user                = var.ssh_username
    use_proxy           = false
    ansible_env_vars    = ["ANSIBLE_SCP_EXTRA_ARGS='-O'", "ANSIBLE_HOST_KEY_CHECKING=False"]
    playbook_file       = var.playbook_file
    groups              = ["default"]
    extra_arguments     = [
      "--become",
      "--become-method=sudo", 
      "--become-user=root",
      "-e", "@inventory/group_vars/all.yml"
    ]
  }
}
