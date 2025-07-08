"""
Poor Man's Ray
==============

This module provides a command-line interface (CLI) to start, stop, and manage EC2 instances,
providing a minimal alternative to Ray for distributed data processing. It is primarily designed
to work with version 2 of the Dolma toolkit.

The CLI offers several commands for managing EC2 instances and executing tasks:
- create: Launch new EC2 instances with specified configurations
- list: Display information about running instances
- terminate: Shut down and remove instances
- run: Execute commands on instances
- setup: Configure AWS credentials on instances
- setup-d2tk: Install and configure the Dolma2 toolkit
- map: Distribute scripts across multiple instances for parallel execution

Examples:
---------

Create a cluster of instances:
    poormanray create --name chipstest --number 5 --instance-type i4i.2xlarge --detach

List instances in a cluster:
    poormanray list --name chipstest --region us-east-1

Set up AWS credentials and D2TK pipeline on instances:
    poormanray setup-d2tk --name chipstest --ssh-key-path ~/.ssh/id_rsa

Set up decon pipeline on instances:
    poormanray setup-decon --name chipstest --ssh-key-path ~/.ssh/id_rsa --github-token xxx

Run a command on all instances in a cluster:
    poormanray run --name chipstest --command "echo 'Hello, world!'" --ssh-key-path ~/.ssh/id_rsa

Distribute and run multiple scripts across instances:
    poormanray map --name chipstest --script tmp/test_scripts/* --ssh-key-path ~/.ssh/id_rsa

Terminate all instances in a cluster:
    poormanray terminate --name chipstest --region us-east-1


Author: Luca Soldaini
Email: luca@soldaini.net
"""

import base64
import datetime
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import boto3
import click
import paramiko
from paramiko.channel import ChannelFile, ChannelStdinFile

if TYPE_CHECKING:
    from mypy_boto3_ec2.client import EC2Client
    from mypy_boto3_ec2.type_defs import InstanceTypeDef, InstanceStatusTypeDef
    from mypy_boto3_ssm.client import SSMClient

from cookbook.cli.utils import (
    get_aws_access_key_id,
    get_aws_secret_access_key,
    make_aws_config,
    make_aws_credentials,
)


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


PACKAGE_MANAGER_DETECTOR = """
#!/bin/bash

# Determine package manager based on OS information in /etc/os-release
determine_package_manager() {
  # Source the OS release file to get variables
  source /etc/os-release

  # First try using ID_LIKE if available
  if [[ -n "$ID_LIKE" ]]; then
    # Debian-based systems
    if [[ "$ID_LIKE" == *"debian"* ]]; then
      echo "apt"
      return
    # Red Hat / Fedora based systems
    elif [[ "$ID_LIKE" == *"fedora"* || "$ID_LIKE" == *"rhel"* ]]; then
      if command -v dnf &>/dev/null; then
        echo "dnf"
      else
        echo "yum"
      fi
      return
    # SUSE-based systems
    elif [[ "$ID_LIKE" == *"suse"* ]]; then
      echo "zypper"
      return
    fi
  fi

  # Fall back to ID if ID_LIKE didn't match or isn't available
  case "$ID" in
    debian|ubuntu|mint|pop|elementary|zorin|kali|parrot|deepin)
      echo "apt"
      ;;
    fedora)
      echo "dnf"
      ;;
    rhel|centos)
      if command -v dnf &>/dev/null; then
        echo "dnf"
      else
        echo "yum"
      fi
      ;;
    amzn)
      if [[ "$VERSION_ID" == "2023"* ]]; then
        echo "dnf"
      else
        echo "yum"
      fi
      ;;
    opensuse*|sles|suse)
      echo "zypper"
      ;;
    alpine)
      echo "apk"
      ;;
    arch|manjaro|endeavouros)
      echo "pacman"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

# Get and display the package manager
PKG_MANAGER=$(determine_package_manager)
"""


D2TK_SETUP = """
#!/bin/bash
# Set up raid drives
sudo yum install mdadm -y
NUM_DRIVES=$(echo "$(ls /dev/nvme*n1 | wc -l) - 1" | bc)
MDADM_CMD="sudo mdadm --create /dev/md0 --level=0 --raid-devices=$NUM_DRIVES"
for i in $(seq 1 $NUM_DRIVES); do
  MDADM_CMD="$MDADM_CMD /dev/nvme${i}n1"
done
eval $MDADM_CMD
sudo mkfs.xfs /dev/md0
sudo mkdir /mnt/raid0
sudo mount /dev/md0 /mnt/raid0
sudo chown -R $USER /mnt/raid0
# Download and set up all packages we need
sudo yum install gcc cmake openssl-devel gcc-c++ htop wget tmux screen -y
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin
sudo yum install git -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc
# Do datamap-rs setups
cd
git clone https://github.com/allenai/datamap-rs.git
cd datamap-rs
s5cmd run examples/all_dressed/s5cmd_asset_downloader.txt
cargo build --release
# Do minhash-rs setups
cd
git clone https://github.com/revbucket/minhash-rs.git
cd minhash-rs
cargo build --release
""".strip()


DOLMA_PYTHON_SETUP = f"""
#!/bin/bash
{PACKAGE_MANAGER_DETECTOR}

set -ex

# install python 3.12 with pip
sudo "${{PKG_MANAGER}}" update
sudo "${{PKG_MANAGER}}" install python3.12 python3.12-pip -y

# install git, tmux, htop
sudo "${{PKG_MANAGER}}" install git tmux htop -y

# install gcc, g++, cmake, openssl-devel
sudo "${{PKG_MANAGER}}" install gcc g++ cmake openssl-devel -y

# install github cli
curl -sS https://webi.sh/gh | sh

# install s5cmd
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin/

# install uv via pip
pip3.12 install uv

# make virtual environment, install dolma
uv python install 3.12
uv venv
uv pip install dolma
""".strip()


def make_decon_python_setup(github_token: str | None = None, host_index: int | None = None, host_count: int | None = None) -> str:
    """Generate the DECON Python setup script with optional GitHub token and PMR environment variables."""
    clone_cmd = "git clone https://github.com/allenai/decon.git"
    if github_token:
        clone_cmd = f"git clone https://{github_token}@github.com/allenai/decon.git"

    # Add PMR environment setup if host_index and host_count are provided
    pmr_setup = ""
    if host_index is not None and host_count is not None:
        pmr_setup = f"""
# Set up PMR environment variables
if ! grep -q "PMR_HOST_INDEX" /etc/environment; then
    echo "PMR_HOST_INDEX={host_index}" | sudo tee -a /etc/environment
    echo "PMR_HOST_COUNT={host_count}" | sudo tee -a /etc/environment
fi
export PMR_HOST_INDEX={host_index}
export PMR_HOST_COUNT={host_count}
"""

    return f"""
#!/bin/bash
{PACKAGE_MANAGER_DETECTOR}

# Set up drives
TOTAL_DRIVES=$(ls /dev/nvme*n1 | wc -l)

if [ $TOTAL_DRIVES -eq 1 ]; then
    echo "No instance store drives found, only root drive exists"
elif [ $TOTAL_DRIVES -eq 2 ]; then
    # Single instance store drive - format as ext4 and mount directly
    echo "Found single instance store drive, formatting as ext4 and mounting"
    sudo mkfs.ext4 /dev/nvme1n1
    sudo mkdir -p /mnt/decon-work
    sudo mount /dev/nvme1n1 /mnt/decon-work
    sudo chown -R $USER /mnt/decon-work
else
    # Multiple instance store drives - create RAID array first
    echo "Found multiple instance store drives, creating RAID array"
    sudo yum install mdadm -y
    NUM_DRIVES=$((TOTAL_DRIVES - 1))
    MDADM_CMD="sudo mdadm --create /dev/md0 --level=0 --raid-devices=$NUM_DRIVES"
    for i in $(seq 1 $NUM_DRIVES); do
        MDADM_CMD="$MDADM_CMD /dev/nvme${{i}}n1"
    done
    eval $MDADM_CMD
    sudo mkfs.ext4 /dev/md0
    sudo mkdir -p /mnt/decon-work
    sudo mount /dev/md0 /mnt/decon-work
    sudo chown -R $USER /mnt/decon-work
fi


set -ex
{pmr_setup}
# install python 3.12 with pip
sudo "${{PKG_MANAGER}}" update
sudo "${{PKG_MANAGER}}" install python3.12 python3.12-pip -y

# create symlink from /usr/bin/python to python3.12
sudo ln -sf /usr/bin/python3.12 /usr/bin/python
sudo ln -sf /usr/bin/pip3.12 /usr/bin/pip

# install git, tmux, htop
sudo "${{PKG_MANAGER}}" install git tmux htop -y

# install gcc, g++, cmake, openssl-devel
sudo "${{PKG_MANAGER}}" install gcc g++ cmake openssl-devel -y

# install github cli
curl -sS https://webi.sh/gh | sh

# install s5cmd
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin/

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc

cd
{clone_cmd}
cd decon
cargo build --release
pip install -r python/requirements.txt

make evals-s3
""".strip()

# Keep the default for backward compatibility
DECON_PYTHON_SETUP = make_decon_python_setup()

class InstanceStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting-down"
    TERMINATED = "terminated"
    STOPPING = "stopping"
    STOPPED = "stopped"

    @classmethod
    def active(cls) -> list["InstanceStatus"]:
        return [
            status
            for status in cls
            if status != cls.TERMINATED and status != cls.STOPPED and status != cls.SHUTTING_DOWN
        ]

    @classmethod
    def unterminated(cls) -> list["InstanceStatus"]:
        return [status for status in cls if status != cls.TERMINATED and status != cls.SHUTTING_DOWN]


DEFAULT_PRIVATE_KEY_NAMES = [
    "id_rsa",  # RSA (deprecated and not generated by default on many systems)
    "id_ecdsa",  # ECDSA
    "id_ecdsa_sk",  # ECDSA with FIDO/U2F
    "id_ed25519",  # Ed25519
    "id_ed25519_sk",  # Ed25519 with FIDO/U2F
]


@dataclass(frozen=True)
class SessionContent:
    stdout: str
    stderr: str

    @classmethod
    def from_channel(cls, channel: paramiko.Channel) -> "SessionContent":
        return cls(
            stdout=channel.recv(4096).decode("utf-8"),
            stderr=channel.recv(4096).decode("utf-8"),
        )

    @classmethod
    def from_command(
        cls,
        channel_stdin: ChannelStdinFile,
        channel_stdout: ChannelFile,
        channel_stderr: ChannelFile,
    ) -> "SessionContent":
        return cls(
            stdout=channel_stdout.read().decode("utf-8"),
            stderr=channel_stderr.read().decode("utf-8"),
        )

    def __str__(self) -> str:
        return f"stdout: {self.stdout.strip()}\nstderr: {self.stderr.strip()}"


@dataclass
class InstanceInfo:
    """
    Represents information about an EC2 instance.

    This class encapsulates all relevant details about an EC2 instance and provides
    methods for instance management operations like creation, description, and termination.

    Attributes:
        instance_id: The unique identifier for the EC2 instance
        instance_type: The type of the instance (e.g., t2.micro, m5.large)
        image_id: The AMI ID used to launch the instance
        state: Current state of the instance (e.g., running, stopped)
        public_ip_address: The public IP address assigned to the instance
        public_dns_name: The public DNS name assigned to the instance
        name: The Name tag value of the instance
        tags: Dictionary of all tags applied to the instance
        zone: The availability zone where the instance is running
        region: The AWS region where the instance is located
    """

    instance_id: str
    instance_type: str
    image_id: str
    state: InstanceStatus
    public_ip_address: str
    public_dns_name: str
    name: str
    tags: dict[str, str]
    zone: str
    created_at: datetime.datetime
    region: str = "us-east-1"

    def __post_init__(self):
        self._status: list[tuple[str, str]] = []

    def _update_status(self, name: str, status: str):
        self._status.append((name, status))

    @property
    def checks(self) -> str:
        all_status = len(self._status)
        all_ok = sum(1 for _, status in self._status if status == "ok")
        return f"{all_ok}/{all_status}"

    @property
    def pretty_checks(self) -> str:
        if len(self._status) == 0:
            # bracket text in yellow
            start, end = "\033[93m", "\033[0m"
        elif sum(1 for _, status in self._status if status == "ok") == len(self._status):
            # bracket text in green
            start, end = "\033[92m", "\033[0m"
        else:
            # bracket text in red
            start, end = "\033[91m", "\033[0m"

        return f"{start}{self.checks}{end}"

    @property
    def pretty_state(self) -> str:
        if self.state == InstanceStatus.RUNNING:
            # bracket text in green
            start, end = "\033[92m", "\033[0m"
        elif self.state == InstanceStatus.PENDING:
            # bracket text in blue
            start, end = "\033[94m", "\033[0m"
        elif self.state == InstanceStatus.SHUTTING_DOWN:
            # bracket text in red
            start, end = "\033[91m", "\033[0m"
        else:
            # bracket text in yellow
            start, end = "\033[93m", "\033[0m"

        return f"{start}{self.state.value}{end}"

    @property
    def pretty_id(self) -> str:
        return f"\033[1m{self.instance_id}\033[0m"

    @property
    def pretty_ip(self) -> str:
        # make the ip address italic
        return f"\033[3m{self.public_ip_address or '·'}\033[0m"

    @classmethod
    def from_instance(
        cls,
        description: Union[dict, "InstanceTypeDef"],
        status: Optional[Union[dict, "InstanceStatusTypeDef"]] = None,
    ) -> "InstanceInfo":
        """
        Creates an InstanceInfo object from an EC2 instance dictionary.

        Args:
            instance: Dictionary containing EC2 instance details or boto3 InstanceTypeDef

        Returns:
            An InstanceInfo object populated with the instance details
        """
        name = str(next((tag.get("Value") for tag in description.get("Tags", []) if tag.get("Key") == "Name"), ""))

        instance = cls(
            instance_id=description.get("InstanceId", ""),
            instance_type=description.get("InstanceType", ""),
            image_id=description.get("ImageId", ""),
            state=InstanceStatus(description.get("State", {}).get("Name", "")),
            public_ip_address=description.get("PublicIpAddress", ""),
            public_dns_name=description.get("PublicDnsName", ""),
            name=name,
            created_at=description.get("LaunchTime", datetime.datetime.min),
            tags={tag["Key"]: tag.get("Value", "") for tag in description.get("Tags", []) if "Key" in tag},
            zone=description.get("Placement", {}).get("AvailabilityZone", ""),
        )

        for instance_status, instance_value in (status or {}).items():
            if instance_status.endswith("Status"):
                assert isinstance(instance_value, dict), f"{instance_value} is {type(instance_value)}, not dict"
                instance._update_status(name=instance_status, status=instance_value.get("Status", ""))
        return instance

    @classmethod
    def describe_instances(
        cls,
        instance_ids: list[str] | None = None,
        client: Union["EC2Client", None] = None,
        region: str | None = None,
        project: str | None = None,
        owner: str | None = None,
        contact: str | None = None,
        statuses: list["InstanceStatus"] | None = None,
    ) -> list["InstanceInfo"]:
        """
        Retrieves information about multiple EC2 instances based on filters.

        Args:
            instance_ids: Optional list of instance IDs to filter by
            tags: Optional dictionary of tags to filter instances by
            client: Optional boto3 EC2 client to use
            region: AWS region to query (defaults to class region)

        Returns:
            List of InstanceInfo objects matching the specified criteria
        """

        # default statuses
        statuses = statuses or InstanceStatus.active()

        # Use provided client or create a new one with the specified region
        client = client or boto3.client("ec2", region_name=region or cls.region)

        filters = []

        # Add filters based on parameters
        filters.append({"Name": "instance-state-name", "Values": [status.value for status in statuses]})

        if instance_ids:
            filters.append({"Name": "instance-id", "Values": instance_ids})

        # raise a warning if owner is set (deprecated)
        if owner:
            logger.warning("The owner tag is deprecated. Use the contact tag instead.")

        if contact:
            filters.append({"Name": "tag:Contact", "Values": [contact]})

        if project:
            filters.append({"Name": "tag:Project", "Values": [project]})

        # Query EC2 API with filters if any are specified
        response_describe = client.describe_instances(**({"Filters": filters} if filters else {}))

        # Get instance status
        response_status = client.describe_instance_status(
            InstanceIds=[
                id_
                for reservation in response_describe.get("Reservations", [])
                for instance in reservation.get("Instances", [])
                if isinstance(id_ := instance.get("InstanceId"), str)
            ]
        )

        instance_statuses = {
            id_: status
            for status in response_status.get("InstanceStatuses", [])
            if isinstance((id_ := status.get("InstanceId")), str)
        }

        # Extract and convert instances from the response
        instances = [
            InstanceInfo.from_instance(description=instance, status=instance_statuses.get(id_, None))
            for reservation in response_describe.get("Reservations", [])
            for instance in reservation.get("Instances", [])
            if isinstance((id_ := instance.get("InstanceId")), str)
        ]

        return sorted(instances, key=lambda x: x.name)

    @classmethod
    def describe_instance(
        cls,
        instance_id: str,
        client: Union["EC2Client", None] = None,
        region: str | None = None,
    ) -> "InstanceInfo":
        """
        Retrieves detailed information about a specific EC2 instance.

        Args:
            instance_id: The ID of the instance to describe
            client: Optional boto3 EC2 client to use
            region: AWS region where the instance is located

        Returns:
            InstanceInfo object containing the instance details
        """
        client = client or boto3.client("ec2", region_name=region or cls.region)
        assert client, "EC2 client is required"

        response = client.describe_instances(InstanceIds=[instance_id])
        return InstanceInfo.from_instance(response.get("Reservations", [])[0].get("Instances", [])[0])

    def pause(self, client: Union["EC2Client", None] = None, wait_for_completion: bool = True) -> bool:
        """
        Pauses this EC2 instance.

        Args:
            client: Optional boto3 EC2 client to use
            wait_for_completion: If True, wait until the instance is fully paused

        Returns:
            True if pause was successful, False otherwise
        """
        client = client or boto3.client("ec2", region_name=self.region)
        assert client, "EC2 client is required"

        # check if the instance is already paused
        if self.state == InstanceStatus.STOPPED:
            logger.info(f"Instance {self.instance_id} ({self.name}) is already paused")
            return True

        try:
            # Pause the instance
            logger.info(f"Pausing instance {self.instance_id} ({self.name})...")
            client.stop_instances(InstanceIds=[self.instance_id])

            # Wait for the instance to be fully paused if requested
            if wait_for_completion:
                logger.info("Waiting for instance to be fully paused...")
                waiter = client.get_waiter("instance_stopped")
                waiter.wait(InstanceIds=[self.instance_id])
                logger.info(f"Instance {self.instance_id} has been paused")

            return True

        except client.exceptions.ClientError as e:
            logger.error(f"Error pausing instance {self.instance_id}: {str(e)}")
            return False

    def resume(self, client: Union["EC2Client", None] = None, wait_for_completion: bool = True) -> bool:
        """
        Resumes this EC2 instance.

        Args:
            client: Optional boto3 EC2 client to use
            wait_for_completion: If True, wait until the instance is fully resumed

        Returns:
            True if resume was successful, False otherwise
        """
        client = client or boto3.client("ec2", region_name=self.region)
        assert client, "EC2 client is required"

        # check if the instance is already running
        if self.state == InstanceStatus.RUNNING:
            logger.info(f"Instance {self.instance_id} ({self.name}) is already running")
            return True

        try:
            # Resume the instance
            logger.info(f"Resuming instance {self.instance_id} ({self.name})...")
            client.start_instances(InstanceIds=[self.instance_id])

            # Wait for the instance to be fully resumed if requested
            if wait_for_completion:
                logger.info("Waiting for instance to be fully resumed...")
                waiter = client.get_waiter("instance_running")
                waiter.wait(InstanceIds=[self.instance_id])
                logger.info(f"Instance {self.instance_id} has been resumed")

            return True

        except client.exceptions.ClientError as e:
            logger.error(f"Error resuming instance {self.instance_id}: {str(e)}")
            return False

    def terminate(self, client: Union["EC2Client", None] = None, wait_for_termination: bool = True) -> bool:
        """
        Terminates this EC2 instance.

        Args:
            client: Optional boto3 EC2 client to use
            wait_for_termination: If True, wait until the instance is fully terminated

        Returns:
            True if termination was successful, False otherwise
        """
        client = client or boto3.client("ec2", region_name=self.region)
        assert client, "EC2 client is required"

        try:
            # Terminate the instance
            logger.info(f"Terminating instance {self.instance_id} ({self.name})...")
            client.terminate_instances(InstanceIds=[self.instance_id])

            # Wait for the instance to be fully terminated if requested
            if wait_for_termination:
                logger.info("Waiting for instance to be fully terminated...")
                waiter = client.get_waiter("instance_terminated")
                waiter.wait(InstanceIds=[self.instance_id])
                logger.info(f"Instance {self.instance_id} has been terminated")

            return True

        except client.exceptions.ClientError as e:
            # Handle common error cases
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "InvalidInstanceID.NotFound":
                logger.info(f"Instance {self.instance_id} not found in region {self.region}")
            else:
                logger.error(f"Error terminating instance {self.instance_id}: {str(e)}")
            return False

    @classmethod
    def get_latest_ami_id(cls, instance_type: str, client: Union["SSMClient", None] = None) -> str:
        """
        Get the latest AMI ID for a given instance type and region
        """
        is_arm = instance_type.startswith(("a1", "c6g", "c7g", "m6g", "m7g", "r6g", "r7g", "t4g", "im4gn", "g5g"))

        # Select appropriate AMI based on architecture
        if is_arm:
            image_id = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
        else:
            image_id = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"

        client = client or boto3.client("ssm")
        assert client, "SSM client is required"

        parameter = client.get_parameter(Name=image_id, WithDecryption=False)
        ami_id = parameter.get("Parameter", {}).get("Value")
        assert ami_id, f"No AMI ID found for {image_id}"
        return ami_id

    @classmethod
    def create_instance(
        cls,
        instance_type: str,
        tags: dict[str, str],
        region: str,
        ami_id: str | None = None,
        wait_for_completion: bool = True,
        key_name: str | None = None,
        storage_type: str | None = None,
        storage_size: int | None = None,
        storage_iops: int | None = None,
        client: Union["EC2Client", None] = None,
    ) -> "InstanceInfo":
        """
        Creates a new EC2 instance and waits until it's running.

        Args:
            instance_type: The EC2 instance type (e.g., 't2.micro')
            tags: Dictionary of tags to apply to the instance
            region: AWS region where to launch the instance
            ami_id: AMI ID to use (defaults to Amazon Linux 2 in the specified region)
            wait_for_completion: Whether to wait for the instance to be running
            key_name: Name of the key pair to use for SSH access to the instance
            storage_type: Type of EBS storage (e.g., 'gp2', 'gp3'). If None, uses AWS default
            storage_size: Size of root volume in GB. If None, uses AWS default
            storage_iops: IOPS for the root volume. If None, uses AWS default
            client: Optional boto3 EC2 client to use

        Returns:
            InstanceInfo object representing the newly created EC2 instance
        """
        # Initialize the EC2 client with the specified region
        client = client or boto3.client("ec2", region_name=region)
        assert client, "EC2 client is required"

        # If AMI ID is not provided, use a default Amazon Linux 2023 AMI (x86_64 or arm64 based on instance type)
        ami_id = ami_id or cls.get_latest_ami_id(instance_type)

        # Prepare the tags format required by EC2
        tag_specifications = [
            {"ResourceType": "instance", "Tags": [{"Key": key, "Value": value} for key, value in tags.items()]}
        ]

        # Prepare instance launch parameters
        launch_params = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "TagSpecifications": tag_specifications,
        }

        # Add key pair if provided
        if key_name:
            launch_params["KeyName"] = key_name

        # Add block device mapping if storage parameters are provided
        if storage_size is not None:
            launch_params.update(
                {
                    "BlockDeviceMappings": [
                        {
                            "DeviceName": "/dev/xvda",
                            "Ebs": {
                                "DeleteOnTermination": True,
                                "VolumeSize": storage_size,
                                **({"VolumeType": storage_type} if storage_type else {}),
                                **({"Iops": storage_iops} if storage_iops else {}),
                            },
                        }
                    ]
                }
            )

        # Launch the EC2 instance
        response = client.run_instances(**launch_params)

        # Extract instance information from the response
        if (instance_def := next(iter(response["Instances"]), None)) is None:
            raise Exception("No instance created")
        else:
            instance = InstanceInfo.from_instance(instance_def)

        # Log instance creation
        logger.info(f"Created instance {instance.instance_id}")

        # Wait for the instance to be in running state if requested
        if wait_for_completion:
            logger.info("Waiting for instance to enter 'running' state...")
            waiter = client.get_waiter("instance_running")
            waiter.wait(InstanceIds=[instance.instance_id])

            # Additionally wait for status checks to pass
            logger.info("Instance is running. Waiting for status checks to pass...")
            waiter = client.get_waiter("instance_status_ok")
            waiter.wait(InstanceIds=[instance.instance_id])
            logger.info(f"Instance {instance.instance_id} is now available and ready to use")

        return instance


class Session:
    def __init__(
        self,
        instance_id: str,
        region: str = "us-east-1",
        private_key_path: str | None = None,
        user: str = "ec2-user",
        client: Union["EC2Client", None] = None,
    ):

        self.private_key_path = private_key_path
        self.user = user
        self.instance = InstanceInfo.describe_instance(instance_id=instance_id, region=region, client=client)

    def make_ssh_client(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client

    def connect(self) -> paramiko.SSHClient:
        client = self.make_ssh_client()
        if self.private_key_path:
            private_key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
            client.connect(self.instance.public_ip_address, username=self.user, pkey=private_key)
        else:
            client.connect(self.instance.public_ip_address, username=self.user)

        return client

    def shell(self, client: paramiko.SSHClient) -> paramiko.Channel:
        # Get local terminal size
        term_width, term_height = (t := shutil.get_terminal_size()).columns, t.lines

        # Open a channel and invoke a shell
        channel = client.invoke_shell()

        # Set the remote terminal size to match local terminal
        channel.resize_pty(width=term_width, height=term_height)

        return channel

    def run_single(self, command: str, timeout: int | None = None, get_pty: bool = False) -> SessionContent:
        # run without screens
        client = self.connect()
        response = client.exec_command(command, timeout=timeout, get_pty=get_pty)
        return SessionContent.from_command(*response)

    def run_in_screen(
        self,
        command: str,
        detach: bool = False,
        timeout: int | None = None,
        terminate: bool = True,
    ) -> SessionContent:
        # run a simple command to check if screen is installed
        if "screen" not in self.run_single("which screen").stdout:
            raise RuntimeError("screen is not installed; cannot run in screen")

        client: None | paramiko.SSHClient = None
        command_hash = hashlib.md5(command.encode()).hexdigest()[:12]

        try:
            client = self.connect()

            logger.info(f"Connected to {self.instance.public_ip_address}")

            channel = self.shell(client)

            # Create a new screen session
            screen_name = f"olmo-cookbook-{command_hash}-{time.time()}"
            channel.send(f"screen -S {screen_name}\n".encode("utf-8"))
            time.sleep(0.1)  # Wait for screen to initialize

            completion_marker = f"CMD_COMPLETED_{command_hash[:20]}"

            # send the command to the server
            if terminate:
                channel.send(f"{command}; echo '{completion_marker}'; screen -X quit".encode("utf-8"))
            else:
                channel.send(f"{command}; echo '{completion_marker}'".encode("utf-8"))

            # send enter to run the command
            channel.send(b"\n")

            # Wait for the command to complete (looking for our marker)
            buffer = ""
            start_time = time.time()

            if detach:
                # sleep 1 second to ensure the command has time to communicate with the server
                time.sleep(1)
                # Detach from screen (Ctrl+A, then d)
                channel.send(b"\x01d")
            else:
                while True:
                    # Check if we have data to receive
                    if channel.recv_ready():
                        chunk = channel.recv(4096).decode("utf-8")
                        buffer += chunk

                        # Print to stdout in real-time
                        sys.stdout.write(chunk)
                        sys.stdout.flush()

                        # Check if our completion marker is in the output
                        if re.search(r"\r?\n" + completion_marker + r"\r?\n", buffer):
                            break

                    # Check for timeout
                    if timeout and time.time() - start_time > timeout:
                        logger.warning(f"\nTimeout waiting for command: {command}")
                        break

                    # Small delay to prevent CPU spinning
                    time.sleep(0.1)

            # we close the connection here to avoid the screen session from hanging
            client.close()
            return SessionContent(stdout=buffer, stderr=buffer)

        except Exception as e:
            client and client.close()  # type: ignore
            raise e

    def run(
        self,
        command: str,
        get_pty: bool = False,
        detach: bool = False,
        terminate: bool = True,
        timeout: int | None = None,
    ) -> SessionContent:

        if self.instance.state != InstanceStatus.RUNNING:
            raise ValueError(f"Instance {self.instance.instance_id} is not running")

        if detach:
            return self.run_in_screen(command, detach=detach, terminate=terminate, timeout=(timeout or 600))
        else:
            return self.run_single(command, get_pty=get_pty, timeout=timeout)


def import_ssh_key_to_ec2(key_name: str, region: str, private_key_path: str) -> str:
    """
    Imports an SSH public key to EC2 as a key pair.

    Args:
        key_name: The name to assign to the key pair in EC2
        region: AWS region where to import the key
        private_key_path: Path to the SSH private key file (defaults to ~/.ssh/id_rsa)

    Returns:
        The key pair ID if the import was successful.
    """
    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client("ec2", region_name=region)

    # Use default SSH private key path if not specified
    if not private_key_path:
        home_dir = os.path.expanduser("~")
        for key_name in DEFAULT_PRIVATE_KEY_NAMES:
            private_key_path = os.path.join(home_dir, ".ssh", key_name)
            if os.path.isfile(private_key_path):
                break

    if not os.path.isfile(private_key_path):
        raise ValueError(f"Private key file not found at {private_key_path}")

    # Generate public key from private key
    try:
        # Load the private key
        with open(private_key_path, "r") as key_file:
            private_key_material = key_file.read().strip()

        # Generate public key using ssh-keygen command
        ssh_public_key_path = f"{private_key_path}.pub"
        try:
            subprocess.run(
                ["ssh-keygen", "-y", "-f", private_key_path],
                check=True,
                stdout=open(ssh_public_key_path, "w"),
                stderr=subprocess.PIPE,
            )
            # Read the generated public key
            with open(ssh_public_key_path, "r") as pub_key_file:
                public_key_material = pub_key_file.read().strip()
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to generate public key: {e.stderr.decode()}")

        logger.info(f"Generated public key from private key and saved to {ssh_public_key_path}")
    except Exception as e:
        logger.error(f"Failed to generate public key from private key: {str(e)}")
        raise ValueError(f"Could not generate public key from private key: {str(e)}")

    h = hashlib.sha256(private_key_material.encode()).hexdigest()
    key_name = f"{key_name}-{h}"

    # Read the public key file
    with open(ssh_public_key_path, "r") as key_file:
        public_key_material = key_file.read().strip()

    h = hashlib.sha256(public_key_material.encode()).hexdigest()
    key_name = f"{key_name}-{h}"

    try:
        # Check if key pair already exists
        try:
            ec2_client.describe_key_pairs(KeyNames=[key_name])
            logger.info(f"Key pair '{key_name}' already exists in region {region}. Skipping import.")
            return key_name
        except ec2_client.exceptions.ClientError:
            # Key doesn't exist, continue with import
            pass

        # Import the key
        response = ec2_client.import_key_pair(KeyName=key_name, PublicKeyMaterial=public_key_material)

        if response["KeyFingerprint"] is None:
            raise ValueError(f"Failed to import key pair '{key_name}' to region {region}")

        logger.info(f"Successfully imported key pair '{key_name}' to region {region}")
        return key_name

    except Exception as e:
        logger.error(f"Error importing SSH key: {str(e)}")
        raise e


def script_to_command(script_path: str, to_file: bool = True) -> str:
    """
    Convert a script to a command that can be executed on an EC2 instance.

    Args:
        script_path: Path to the script to convert
        to_file: Whether to save the script to a file and run it from there

    Returns:
        The command to execute the script on the EC2 instance
    """
    # check if the script path is a file
    assert os.path.isfile(script_path), f"Script file not found: {script_path}"

    # read the script content
    with open(script_path, "rb") as f:
        script_content = f.read()

    # encode script using base64
    b64_script_content = base64.b64encode(script_content).decode()

    if to_file:
        file_name, extension = os.path.splitext(os.path.basename(script_path))
        h = hashlib.sha256(script_content).hexdigest()
        script_path = f"{file_name}-{h}{extension}"
        return f"echo '{b64_script_content}' | base64 -d > {script_path} && chmod +x {script_path} && bash {script_path}"
    else:
        return f"echo {b64_script_content} | base64 -d | bash"


@click.group()
def cli():
    pass


T = TypeVar("T", bound=Callable)


def common_cli_options(f: T) -> T:
    ssh_home = os.path.join(os.path.expanduser("~"), ".ssh")
    default_key_names = ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"]
    default_key_path = next(
        (
            os.path.join(ssh_home, key_name)
            for key_name in default_key_names
            if os.path.exists(os.path.join(ssh_home, key_name))
        ),
        None,
    )

    def validate_command_or_script(
        ctx: click.Context, param: click.Parameter, value: str | None
    ) -> str | list[str] | None:
        if param.name == "script" and value is not None:
            if ctx.params.get("command", None) is not None:
                raise click.UsageError("Cannot provide both --command and --script")
            if os.path.isfile(value):
                return os.path.abspath(value)
            if os.path.isdir(value):
                # get all the scripts in the scripts directory
                scripts = [
                    os.path.abspath(file_path)
                    for root, _, files in os.walk(value)
                    for file_name in files
                    if os.path.isfile(file_path := os.path.join(root, file_name)) and os.access(file_path, os.X_OK)
                ]
                assert len(scripts) > 0, "No executable scripts found in the given directory"
                return scripts
            raise click.UsageError(f"Script file or directory not found: {value}")
        elif param.name == "command" and value is not None:
            if ctx.params.get("script", None) is not None:
                raise click.UsageError("Cannot provide both --command and --script")
            return value

    click_decorators = [
        click.option("-n", "--name", type=str, required=True, help="Cluster name"),
        click.option("-t", "--instance-type", type=str, default="i4i.xlarge", help="Instance type"),
        click.option("-N", "--number", type=int, default=1, help="Number of instances"),
        click.option("-r", "--region", type=str, default="us-east-1", help="Region"),
        click.option("-T", "--timeout", type=int, default=None, help="Timeout for the command"),
        click.option(
            "-o",
            "--owner",
            type=str,
            default=os.getenv("USER") or os.getenv("USERNAME"),
            help="Owner of the cluster. Useful for cost tracking.",
        ),
        click.option(
            "-S/-NS",
            "--spindown/--no-spindown",
            type=bool,
            default=False,
            help="Whether to have the instance self-terminate after the command is run",
        ),
        click.option(
            "-i",
            "--instance-id",
            multiple=True,
            default=None,
            type=click.UNPROCESSED,
            callback=lambda _, __, value: list(value) or None,
            help="Instance ID to work on; can be used multiple times. If none, command applies to all instances",
        ),
        click.option(
            "-k",
            "--ssh-key-path",
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            default=default_key_path,
            help="Path to the SSH private key file",
        ),
        click.option(
            "-a",
            "--ami-id",
            type=str,
            default=None,
            help="AMI ID to use for the instances",
        ),
        click.option(
            "-d/-nd",
            "--detach/--no-detach",
            "detach",
            type=bool,
            default=False,
            help="Whether to detach from the instances after creation",
        ),
        click.option(
            "-c",
            "--command",
            type=str,
            default=None,
            callback=validate_command_or_script,
            help="Command to execute on the instances",
        ),
        click.option(
            "-s",
            "--script",
            type=click.Path(exists=True, file_okay=True, dir_okay=True),
            default=None,
            callback=validate_command_or_script,
            help="Path to a script file or directory containing scripts to execute on the instances",
        ),
    ]

    return reduce(lambda f, decorator: decorator(f), click_decorators, f)


@common_cli_options
@click.option(
    "--storage-type",
    type=click.Choice(["gp3", "gp2", "io1", "io2", "io2e", "st1", "sc1"]),
    default=None,
    help="Storage type to use for the instances",
)
@click.option(
    "--storage-size",
    type=int,
    default=None,
    help="Storage size to use for the instances",
)
@click.option(
    "--storage-iops",
    type=int,
    default=None,
    help="IOPS for the root volume",
)
def create_instances(
    name: str,
    instance_type: str,
    number: int,
    region: str,
    owner: str,
    ssh_key_path: str,
    ami_id: str | None,
    detach: bool,
    storage_type: str | None,
    storage_size: int | None,
    storage_iops: int | None,
    **kwargs,
):
    """
    Spin up one or more EC2 instances.

    Args:
        name: Project name to tag instances with
        instance_type: EC2 instance type (e.g., t2.micro)
        number: Number of instances to create
        region: AWS region to create instances in
        owner: Owner name to tag instances with
        ssh_key_path: Path to SSH private key file
        ami_id: Optional AMI ID to use (if None, latest Amazon Linux 2 AMI will be used)
        detach: Whether to detach after creation without waiting for completion
        storage_type: Type of EBS storage (e.g., 'gp2', 'gp3'). If None, uses AWS default
        storage_size: Size of root volume in GB. If None, uses AWS default
        storage_iops: IOPS for the root volume. If None, uses AWS default
        **kwargs: Additional keyword arguments

    Returns:
        List of created InstanceInfo objects
    """
    logger.info(f"Creating {number} instances of type {instance_type} in region {region}")

    assert owner is not None, "Cannot determine owner from environment; please specify --owner"

    # Create tags for the instances
    tags = {"Project": name, "Contact": owner}
    logger.info(f"Using tags: {tags}")

    # Import SSH key to EC2
    logger.info(f"Importing SSH key to EC2 in region {region}...")
    key_name = import_ssh_key_to_ec2(key_name=f"{owner}-{name}", region=region, private_key_path=ssh_key_path)
    logger.info(f"Imported SSH key with name: {key_name}")

    # Check for existing instances with the same tags to determine starting index
    existing_instances = InstanceInfo.describe_instances(
        region=region,
        project=name,
        statuses=InstanceStatus.unterminated(),
    )
    if len(existing_instances) > 0:
        logger.info(f"Found {len(existing_instances)} existing instances with the same tags.")
        # Extract the highest numeric suffix from existing instance names
        start_id = (
            max(
                int(_match.group(1))
                for instance in existing_instances
                if (_match := re.search(r"-(\d+)$", instance.name)) is not None
            )
            + 1
        )
        logger.info(f"Will start numbering new instances from {start_id}")
    else:
        start_id = 0
        logger.info("No existing instances found. Starting with index 0")

    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client("ec2", region_name=region)
    logger.debug(f"Initialized EC2 client for region {region}")

    instances = []
    total_to_create = start_id + number

    # Create each instance
    for i in range(start_id, total_to_create):
        logger.info(f"Creating instance {i + 1 - start_id} of {number} (index: {i})...")

        instance = InstanceInfo.create_instance(
            instance_type=instance_type,
            tags=tags | {"Name": f"{name}-{i:04d}"},  # Add Name tag with index
            key_name=key_name,
            region=region,
            ami_id=ami_id,
            wait_for_completion=not detach,
            client=ec2_client,
            storage_type=storage_type,
            storage_size=storage_size,
            storage_iops=storage_iops,
        )
        logger.info(f"Created instance {instance.instance_id} with name {instance.name}")
        instances.append(instance)

    logger.info(f"Successfully created {len(instances)} instances")
    return instances


@common_cli_options
def list_instances(
    name: str,
    region: str,
    instance_id: list[str] | None,
    **kwargs,
):
    """
    List all instances with the given name and owner.

    Args:
        name: Project name to filter instances by
        region: AWS region to search in
        owner: Owner name to filter instances by
        **kwargs: Additional keyword arguments
    """
    logger.info(f"Listing instances with project={name} in region {region}")

    client = boto3.client("ec2", region_name=region)

    # Retrieve matching instances
    instances = InstanceInfo.describe_instances(
        region=region,
        project=name,
        statuses=InstanceStatus.unterminated(),
        client=client,
    )
    logger.info(f"Found {len(instances)} matching instances")

    # Display instance details
    for i, instance in enumerate(sorted(instances, key=lambda x: x.name)):
        if instance_id is not None and instance.instance_id not in instance_id:
            continue

        print(f"Id:     {instance.pretty_id}")
        print(f"Name:   {instance.name}")
        print(f"Type:   {instance.instance_type}")
        print(f"State:  {instance.pretty_state}")
        print(f"IP:     {instance.pretty_ip}")
        print(f"Status: {instance.pretty_checks}")
        print(f"Tags:   {json.dumps(instance.tags, sort_keys=True)}")

        if i < len(instances) - 1:
            # add separator before next instance
            print()


@common_cli_options
def terminate_instances(
    name: str,
    region: str,
    instance_id: list[str] | None,
    detach: bool,
    **kwargs,
):
    """
    Terminate EC2 instances matching the specified criteria.

    Args:
        name: Project name to filter instances by
        region: AWS region where instances are located
        instance_id: Optional list of specific instance IDs to terminate
        detach: Whether to return immediately without waiting for termination to complete
        **kwargs: Additional keyword arguments
    """
    logger.info(f"Terminating instances with project={name} in region {region}")

    client = boto3.client("ec2", region_name=region)

    # Retrieve instances matching the project and owner tags
    instances = InstanceInfo.describe_instances(
        region=region,
        project=name,
        statuses=InstanceStatus.unterminated(),
        client=client,
    )
    logger.info(f"Found {len(instances)} instances matching the specified tags")

    # Filter by instance ID if provided
    if instance_id is not None:
        logger.info(f"Filtering to {len(instance_id)} specified instance IDs")
        instances = [instance for instance in instances if instance.instance_id in instance_id]
        logger.info(f"After filtering, {len(instances)} instances will be terminated")

    # Terminate each instance
    for instance in instances:
        logger.info(f"Terminating instance {instance.instance_id} ({instance.name})")
        success = instance.terminate(wait_for_termination=not detach, client=client)
        if success:
            logger.info(f"Successfully terminated instance {instance.instance_id} ({instance.name})")
        else:
            logger.error(f"Failed to terminate instance {instance.instance_id} ({instance.name})")

    logger.info(f"Termination commands completed for {len(instances)} instances")


@common_cli_options
def pause_instances(
    name: str,
    region: str,
    instance_id: list[str] | None,
    detach: bool,
    **kwargs,
):
    """
    Pause EC2 instances matching the specified criteria.

    Args:
        name: Project name to filter instances by
        region: AWS region where instances are located
        instance_id: Optional list of specific instance IDs to pause
        detach: Whether to return immediately without waiting for pause to complete
    """
    logger.info(f"Pausing instances with project={name} in region {region}")

    client = boto3.client("ec2", region_name=region)

    # Retrieve instances matching the project and owner tags
    instances = InstanceInfo.describe_instances(
        region=region,
        project=name,
        statuses=[InstanceStatus.RUNNING],
        client=client,
    )
    logger.info(f"Found {len(instances)} instances matching the specified tags")

    # Filter by instance ID if provided
    if instance_id is not None:
        logger.info(f"Filtering to {len(instance_id)} specified instance IDs")
        instances = [instance for instance in instances if instance.instance_id in instance_id]
        logger.info(f"After filtering, {len(instances)} instances will be paused")

    # Pause each instance
    for instance in instances:
        logger.info(f"Pausing instance {instance.instance_id} ({instance.name})")
        success = instance.pause(wait_for_completion=not detach, client=client)
        if success:
            logger.info(f"Successfully paused instance {instance.instance_id} ({instance.name})")
        else:
            logger.error(f"Failed to pause instance {instance.instance_id} ({instance.name})")


@common_cli_options
def resume_instances(
    name: str,
    region: str,
    instance_id: list[str] | None,
    detach: bool,
    **kwargs,
):
    """
    Resume EC2 instances matching the specified criteria.

    Args:
        name: Project name to filter instances by
        region: AWS region where instances are located
        instance_id: Optional list of specific instance IDs to resume
        detach: Whether to return immediately without waiting for resume to complete
    """
    client = boto3.client("ec2", region_name=region)

    logger.info(f"Resuming instances with project={name} in region {region}")

    # Retrieve instances matching the project and owner tags
    instances = InstanceInfo.describe_instances(
        region=region,
        project=name,
        statuses=[InstanceStatus.STOPPED],
        client=client,
    )
    logger.info(f"Found {len(instances)} instances matching the specified tags")

    # Filter by instance ID if provided
    if instance_id is not None:
        logger.info(f"Filtering to {len(instance_id)} specified instance IDs")
        instances = [instance for instance in instances if instance.instance_id in instance_id]
        logger.info(f"After filtering, {len(instances)} instances will be resumed")

    # Resume each instance
    for instance in instances:
        success = instance.resume(wait_for_completion=not detach, client=client)
        if success:
            logger.info(f"Successfully resumed instance {instance.instance_id} ({instance.name})")
        else:
            logger.error(f"Failed to resume instance {instance.instance_id} ({instance.name})")

    logger.info(f"Resume commands completed for {len(instances)} instances")


@common_cli_options
def run_command(
    name: str,
    region: str,
    instance_id: list[str] | None,
    command: str | None,
    script: str | None,
    ssh_key_path: str,
    detach: bool,
    spindown: bool,
    timeout: int | None = None,
    **kwargs,
):
    """
    Run a command or script on EC2 instances.

    Args:
        name: Project name to filter instances by
        region: AWS region where instances are located
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to run command on
        command: Command string to execute on instances
        script: Path to script file to execute on instances
        ssh_key_path: Path to SSH private key for authentication
        detach: Whether to run command in detached mode
        timeout: Optional timeout in seconds for command execution
        **kwargs: Additional keyword arguments
    """
    logger.info(f"Running command on instances with project={name} in region {region}")

    # Validate command/script parameters
    if command is None and script is None:
        raise click.UsageError("Either --command or --script must be provided")

    if command is not None and script is not None:
        raise click.UsageError("--command and --script cannot both be provided")

    # Retrieve instances matching the project and owner tags
    instances = InstanceInfo.describe_instances(region=region, project=name)
    logger.info(f"Found {len(instances)} instances matching the specified tags")

    # Filter by instance ID if provided
    if instance_id is not None:
        logger.info(f"Filtering to {len(instance_id)} specified instance IDs")
        instances = [instance for instance in instances if instance.instance_id in instance_id]
        logger.info(f"After filtering, command will run on {len(instances)} instances")

    # Process each instance
    for instance in instances:
        logger.info(f"Running command on instance {instance.instance_id} ({instance.name})")

        # Convert script to command if script is provided
        command_to_run = script_to_command(script, to_file=True) if script is not None else command
        assert command_to_run is not None, "command and script cannot both be None"  # this should never happen

        if spindown:
            # have the instance self-terminate after the command is run
            command_to_run = f"{command_to_run}; aws ec2 terminate-instances --instance-ids {instance.instance_id}"

        # Create SSH session
        session = Session(
            instance_id=instance.instance_id,
            region=region,
            private_key_path=ssh_key_path,
            user="ec2-user",
        )

        # Verify instance is running
        if instance.state != InstanceStatus.RUNNING:
            logger.error(f"Instance {instance.instance_id} is not running (state: {instance.state})")
            raise ValueError(f"Instance {instance.instance_id} is not running")

        # Execute command
        logger.debug(f"Executing command on {instance.instance_id}")
        output_ = session.run(command_to_run, detach=detach, timeout=timeout)
        print(f"Instance {instance.instance_id}:")
        print(output_)
        print()

    logger.info(f"Command execution completed on {len(instances)} instances")


@common_cli_options
def setup_instances(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    ssh_key_path: str,
    **kwargs,
):
    """
    Set up AWS credentials on EC2 instances.

    Args:
        name: Project name to filter instances by
        region: AWS region where instances are located
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to set up
        ssh_key_path: Path to SSH private key for authentication
        detach: Whether to run setup in detached mode
        **kwargs: Additional keyword arguments
    """
    logger.info(f"Setting up AWS credentials on instances with project={name}, owner={owner} in region {region}")

    # Get AWS credentials from environment
    aws_access_key_id = get_aws_access_key_id()
    aws_secret_access_key = get_aws_secret_access_key()

    # Validate credentials
    if aws_access_key_id is None or aws_secret_access_key is None:
        logger.error("AWS credentials not found in environment variables")
        raise ValueError(
            "No AWS credentials found; "
            "please set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        )

    # Generate AWS config files
    logger.debug("Generating AWS config and credentials files")
    aws_config = make_aws_config()
    aws_credentials = make_aws_credentials(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
    )

    # Base64 encode the AWS config and credentials for secure transfer
    aws_config_base64 = base64.b64encode(aws_config.encode("utf-8")).decode("utf-8")
    aws_credentials_base64 = base64.b64encode(aws_credentials.encode("utf-8")).decode("utf-8")

    screen_install = f"{PACKAGE_MANAGER_DETECTOR} sudo ${{PKG_MANAGER}} install -y screen"
    screen_install_base64 = base64.b64encode(screen_install.encode("utf-8")).decode("utf-8")

    # Create setup command to create AWS config directory and write files
    setup_command = [
        "mkdir -p ~/.aws",
        f"echo '{aws_config_base64}' | base64 -d > ~/.aws/config",
        f"echo '{aws_credentials_base64}' | base64 -d > ~/.aws/credentials",
        f"echo '{screen_install_base64}' | base64 -d > screen_setup.sh",
        "chmod +x screen_setup.sh",
        "./screen_setup.sh",
    ]

    # Execute command on the instances
    logger.info("Running AWS credential setup command on instances")
    run_command(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        command=" && ".join(setup_command),
        script=None,
        ssh_key_path=ssh_key_path,
        detach=False,
        spindown=False,
        screen=True,
    )
    logger.info("AWS credential setup completed")


@common_cli_options
def setup_dolma2_toolkit(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    ssh_key_path: str,
    detach: bool,
    **kwargs,
):
    """
    Set up the Dolma2 toolkit on EC2 instances.

    Args:
        name: Project name to filter instances by
        region: AWS region to search in
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to target
        ssh_key_path: Path to SSH private key file
        detach: Whether to run setup in detached mode
        **kwargs: Additional keyword arguments
    """
    # First set up AWS credentials on the instances
    logger.info(f"Setting up AWS credentials on instances with project={name}, owner={owner} in region {region}")
    setup_instances(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        ssh_key_path=ssh_key_path,
    )

    # Encode the Dolma2 toolkit setup script for secure transfer
    logger.debug("Preparing Dolma2 toolkit setup script")
    base64_encoded_setup_command = base64.b64encode(D2TK_SETUP.encode("utf-8")).decode("utf-8")

    # Create command to write and execute the setup script
    command = [
        f"echo '{base64_encoded_setup_command}' | base64 -d > setup.sh",
        "chmod +x setup.sh",
        "./setup.sh",
    ]

    # Run the Dolma2 toolkit setup command on the instances
    logger.info(f"Setting up Dolma2 toolkit on instances with project={name}, owner={owner}")
    run_command(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        command=" && ".join(command),
        script=None,
        ssh_key_path=ssh_key_path,
        detach=detach,
        spindown=False,
        screen=True,
    )
    logger.info("Dolma2 toolkit setup completed")


@common_cli_options
def setup_dolma_python(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    ssh_key_path: str,
    detach: bool,
    **kwargs,
):
    """
    Set up the Dolma Python on EC2 instances.

    Args:
        name: Project name to filter instances by
        region: AWS region to search in
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to target
        ssh_key_path: Path to SSH private key file
        detach: Whether to run setup in detached mode
        **kwargs: Additional keyword arguments
    """
    # First set up AWS credentials on the instances
    logger.info(f"Setting up AWS credentials on instances with project={name}, owner={owner} in region {region}")
    setup_instances(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        ssh_key_path=ssh_key_path,
    )

    # Encode the Dolma2 toolkit setup script for secure transfer
    logger.debug("Preparing Dolma2 toolkit setup script")
    base64_encoded_setup_command = base64.b64encode(DOLMA_PYTHON_SETUP.encode("utf-8")).decode("utf-8")

    # Create command to write and execute the setup script
    command = [
        f"echo '{base64_encoded_setup_command}' | base64 -d > setup.sh",
        "chmod +x setup.sh",
        "./setup.sh",
    ]

    # Run the Dolma2 toolkit setup command on the instances
    logger.info(f"Setting up Dolma Python on instances with project={name}, owner={owner}")
    run_command(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        command=" && ".join(command),
        script=None,
        ssh_key_path=ssh_key_path,
        detach=detach,
        spindown=False,
        screen=True,
    )
    logger.info("Dolma Python setup completed")


@common_cli_options
@click.option(
    "-g",
    "--github-token",
    type=str,
    default=None,
    help="GitHub personal access token for cloning private repositories",
)
def setup_decon(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    ssh_key_path: str,
    detach: bool,
    github_token: str | None,
    **kwargs,
):
    """
    Set up the DECON toolkit on EC2 instances.

    Args:
        name: Project name to filter instances by
        region: AWS region to search in
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to target
        ssh_key_path: Path to SSH private key file
        detach: Whether to run setup in detached mode
        github_token: GitHub personal access token with read permissions on for cloning private github.com/allenai/decon.git"
        **kwargs: Additional keyword arguments
    """
    # First set up AWS credentials on the instances
    logger.info(f"Setting up AWS credentials on instances with project={name}, owner={owner} in region {region}")
    setup_instances(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        ssh_key_path=ssh_key_path,
    )

    # Get all instances that will be set up
    instances = InstanceInfo.describe_instances(region=region, project=name)

    # Filter by instance ID if provided
    if instance_id is not None:
        instances = [instance for instance in instances if instance.instance_id in instance_id]

    logger.info(f"Setting up Decon on {len(instances)} instances with PMR environment variables")

    # Set up each instance with its specific host index
    for idx, instance in enumerate(instances):
        logger.info(f"Setting up Decon on instance {instance.instance_id} ({instance.name}) with PMR_HOST_INDEX={idx}")

        # Generate the setup script with the specific host index for this instance
        decon_setup_script = make_decon_python_setup(github_token, host_index=idx, host_count=len(instances))
        base64_encoded_setup_command = base64.b64encode(decon_setup_script.encode("utf-8")).decode("utf-8")

        # Create command to write and execute the setup script
        command = [
            f"echo '{base64_encoded_setup_command}' | base64 -d > setup.sh",
            "chmod +x setup.sh",
            "./setup.sh",
        ]

        # Run the setup command on this specific instance
        run_command(
            name=name,
            region=region,
            owner=owner,
            instance_id=[instance.instance_id],
            command=" && ".join(command),
            script=None,
            ssh_key_path=ssh_key_path,
            detach=detach,
            spindown=False,
            screen=True,
        )

    logger.info("Decon setup completed on all instances")


@common_cli_options
def map_commands(
    name: str,
    region: str,
    instance_id: list[str] | None,
    ssh_key_path: str,
    script: list[str],
    spindown: bool,
    **kwargs,
):
    """
    Map and distribute scripts across multiple EC2 instances.

    This function distributes a list of scripts evenly across available instances
    and executes them in parallel.

    Args:
        name: Project name to filter instances by
        region: AWS region to search in
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to target
        ssh_key_path: Path to SSH private key file
        script: List of script paths to distribute and execute
        **kwargs: Additional keyword arguments
    """
    # Set random seed for reproducible distribution
    random.seed(42)

    # Validate script input
    assert isinstance(script, list) and len(script) > 0, "script must be a list with at least one script"

    # Setup a UUID for the job; we will use to track all the jobs
    job_uuid = str(uuid.uuid4())

    logging.info(f"Starting job with UUID: {job_uuid}")

    # Make a copy of the script list and shuffle it for even distribution
    script = script[:]
    random.shuffle(script)
    logger.info(f"Found {len(script):,} scripts to distribute")

    # Get all the instances with the given name and owner
    logger.info(f"Retrieving instances with project={name} in region {region}")
    instances = InstanceInfo.describe_instances(region=region, project=name)

    # filter by instance_id if provided
    if instance_id is not None:
        instances = [instance for instance in instances if instance.instance_id in instance_id]

    assert len(instances) > 0, "No instances found with the given name and owner"
    random.shuffle(instances)

    logger.info(f"Found {len(instances):,} instances to map {len(script):,} scripts to!")

    transfer_scripts_commands: list[list[str]] = []

    # Distribute scripts across instances
    for i, instance in enumerate(instances):
        # Calculate the range of scripts for this instance
        ratio = len(script) / len(instances)
        start_idx = round(ratio * i)
        end_idx = round(ratio * (i + 1))
        instance_scripts = script[start_idx:end_idx]

        # Prepare commands to transfer and execute scripts
        transfer_scripts_commands.append([])

        # Create a directory for the scripts associated with this job
        # and add a run_all.sh script that will run all the scripts at once
        transfer_scripts_commands[-1].append(f"mkdir -p {job_uuid}")
        transfer_scripts_commands[-1].append(f"echo '#!/usr/bin/env bash' >> {job_uuid}/run_all.sh")
        transfer_scripts_commands[-1].append(f"echo 'set -x' >> {job_uuid}/run_all.sh")
        transfer_scripts_commands[-1].append(f"chmod +x {job_uuid}/run_all.sh")

        for one_script in instance_scripts:
            # Read and base64 encode each script
            with open(one_script, "rb") as f:
                base64_encoded_script = base64.b64encode(f.read()).decode("utf-8")

            # Create commands to decode, save, and execute the script
            filename = os.path.basename(one_script)

            # Create commands to decode, save, and execute the script
            cmds = [
                # Decode the script into a file on the instance
                f"echo {base64_encoded_script} | base64 -d > {job_uuid}/{filename}",
                # Make the script executable
                f"chmod +x {job_uuid}/{filename}",
                # Add a start marker to the run_all.log file
                f'echo "$(date) - {job_uuid}/{filename} - START" >> {job_uuid}/run_all.log',
                # Add the script to the run_all.sh file
                f"echo './{job_uuid}/{filename}' >> {job_uuid}/run_all.sh",
                # Add a done marker to the run_all.log file
                f'echo "$(date) - {job_uuid}/{filename} - DONE" >> {job_uuid}/run_all.log',
            ]

            # add scripts to setup
            transfer_scripts_commands[-1].extend(cmds)

        # add scripts to stop the run
        if spindown:
            stop_command = f"aws ec2 stop-instances --instance-ids {instance.instance_id}"
            transfer_scripts_commands[-1].append(f"echo '{stop_command}'>> {job_uuid}/run_all.sh")

    # wrapping up the runner function
    runner_fn = partial(
        run_command, name=name, region=region, ssh_key_path=ssh_key_path, script=None, spindown=False
    )

    for instance, setup_commands in zip(instances, transfer_scripts_commands):
        curr_instance_id = instance.instance_id
        logger.info(f"Copying scripts to instance {curr_instance_id}")
        runner_fn(
            instance_id=[curr_instance_id],
            command="; ".join(setup_commands),
            detach=False,
            screen=False,
        )
    logger.info(f"Scripts transferred on {len(instances):,} instances.")

    for i, instance in enumerate(instances):
        curr_instance_id = instance.instance_id
        logger.info(f"Running {job_uuid}/run_all.sh on instance {curr_instance_id}")
        runner_fn(
            instance_id=[curr_instance_id],
            command=f"bash {job_uuid}/run_all.sh",
            detach=True,
            screen=True,
        )
    logger.info(f"Job {job_uuid} started on {len(instances):,} instances.")


cli.command(name="create")(create_instances)
cli.command(name="list")(list_instances)
cli.command(name="terminate")(terminate_instances)
cli.command(name="run")(run_command)
cli.command(name="setup")(setup_instances)
cli.command(name="setup-d2tk")(setup_dolma2_toolkit)
cli.command(name="setup-dolma-python")(setup_dolma_python)
cli.command(name="setup-decon")(setup_decon)
cli.command(name="map")(map_commands)
cli.command(name="pause")(pause_instances)
cli.command(name="resume")(resume_instances)


if __name__ == "__main__":
    cli({})
