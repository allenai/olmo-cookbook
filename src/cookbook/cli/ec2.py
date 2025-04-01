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
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Callable, TypeVar, Union

import boto3
import click
import paramiko
from paramiko.channel import ChannelFile, ChannelStdinFile

if TYPE_CHECKING:
    from mypy_boto3_ec2.client import EC2Client
    from mypy_boto3_ec2.type_defs import InstanceTypeDef

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
sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install gcc-c++ -y
sudo yum install htop -y
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin
sudo yum install git -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc
# Do datamap-rs setups
cd
git clone https://github.com/revbucket/datamap-rs.git
cd datamap-rs
s5cmd run examples/all_dressed/s5cmd_asset_downloader.txt
cargo build --release
# Do minhash-rs setups
cd
git clone https://github.com/revbucket/minhash-rs.git
cd minhash-rs
cargo build --release
"""


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


@dataclass(frozen=True)
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
    state: str
    public_ip_address: str
    public_dns_name: str
    name: str
    tags: dict[str, str]
    zone: str
    created_at: datetime.datetime
    region: str = "us-east-1"

    @classmethod
    def from_instance(cls, instance: Union[dict, "InstanceTypeDef"]) -> "InstanceInfo":
        """
        Creates an InstanceInfo object from an EC2 instance dictionary.

        Args:
            instance: Dictionary containing EC2 instance details or boto3 InstanceTypeDef

        Returns:
            An InstanceInfo object populated with the instance details
        """
        name = str(next((tag.get("Value") for tag in instance.get("Tags", []) if tag.get("Key") == "Name"), ""))
        return cls(
            instance_id=instance.get("InstanceId", ""),
            instance_type=instance.get("InstanceType", ""),
            image_id=instance.get("ImageId", ""),
            state=instance.get("State", {}).get("Name", ""),
            public_ip_address=instance.get("PublicIpAddress", ""),
            public_dns_name=instance.get("PublicDnsName", ""),
            name=name,
            created_at=instance.get("LaunchTime", datetime.datetime.min),
            tags={tag["Key"]: tag.get("Value", "") for tag in instance.get("Tags", []) if "Key" in tag},
            zone=instance.get("Placement", {}).get("AvailabilityZone", ""),
        )

    @classmethod
    def describe_instances(
        cls,
        instance_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
        only_running: bool = True,
        client: Union["EC2Client", None] = None,
        region: str | None = None,
    ) -> list["InstanceInfo"]:
        """
        Retrieves information about multiple EC2 instances based on filters.

        Args:
            instance_ids: Optional list of instance IDs to filter by
            tags: Optional dictionary of tags to filter instances by
            only_running: If True, only return instances in 'running' state
            client: Optional boto3 EC2 client to use
            region: AWS region to query (defaults to class region)

        Returns:
            List of InstanceInfo objects matching the specified criteria
        """
        # Use provided client or create a new one with the specified region
        client = client or boto3.client("ec2", region_name=region or cls.region)

        filters = []

        # Add filters based on parameters
        if only_running:
            filters.append({"Name": "instance-state-name", "Values": ["running"]})

        if instance_ids:
            filters.append({"Name": "instance-id", "Values": instance_ids})

        if tags:
            for key, value in tags.items():
                filters.append({"Name": f"tag:{key}", "Values": [value]})

        # Query EC2 API with filters if any are specified
        response = client.describe_instances(**({"Filters": filters} if filters else {}))

        # Extract and convert instances from the response
        instances = [
            InstanceInfo.from_instance(instance)
            for reservation in response.get("Reservations", [])
            for instance in reservation.get("Instances", [])
        ]

        return instances

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
        response = client.describe_instances(InstanceIds=[instance_id])
        return InstanceInfo.from_instance(response.get("Reservations", [])[0].get("Instances", [])[0])

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
    def create_instance(
        cls,
        instance_type: str,
        tags: dict[str, str],
        region: str,
        ami_id: str | None = None,
        wait_for_completion: bool = True,
        key_name: str | None = None,
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
            client: Optional boto3 EC2 client to use

        Returns:
            InstanceInfo object representing the newly created EC2 instance
        """
        # Initialize the EC2 client with the specified region
        client = client or boto3.client("ec2", region_name=region)

        # If AMI ID is not provided, use a default Amazon Linux 2 AMI based on region
        if not ami_id:
            # Get the latest Amazon Linux 2 AMI
            response = client.describe_images(
                Owners=["amazon"],
                Filters=[
                    {"Name": "name", "Values": ["amzn2-ami-hvm-*-x86_64-gp2"]},
                    {"Name": "state", "Values": ["available"]},
                ],
            )

            # Sort images by creation date and get the latest one
            ami_id = sorted(
                response["Images"],
                key=lambda x: x.get("CreationDate", datetime.datetime.min),
                reverse=True,
            )[0].get("ImageId")
            assert ami_id, "No AMI ID found"

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

            while not detach:
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

            if not terminate:
                # Detach from screen (Ctrl+A, then d)
                channel.send(b"\x01d")
                time.sleep(0.5)

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

        assert self.instance.state == "running", f"Instance {self.instance.instance_id} is not running"

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
        ssh_public_key_path: Path to the SSH public key file (defaults to ~/.ssh/id_rsa.pub)

    Returns:
        The key pair ID if the import was successful.
    """
    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client("ec2", region_name=region)

    # Use default SSH public key path if not specified
    if not private_key_path:
        home_dir = os.path.expanduser("~")
        private_key_path = os.path.join(home_dir, ".ssh", "id_rsa")

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
        except ec2_client.exceptions.ClientError as e:
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
        return f"echo {b64_script_content} > {script_path} && chmod +x {script_path} && {script_path}"
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
def create_instances(
    name: str,
    instance_type: str,
    number: int,
    region: str,
    owner: str,
    ssh_key_path: str,
    ami_id: str | None,
    detach: bool,
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
        **kwargs: Additional keyword arguments

    Returns:
        List of created InstanceInfo objects
    """
    logger.info(f"Creating {number} instances of type {instance_type} in region {region}")

    assert owner is not None, "Cannot determine owner from environment; please specify --owner"

    # Create tags for the instances
    tags = {"Project": name, "Owner": owner}
    logger.info(f"Using tags: {tags}")

    # Import SSH key to EC2
    logger.info(f"Importing SSH key to EC2 in region {region}...")
    key_name = import_ssh_key_to_ec2(key_name=f"{owner}-{name}", region=region, private_key_path=ssh_key_path)
    logger.info(f"Imported SSH key with name: {key_name}")

    # Check for existing instances with the same tags to determine starting index
    existing_instances = InstanceInfo.describe_instances(region=region, tags=tags)
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
            tags=tags | {"Name": f"{name}-{i}"},  # Add Name tag with index
            key_name=key_name,
            region=region,
            ami_id=ami_id,
            wait_for_completion=not detach,
        )
        logger.info(f"Created instance {instance.instance_id} with name {instance.name}")
        instances.append(instance)

    logger.info(f"Successfully created {len(instances)} instances")
    return instances


@common_cli_options
def list_instances(
    name: str,
    region: str,
    owner: str,
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
    logger.info(f"Listing instances with project={name}, owner={owner} in region {region}")

    # Create filter tags
    tags = {"Project": name, "Owner": owner}

    # Retrieve matching instances
    instances = InstanceInfo.describe_instances(region=region, tags=tags)
    logger.info(f"Found {len(instances)} matching instances")

    # Display instance details
    for instance in sorted(instances, key=lambda x: x.name):
        print(f"Id:     {instance.instance_id}")
        print(f"Name:   {instance.name}")
        print(f"Type:   {instance.instance_type}")
        print(f"State:  {instance.state}")
        print(f"IP:     {instance.public_ip_address}")
        print(f"Tags:   {json.dumps(instance.tags, sort_keys=True)}")
        print()


@common_cli_options
def terminate_instances(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    detach: bool,
    **kwargs,
):
    """
    Terminate EC2 instances matching the specified criteria.

    Args:
        name: Project name to filter instances by
        region: AWS region where instances are located
        owner: Owner name to filter instances by
        instance_id: Optional list of specific instance IDs to terminate
        detach: Whether to return immediately without waiting for termination to complete
        **kwargs: Additional keyword arguments
    """
    logger.info(f"Terminating instances with project={name}, owner={owner} in region {region}")

    # Retrieve instances matching the project and owner tags
    instances = InstanceInfo.describe_instances(region=region, tags={"Project": name, "Owner": owner})
    logger.info(f"Found {len(instances)} instances matching the specified tags")

    # Filter by instance ID if provided
    if instance_id is not None:
        logger.info(f"Filtering to {len(instance_id)} specified instance IDs")
        instances = [instance for instance in instances if instance.instance_id in instance_id]
        logger.info(f"After filtering, {len(instances)} instances will be terminated")

    # Terminate each instance
    for instance in instances:
        logger.info(f"Terminating instance {instance.instance_id} ({instance.name})")
        success = instance.terminate(wait_for_termination=not detach)
        if success:
            logger.info(f"Successfully terminated instance {instance.instance_id} ({instance.name})")
        else:
            logger.error(f"Failed to terminate instance {instance.instance_id} ({instance.name})")

    logger.info(f"Termination commands completed for {len(instances)} instances")


@common_cli_options
def run_command(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    command: str | None,
    script: str | None,
    ssh_key_path: str,
    detach: bool,
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
    logger.info(f"Running command on instances with project={name}, owner={owner} in region {region}")

    # Validate command/script parameters
    if command is None and script is None:
        raise click.UsageError("Either --command or --script must be provided")

    if command is not None and script is not None:
        raise click.UsageError("--command and --script cannot both be provided")

    # Retrieve instances matching the project and owner tags
    instances = InstanceInfo.describe_instances(region=region, tags={"Project": name, "Owner": owner})
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
        command_to_run = script_to_command(script, to_file=False) if script is not None else command
        assert command_to_run is not None, "command and script cannot both be None"  # this should never happen

        # Create SSH session
        session = Session(
            instance_id=instance.instance_id,
            region=region,
            private_key_path=ssh_key_path,
            user="ec2-user",
        )

        # Verify instance is running
        if instance.state != "running":
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
    detach: bool,
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

    # Create setup command to create AWS config directory and write files
    setup_command = [
        "mkdir -p ~/.aws",
        f"echo '{aws_config_base64}' | base64 -d > ~/.aws/config",
        f"echo '{aws_credentials_base64}' | base64 -d > ~/.aws/credentials",
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
        detach=detach,
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
        detach=detach,
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
        screen=True,
    )
    logger.info("Dolma2 toolkit setup completed")


@common_cli_options
def map_commands(
    name: str,
    region: str,
    owner: str,
    instance_id: list[str] | None,
    ssh_key_path: str,
    script: list[str],
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

    # Make a copy of the script list and shuffle it for even distribution
    script = script[:]
    random.shuffle(script)
    logger.info(f"Found {len(script):,} scripts to distribute")

    # Get all the instances with the given name and owner
    logger.info(f"Retrieving instances with project={name}, owner={owner} in region {region}")
    instances = InstanceInfo.describe_instances(region=region, tags={"Project": name, "Owner": owner})
    assert len(instances) > 0, "No instances found with the given name and owner"
    random.shuffle(instances)

    logger.info(f"Found {len(instances):,} instances to map {len(script):,} scripts to")

    # Distribute scripts across instances
    for i, instance in enumerate(instances):
        # Calculate the range of scripts for this instance
        ratio = len(script) / len(instances)
        start_idx = round(ratio * i)
        end_idx = round(ratio * (i + 1))
        instance_scripts = script[start_idx:end_idx]

        # Prepare commands to transfer and execute scripts
        run_command_scripts = []
        for one_script in instance_scripts:
            # Read and base64 encode each script
            with open(one_script, "rb") as f:
                base64_encoded_script = base64.b64encode(f.read()).decode("utf-8")

            # Create commands to decode, save, and execute the script
            filename = os.path.basename(one_script)
            run_command_scripts.append(f"echo {base64_encoded_script} | base64 -d > {filename}")
            run_command_scripts.append(f"chmod +x {filename}")
            run_command_scripts.append(f"./{filename}")

        # Log which scripts are being sent to which instance
        script_names = [f"`{os.path.basename(s)}`" for s in instance_scripts]
        logger.info(
            f"Running {len(instance_scripts):,} scripts on instance {instance.instance_id}: {'; '.join(script_names)}"
        )

        # Execute the scripts on the instance
        run_command(
            name=name,
            region=region,
            owner=owner,
            instance_id=[instance.instance_id],
            command="; ".join(run_command_scripts),
            script=None,
            ssh_key_path=ssh_key_path,
            detach=True,
            screen=True,
        )


cli.command(name="create")(create_instances)
cli.command(name="list")(list_instances)
cli.command(name="terminate")(terminate_instances)
cli.command(name="run")(run_command)
cli.command(name="setup")(setup_instances)
cli.command(name="setup-d2tk")(setup_dolma2_toolkit)
cli.command(name="map")(map_commands)

if __name__ == "__main__":
    cli({})
