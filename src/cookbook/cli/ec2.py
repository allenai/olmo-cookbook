import subprocess
import shlex
import boto3
import base64
import click
import os
import logging
import hashlib
from cookbook.cli.utils import (
    make_aws_config,
    make_aws_credentials,
    get_aws_access_key_id,
    get_aws_secret_access_key,
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
sudo yum install g++ -y
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
git checkout dclm
s5cmd run examples/all_dressed/s5cmd_asset_downloader.txt
cargo build --release
# Do minhash-rs setups
cd
git clone https://github.com/revbucket/minhash-rs.git
cd minhash-rs
git checkout refac2025
cargo build --release
"""


D2TK_TEST_SCRIPT = """
#!/bin/bash

# Check if an argument was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory-name>"
    echo "Example: $0 CC-MAIN-2023-06"
    echo "See one of the folders in s3://ai2-oe-data/contrib/datacomp/DCLM-pool/"
    exit 1
fi

# Store the input argument
X=$1
echo "Processing directory: $X"

# Step 1: Copy from S3 to local storage
echo "Copying data from S3 to local storage..."
s5cmd cp -sp "s3://allennlp-mattj/scratch/test_all_dressed/$X/*" "/mnt/raid0/$X"
mkdir -p "/mnt/raid0/${X}_output"

# Step 2: Run the map operation
echo "Running map operation..."
cd ~/datamap-rs
git checkout dclm
cargo run --release -- map --input-dir "/mnt/raid0/$X" --output-dir "/mnt/raid0/${X}_output" --config examples/all_dressed/config.yaml > "/mnt/raid0/${X}_output/map.log"

# Step 3: Run the deduplication operation on JUST the outputs
echo "Running deduplication..."
cd ~/minhash-rs
git checkout refac2025
cargo run --release -- exact-dedup  --config examples/all_dressed/ed_stub.yaml --input-dir-override "/mnt/raid0/${X}_output/step_final" --output-dir-override "/mnt/raid0/${X}_output/step_final_exactdedup" > "/mnt/raid0/${X}_output/exactdedup.log"


# Step 4: Reshard the output data to be a better size
cd ~/datamap-rs
git checkout reshardh
cargo run --release -- reshard --input-dir "/mnt/raid0/${X}_output/step_final_exactdedup/" --output-dir "/mnt/raid0/${X}_output/step_final_exactdedup_reshard/" --max-lines 65536

# Step 5: Copy results back to S3
echo "Copying results back to S3..."
s5cmd cp -sp "/mnt/raid0/${X}_output/step_final_exactdedup_reshard/" "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed/$X/"
s5cmd cp -sp "/mnt/raid0/${X}_output/step_12/" "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/non_english/$X/"

# Step 6: Clean up local storage
echo "Cleaning up local storage..."
rm -rf "/mnt/raid0/${X}_output"
rm -rf "/mnt/raid0/$X"
rm -rf "/mnt/raid0/ed_working_dir"

echo "Processing complete for $X"
"""


def create_ec2_instance(
    instance_type: str,
    tags: dict[str, str],
    region: str,
    ami_id: str | None = None,
    wait_for_status: bool = True,
    key_name: str | None = None
) -> str:
    """
    Creates a new EC2 instance and waits until it's running.

    Args:
        instance_type: The EC2 instance type (e.g., 't2.micro')
        tags: Dictionary of tags to apply to the instance
        region: AWS region where to launch the instance
        ami_id: AMI ID to use (defaults to Amazon Linux 2 in the specified region)
        wait_for_status: Whether to wait for the instance to be in running state
        key_name: Name of the key pair to use for SSH access to the instance

    Returns:
        The instance ID of the newly created EC2 instance
    """
    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client('ec2', region_name=region)

    # If AMI ID is not provided, use a default Amazon Linux 2 AMI based on region
    if not ami_id:
        # Get the latest Amazon Linux 2 AMI
        response = ec2_client.describe_images(
            Owners=['amazon'],
            Filters=[
                {
                    'Name': 'name',
                    'Values': ['amzn2-ami-hvm-*-x86_64-gp2']
                },
                {
                    'Name': 'state',
                    'Values': ['available']
                }
            ]
        )

        # Sort images by creation date and get the latest one
        ami_id = sorted(
            response['Images'],
            key=lambda x: x['CreationDate'],
            reverse=True
        )[0]['ImageId']

    # Prepare the tags format required by EC2
    tag_specifications = [
        {
            'ResourceType': 'instance',
            'Tags': [{'Key': key, 'Value': value} for key, value in tags.items()]
        }
    ]

    # Prepare instance launch parameters
    launch_params = {
        'ImageId': ami_id,
        'InstanceType': instance_type,
        'MinCount': 1,
        'MaxCount': 1,
        'TagSpecifications': tag_specifications
    }

    # Add key pair if provided
    if key_name:
        launch_params['KeyName'] = key_name

    # Launch the EC2 instance
    response = ec2_client.run_instances(**launch_params)

    # Get the instance ID
    instance_id = response['Instances'][0]['InstanceId']
    logger.info(f"Created instance {instance_id}")

    # Wait for the instance to be in running state if requested
    if wait_for_status:
        logger.info("Waiting for instance to enter 'running' state...")
        waiter = ec2_client.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])

        # Additionally wait for status checks to pass
        logger.info("Instance is running. Waiting for status checks to pass...")
        waiter = ec2_client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[instance_id])
        logger.info(f"Instance {instance_id} is now available and ready to use")

    return instance_id


def list_ec2_instances_by_tags(region: str, tags: dict[str, str]) -> list:
    """
    Lists all EC2 instances in a region that match the specified tags.

    Args:
        region: AWS region to search for instances
        tags: Optional dictionary of tags to filter instances by (key-value pairs)

    Returns:
        List of dictionaries containing instance details (id, state, name, etc.)
    """
    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client('ec2', region_name=region)

    # Prepare filters if tags are provided
    filters = []
    if tags:
        for key, value in tags.items():
            filters.append({
                'Name': f'tag:{key}',
                'Values': [value]
            })

    # Describe instances with the given filters
    if filters:
        response = ec2_client.describe_instances(Filters=filters)
    else:
        response = ec2_client.describe_instances()

    instances = []

    # Extract instance information from the response
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            # Extract instance tags into a more accessible format
            instance_tags = {}
            if 'Tags' in instance:
                for tag in instance['Tags']:
                    instance_tags[tag['Key']] = tag['Value']

            # Extract name from tags if available
            name = instance_tags.get('Name', 'Unnamed')

            # Create a simplified instance object with relevant details
            instance_info = {
                'InstanceId': instance['InstanceId'],
                'InstanceType': instance['InstanceType'],
                'State': instance['State']['Name'],
                'PublicIpAddress': instance.get('PublicIpAddress', 'N/A'),
                'PrivateIpAddress': instance.get('PrivateIpAddress', 'N/A'),
                'Name': name,
                'LaunchTime': instance['LaunchTime'],
                'Tags': instance_tags
            }

            instances.append(instance_info)

    return instances


def terminate_ec2_instance(instance_id: str, region: str, wait_for_termination: bool = True) -> bool:
    """
    Terminates an EC2 instance given its ID.

    Args:
        instance_id: The ID of the instance to terminate
        region: AWS region where the instance is located
        wait_for_termination: Whether to wait until the instance is fully terminated

    Returns:
        Boolean indicating if the termination was successful
    """
    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client('ec2', region_name=region)

    try:
        # Check if the instance exists before attempting to terminate
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        if not response['Reservations'] or not response['Reservations'][0]['Instances']:
            logger.info(f"Instance {instance_id} not found in region {region}")
            return False

        # Get current state
        current_state = response['Reservations'][0]['Instances'][0]['State']['Name']

        # If instance is already terminated or terminating, inform the user
        if current_state in ['terminated', 'shutting-down']:
            logger.info(f"Instance {instance_id} is already {current_state}")
            return True

        # Get instance name if available
        instance_name = "Unnamed"
        if 'Tags' in response['Reservations'][0]['Instances'][0]:
            for tag in response['Reservations'][0]['Instances'][0]['Tags']:
                if tag['Key'] == 'Name':
                    instance_name = tag['Value']
                    break

        # Terminate the instance
        logger.info(f"Terminating instance {instance_id} ({instance_name})...")
        ec2_client.terminate_instances(InstanceIds=[instance_id])

        # Wait for the instance to be fully terminated if requested
        if wait_for_termination:
            logger.info("Waiting for instance to be fully terminated...")
            waiter = ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} has been terminated")

        return True

    except ec2_client.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'InvalidInstanceID.NotFound':
            logger.info(f"Instance {instance_id} not found in region {region}")
        else:
            logger.error(f"Error terminating instance {instance_id}: {str(e)}")
        return False


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
    ec2_client = boto3.client('ec2', region_name=region)

    # Use default SSH public key path if not specified
    if not private_key_path:
        home_dir = os.path.expanduser("~")
        private_key_path = os.path.join(home_dir, ".ssh", "id_rsa")

    if not os.path.isfile(private_key_path):
        raise ValueError(f"Private key file not found at {private_key_path}")

    # Generate public key from private key
    try:
        # Load the private key
        with open(private_key_path, 'r') as key_file:
            private_key_material = key_file.read().strip()

        # Generate public key using ssh-keygen command
        ssh_public_key_path = f"{private_key_path}.pub"
        try:
            subprocess.run(
                ["ssh-keygen", "-y", "-f", private_key_path],
                check=True,
                stdout=open(ssh_public_key_path, 'w'),
                stderr=subprocess.PIPE
            )
            # Read the generated public key
            with open(ssh_public_key_path, 'r') as pub_key_file:
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
    with open(ssh_public_key_path, 'r') as key_file:
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
        response = ec2_client.import_key_pair(
            KeyName=key_name,
            PublicKeyMaterial=public_key_material
        )

        if response['KeyFingerprint'] is None:
            raise ValueError(f"Failed to import key pair '{key_name}' to region {region}")

        logger.info(f"Successfully imported key pair '{key_name}' to region {region}")
        return key_name

    except Exception as e:
        logger.error(f"Error importing SSH key: {str(e)}")
        raise e



def execute_command_on_ec2(
    instance_id: str,
    command: str,
    region: str,
    private_key_path: str,
    user: str = "ec2-user",
    wait_for_response: bool = True,
    timeout: int = 60
) -> tuple[bool, str, str]:
    """
    Executes a command on an EC2 instance over SSH.

    Args:
        instance_id: The ID of the EC2 instance to connect to
        command: The command to execute on the instance
        region: AWS region where the instance is located
        private_key_path: Path to the SSH private key file for authentication
        user: SSH username to use for connection (defaults to ec2-user)
        wait_for_response: Whether to wait for command completion (True) or run in background (False)
        timeout: Connection timeout in seconds (only applicable when wait_for_response is True)

    Returns:
        Tuple containing (success_flag, stdout, stderr)
        - success_flag: Boolean indicating if the command execution was successful
        - stdout: Command's standard output (empty string if wait_for_response is False)
        - stderr: Command's standard error (empty string if wait_for_response is False)
    """
    # Initialize the EC2 client with the specified region
    ec2_client = boto3.client('ec2', region_name=region)

    # Get instance details to obtain the public IP address
    response = ec2_client.describe_instances(InstanceIds=[instance_id])

    if not response['Reservations'] or not response['Reservations'][0]['Instances']:
        logger.error(f"Instance {instance_id} not found in region {region}")
        return False, "", f"Instance {instance_id} not found"

    instance = response['Reservations'][0]['Instances'][0]

    # Check if instance is running
    if instance['State']['Name'] != 'running':
        logger.error(f"Instance {instance_id} is not in running state (current state: {instance['State']['Name']})")
        return False, "", f"Instance is not running (state: {instance['State']['Name']})"

    # Get the public IP address
    if 'PublicIpAddress' not in instance:
        logger.error(f"Instance {instance_id} does not have a public IP address")
        return False, "", "Instance does not have a public IP address"

    public_ip = instance['PublicIpAddress']

    # Check if private key file exists
    if not os.path.isfile(private_key_path):
        logger.error(f"Private key file not found at {private_key_path}")
        return False, "", f"Private key file not found at {private_key_path}"

    # Set correct permissions for the private key file (required by SSH)
    try:
        os.chmod(private_key_path, 0o600)
    except Exception as e:
        logger.warning(f"Could not set permissions on private key file: {str(e)}")

    # Construct SSH command
    ssh_options = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout={timeout}",
        "-i", private_key_path
    ]

    if wait_for_response:
        # Execute command and wait for response
        logger.info(f"Executing command on {instance_id} ({public_ip}) and waiting for response")
        ssh_command = ["ssh"] + ssh_options + [f"{user}@{public_ip}", f"'{command}'"]

        print(" ".join(ssh_command))

        process = subprocess.run(
            shlex.split(" ".join(ssh_command)),
            capture_output=True,
            text=True
        )

        success = process.returncode == 0
        stdout = process.stdout
        stderr = process.stderr

        if success:
            logger.info(f"Command executed successfully on {instance_id}")
        else:
            logger.error(f"Command execution failed on {instance_id}: {stderr}")

        return success, stdout, stderr
    else:
        # Execute command in background
        logger.info(f"Executing command on {instance_id} ({public_ip}) in background")
        # Use nohup to allow the command to continue running after SSH connection closes
        background_cmd = f"nohup {command} > /dev/null 2>&1 &"
        ssh_command = ["ssh"] + ssh_options + [f"{user}@{public_ip}", background_cmd]

        process = subprocess.run(
            ssh_command,
            capture_output=True,
            text=True
        )

        success = process.returncode == 0

        if success:
            logger.info(f"Background command started on {instance_id}")
        else:
            logger.error(f"Failed to start background command on {instance_id}: {process.stderr}")

        return success, "", process.stderr

@click.group()
def cli():
    pass


@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--instance",
    type=str,
    default="i4i.xlarge",
    help="Instance type to use",
)
@click.option(
    "--number",
    type=int,
    default=1,
    help="Number of instances to spin up",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
@click.option(
    "--ssh-key-path",
    type=str,
    default=os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
    help="Path to the SSH private key file",
)
def create_instances(
    name: str,
    instance: str,
    number: int,
    region: str,
    owner: str,
    ssh_key_path: str
):
    """
    Spin up one or more EC2 instances.
    """

    assert owner is not None, "Cannot determine owner from environment; please specify --owner"

    # these are shared tags
    tags = {"Project": name, "Owner": owner}

    # add ssh key to ec2
    logger.info(f"Importing SSH key to EC2 in region {region}...")
    key_name = import_ssh_key_to_ec2(key_name=f"{owner}-{name}", region=region, private_key_path=ssh_key_path)

    # first, look up to see if there are any existing instances with the same tags
    existing_instances = list_ec2_instances_by_tags(region=region, tags=tags)

    for i in range(len(existing_instances), len(existing_instances) + number):
        logger.info(f"Creating instance {i + 1} of {number}...")
        instance_id = create_ec2_instance(
            instance_type=instance,
            tags={**tags, "Name": f"{name}-{i:04d}"},
            region=region,
            wait_for_status=True,
            key_name=key_name,
        )
        logger.info(f"Created instance {instance_id}")


@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
def list_instances(
    name: str,
    region: str,
    owner: str,
):
    """
    List all instances with the given name and owner.
    """

    tags = {"Project": name, "Owner": owner}
    instances = list_ec2_instances_by_tags(region=region, tags=tags)
    for instance in instances:
        # print the instance id, name, state, public ip, private ip, and tags
        print(f"Id:     {instance['InstanceId']}")
        print(f"Name:   {instance['Name']}")
        print(f"State:  {instance['State']}")
        print(f"IP:     {instance['PublicIpAddress']}")
        print("Tags:")
        # Find the maximum length of tag keys for padding
        max_key_length = max(len(key) for key in instance['Tags'].keys()) if instance['Tags'] else 0

        # Print each tag with padded key names
        for key, value in instance['Tags'].items():
            print(f"  {key:{max_key_length}}: {value}")
        print()


@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
@click.option(
    "--instance-id",
    type=str,
    default=None,
    help="Instance ID to terminate; if none, terminate all instances with the given name and owner",
)
def terminate_instances(
    name: str,
    region: str,
    owner: str,
    instance_id: str | None,
):
    if instance_id is None:
        instances = [
            instance['InstanceId']
            for instance in list_ec2_instances_by_tags(region=region, tags={"Project": name, "Owner": owner})
        ]
    else:
        instances = [instance_id]

    for instance in instances:
        success = terminate_ec2_instance(instance_id=instance, region=region, wait_for_termination=True)
        print(f"Id:      {instance}")
        print(f"Success: {success}")
        print()


@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
@click.option(
    "--instance-id",
    type=str,
    default=None,
    help="Instance ID to run the command on; if none, run the command on all instances with the given name and owner",
)
@click.option(
    "--command",
    type=str,
    required=True,
    help="Command to run on the instance",
)
@click.option(
    "--wait-for-response",
    type=bool,
    default=True,
    help="Whether to wait for the command to complete",
)
@click.option(
    "--ssh-key-path",
    type=str,
    default=os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
    help="Path to the SSH private key file",
)
def run_command(
    name: str,
    region: str,
    owner: str,
    instance_id: str | None,
    command: str,
    ssh_key_path: str,
    wait_for_response: bool,
):
    if instance_id is None:
        instances = [
            instance['InstanceId']
            for instance in list_ec2_instances_by_tags(region=region, tags={"Project": name, "Owner": owner})
        ]
    else:
        instances = [instance_id]

    for instance in instances:
        success, stdout, stderr = execute_command_on_ec2(
            instance_id=instance,
            command=command,
            region=region,
            private_key_path=ssh_key_path,
            user="ec2-user",
            wait_for_response=wait_for_response
        )
        print(f"Id:      {instance}")
        print(f"Stdout:  {stdout}")
        print(f"Stderr:  {stderr}")
        print(f"Success: {success}")
        print()


@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
@click.option(
    "--instance-id",
    type=str,
    default=None,
    help="Instance ID to run the command on; if none, run the command on all instances with the given name and owner",
)
@click.option(
    "--ssh-key-path",
    type=str,
    default=os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
    help="Path to the SSH private key file",
)
def setup_instances(
    name: str,
    region: str,
    owner: str,
    instance_id: str | None,
    ssh_key_path: str,
):
    aws_access_key_id = get_aws_access_key_id()
    aws_secret_access_key = get_aws_secret_access_key()

    if aws_access_key_id is None or aws_secret_access_key is None:
        raise ValueError(
            "No AWS credentials found; "
            "please set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        )

    aws_config = make_aws_config()
    aws_credentials = make_aws_credentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # base64 encode the aws config and credentials
    aws_config_base64 = base64.b64encode(aws_config.encode("utf-8")).decode("utf-8")
    aws_credentials_base64 = base64.b64encode(aws_credentials.encode("utf-8")).decode("utf-8")

    setup_command = [
        "mkdir -p ~/.aws",
        f"echo '{aws_config_base64}' | base64 -d > ~/.aws/config",
        f"echo '{aws_credentials_base64}' | base64 -d > ~/.aws/credentials",
    ]

    # execute command on the instance to echo aws config to .aws/config and aws credentials to .aws/credentials
    run_command(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        command=" && ".join(setup_command),
        ssh_key_path=ssh_key_path,
        wait_for_response=True,
    )



@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
@click.option(
    "--instance-id",
    type=str,
    default=None,
    help="Instance ID to run the command on; if none, run the command on all instances with the given name and owner",
)
@click.option(
    "--ssh-key-path",
    type=str,
    default=os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
    help="Path to the SSH private key file",
)
def setup_dolma2_toolkit(
    name: str,
    region: str,
    owner: str,
    instance_id: str | None,
    ssh_key_path: str,
):

    logger.info("Setting up AWS credentials...")
    setup_instances(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        ssh_key_path=ssh_key_path,
    )

    base64_encoded_setup_command = base64.b64encode(D2TK_SETUP.encode("utf-8")).decode("utf-8")

    command = [
        f"echo '{base64_encoded_setup_command}' | base64 -d > setup.sh",
        "chmod +x setup.sh",
        "./setup.sh",
    ]

    logger.info("Setting up data processing...")
    run_command(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        command=" && ".join(command),
        ssh_key_path=ssh_key_path,
        wait_for_response=True,
    )


@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the project connected to the instance",
)
@click.option(
    "--region",
    type=str,
    default="us-east-1",
    help="Region to spin up the instances in",
)
@click.option(
    "--owner",
    type=str,
    default=os.getenv("USER") or os.getenv("USERNAME"),
    help="Owner of the project connected to the instance",
)
@click.option(
    "--instance-id",
    type=str,
    default=None,
    help="Instance ID to run the command on; if none, run the command on all instances with the given name and owner",
)
@click.option(
    "--ssh-key-path",
    type=str,
    default=os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
    help="Path to the SSH private key file",
)
def test_dolma2_toolkit(
    name: str,
    region: str,
    owner: str,
    instance_id: str | None,
    ssh_key_path: str,
):

    base64_encoded_test_command = base64.b64encode(D2TK_TEST_SCRIPT.encode("utf-8")).decode("utf-8")

    command = [
        f"echo '{base64_encoded_test_command}' | base64 -d > test.sh",
        "chmod +x test.sh",
        "./test.sh cc_2021_49_small",
    ]

    logger.info("Testing data processing...")
    run_command(
        name=name,
        region=region,
        owner=owner,
        instance_id=instance_id,
        command="; ".join(command),
        ssh_key_path=ssh_key_path,
        wait_for_response=True,
    )

cli.command(name="create")(create_instances)
cli.command(name="list")(list_instances)
cli.command(name="terminate")(terminate_instances)
cli.command(name="run")(run_command)
cli.command(name="setup")(setup_instances)
cli.command(name="setup-d2tk")(setup_dolma2_toolkit)
cli.command(name="test-d2tk")(test_dolma2_toolkit)


if __name__ == "__main__":
    cli({})
