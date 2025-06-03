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


sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install g++ -y
sudo yum install htop -y
aws configure set aws_access_key_id XXX
aws configure set aws_secret_access_key XXXX
aws configure set default.region us-east-1


wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin

sudo yum install pip -y
pip install zstandard tqdm
echo 'zpy() { python3 -i -c "import zstandard, json; reader = lambda x : [json.loads(_) for _ in zstandard.ZstdDecompressor().stream_reader(open(x, \"rb\").read()).read().splitlines()]" "$@"; }' >> ~/.bashrc
cat << 'EOF' >> ~/.bashrc

# Function to terminate the current EC2 instance
kys() {
  # Get authentication token for EC2 metadata service
  TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

  # Get the instance ID using the token
  INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id)

  # Terminate the instance
  aws ec2 terminate-instances --instance-ids $INSTANCE_ID
}
EOF


source ~/.bashrc


# Download all the artifacts needed for tagging
s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/WebOrganizer/fasttext/models/Topic/may31_lr05_ng3_n3M6_ova_combined-v3.bin /mnt/raid0/models/
s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/fasttext_models/oh_uc_wc_eli5_fasttext_model_bigram_200k.bin /mnt/raid0/models/


sudo yum install git -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc
git clone https://github.com/revbucket/minhash-rs.git
git clone https://github.com/allenai/datamap-rs.git
