import s3fs
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from rich.console import Console
from rich.progress import Progress
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

console = Console()


def check_file_size(fs: s3fs.S3FileSystem, file_path: str) -> str | None:
    """Check if a file is 0 bytes and return its path if so."""
    try:
        info = fs.info(file_path)
        if info.get("size", 0) == 0:
            return file_path
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return None


def find_zero_byte_files(bucket: str, prefix: str, max_workers: int = 12) -> List[str]:
    """Find all zero-byte files under a given S3 prefix using threading."""
    fs = s3fs.S3FileSystem()
    zero_byte_files = []

    try:
        # List all files under the prefix
        console.print(f"[blue]Scanning S3 bucket '{bucket}' with prefix '{prefix}'...[/blue]")

        files = []
        npys = fs.glob(f"{bucket}/{prefix}**/*.npy")
        manifests = fs.glob(f"{bucket}/{prefix}**/*.csv.gz")

        files.extend(npys)
        files.extend(manifests)

        total_files = len(files)
        console.print(f"[green]Found {total_files} files to check[/green]")
        logger.info(f"Found {total_files} files to process")

        if total_files == 0:
            console.print("[yellow]No files found in the specified location[/yellow]")
            return zero_byte_files

        # Use ThreadPoolExecutor to check files concurrently with progress bar
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Checking file sizes...", total=total_files)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(check_file_size, fs, file_path): file_path for file_path in files
                }

                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        zero_byte_files.append(result)
                        logger.info(f"Found zero-byte file: {result}")

                    progress.advance(task)

    except Exception as e:
        logger.error(f"Error accessing S3: {e}")
        console.print(f"[red]Error accessing S3: {e}[/red]")

    return zero_byte_files


# Example usage
if __name__ == "__main__":
    bucket_name = "ai2-llm"
    prefix_path = "preprocessed/olmo3-final/"

    console.print("[bold]Starting zero-byte file check[/bold]")
    logger.info(f"Starting scan of bucket: {bucket_name}, prefix: {prefix_path}")

    zero_byte_files = find_zero_byte_files(bucket_name, prefix_path)

    if zero_byte_files:
        console.print(f"[red]Found {len(zero_byte_files)} zero-byte files:[/red]")
        logger.warning(f"Found {len(zero_byte_files)} zero-byte files")
        for file_path in zero_byte_files:
            console.print(f"  [red]• {file_path}[/red]")
    else:
        console.print("[green]✓ No zero-byte files found.[/green]")
        logger.info("No zero-byte files found")

    console.print("[bold]Scan complete![/bold]")
