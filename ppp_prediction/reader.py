import shutil
import subprocess
from typing import overload
from pathlib import Path
import pandas as pd
from io import StringIO

def is_tabix_installed():
    """
    Check if tabix is installed on the system.

    Returns:
        bool: True if tabix is installed, False otherwise.
    """
    return shutil.which("tabix") is not None

def is_tbi_exist(file_path):
    """
    Check if the .tbi index file exists for the given file.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the .tbi index file exists, False otherwise.
    """
    return Path(file_path + ".tbi").exists()

@overload
def tabix_reader(file_path: str, chrom: str, start: int, end: int) -> pd.DataFrame:
    """
    Read data from a tabix-indexed file for a specific genomic region.

    Args:
        file_path (str): Path to the tabix-indexed file.
        chrom (str): Chromosome name.
        start (int): Start position of the region.
        end (int): End position of the region.
        region (None): Unused argument.

    Returns:
        pd.DataFrame: DataFrame containing the data from the specified region.
    """
    ...

@overload
def tabix_reader(file_path: str, region: str) -> pd.DataFrame:
    """
    Read data from a tabix-indexed file for a specific genomic region.

    Args:
        file_path (str): Path to the tabix-indexed file.
        chrom (None): Unused argument.
        start (None): Unused argument.
        end (None): Unused argument.
        region (str): Genomic region in the format "chrom:start-end".

    Returns:
        pd.DataFrame: DataFrame containing the data from the specified region.
    """
    ...

def tabix_reader(file_path=None, chrom=None, start=None, end=None, region=None):
    """
    Read data from a tabix-indexed file for a specific genomic region.

    Args:
        file_path (str, optional): Path to the tabix-indexed file. Defaults to None.
        chrom (str, optional): Chromosome name. Defaults to None.
        start (int, optional): Start position of the region. Defaults to None.
        end (int, optional): End position of the region. Defaults to None.
        region (str, optional): Genomic region in the format "chrom:start-end". Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the data from the specified region.

    Raises:
        Exception: If tabix is not installed or the .tbi index file is not found.
    """

    if all(map(lambda x: x== None, [chrom, start, end, region])):
        print("Will load entire file, as region is not provided.")

        return pd.read_csv(file_path, sep="\t", header=None)
    else:
        print("Will load region from file.")
        if not is_tabix_installed():
            raise Exception("tabix is not installed")
        if not is_tbi_exist(file_path):
            raise Exception(f"{file_path}.tbi not found")
        if not region:
            region = ""
            if chrom:
                region = chrom
            if start:
                region += f":{start}"
            if end:
                region += f"-{end}"
        
            


        cmd = f"tabix -h {file_path} {region}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(cmd)
            raise Exception(result.stderr)
        stdout = result.stdout
        if stdout == "":
            print(f"no data found in {region}")
            return None

        first_line = stdout.split("\n")[0]
        if all([isinstance(i, str) for i in first_line.split("\t")]):
            header = 0
        else:
            header = None

        data = pd.read_csv(StringIO(stdout), sep="\t", header=header)
        return data
