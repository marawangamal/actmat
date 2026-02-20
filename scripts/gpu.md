<!-- `1. Terminology Mapping
Think of Slurm as the reservation and PyTorch as the usage.

Slurm: Resource Allocation
Term	Flag	Description
Nodes	--nodes	Number of physical server boxes.
Tasks	--ntasks-per-node	Number of independent Python processes (Set 1 per GPU).
CPUs per Task	--cpus-per-task	Cores dedicated to one process (Feeds the DataLoader).
Memory	--mem	Total System RAM shared by all tasks on the node.
PyTorch: Execution
Term	Context	Description
World Size	Distributed	Total count of all processes (Nodes × Tasks).
Rank	Distributed	The unique global ID of a specific process (0,1,2...).
Local Rank	Distributed	The process ID relative to its node (0,1,2...).
Num Workers	DataLoader	Number of CPU subprocesses used to load data into RAM.
2. Recommended Configuration (4-GPU Example)
To minimize queue time at Mila while maintaining performance:

The Allocation Command:

Bash
salloc --partition=main --gres=gpu:rtx8000:4 --ntasks-per-node=4 --cpus-per-task=4 --mem=32G
The Code Logic:

Tasks = 4: Each process controls exactly one GPU.

CPUs = 16 (Total): Requesting 4 per task (4×4) is easier for Slurm to find than requesting 16 per task (4×16=64).

Num Workers = 4: Matches the allocated CPUs per task for optimal data flow.

3. Troubleshooting & "Hangs"
If your code starts but shows no progress, check these three areas:

Issue	Cause	Fix
SHM Error/Hang	Shared memory (/dev/shm) is full.	Increase --mem or lower num_workers.
DDP Handshake	Processes can't "find" each other.	Ensure MASTER_ADDR and MASTER_PORT are set.
Queue Priority	Existing jobs reduced your priority score.	Check squeue -u $USER. If "Priority," try the main partition.
Would you like me to provide the specific Bash commands to check your Shared Memory usage or Priority score on the Mila cluster? -->


# Slurm and PyTorch: Distributed Training Guide

## 1. Terminology Mapping
Think of **Slurm** as the resource reservation and **PyTorch** as the execution usage.

### Slurm: Resource Allocation
| Term | Flag | Description |
| :--- | :--- | :--- |
| **Nodes** | `--nodes` | Number of physical server boxes. |
| **Tasks** | `--ntasks-per-node` | Number of independent Python processes (Set 1 per GPU). |
| **CPUs per Task** | `--cpus-per-task` | Cores dedicated to one process (Feeds the DataLoader). |
| **Memory** | `--mem` | Total System RAM shared by all tasks on the node. |

### PyTorch: Execution
| Term | Context | Description |
| :--- | :--- | :--- |
| **World Size** | Distributed | Total count of all processes (Nodes × Tasks). |
| **Rank** | Distributed | The unique global ID of a specific process (0, 1, 2...). |
| **Local Rank** | Distributed | The process ID relative to its node (0, 1, 2...). |
| **Num Workers** | DataLoader | Number of CPU subprocesses used to load data into RAM. |

---

## 2. Recommended Configuration (4-GPU Example)
To minimize queue time at Mila while maintaining performance:

### The Allocation Command:
salloc --partition=main --gres=gpu:rtx8000:4 --ntasks-per-node=4 --cpus-per-task=4 --mem=32G

### The Code Logic:
* **Tasks = 4:** Each process controls exactly one GPU.
* **CPUs = 16 (Total):** Requesting 4 per task (4x4) is easier for Slurm to find than requesting 16 per task (4x16=64).
* **Num Workers = 4:** Matches the allocated CPUs per task for optimal data flow.

---

## 3. Troubleshooting & "Hangs"
If your code starts but shows no progress, check these three areas:

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **SHM Error/Hang** | Shared memory (/dev/shm) is full. | Increase --mem or lower num_workers. |
| **DDP Handshake** | Processes can't "find" each other. | Ensure MASTER_ADDR and MASTER_PORT are set. |
| **Queue Priority** | Existing jobs reduced priority score. | Check `squeue -u $USER`. If "Priority," try 'main' partition. |