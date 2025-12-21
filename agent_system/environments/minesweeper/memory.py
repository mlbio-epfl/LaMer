from typing import List, Tuple, Dict, Union, Any

class SimpleMemoryMineSweeper:
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self, num_processes=0):
        self._data = [{} for _ in range(num_processes)]
        self.keys = None
        self.num_processes = num_processes

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, num_processes: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(num_processes)]
        self.num_processes = num_processes
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `num_processes`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.num_processes):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int = 7,
        obs_key: str = "text_obs",
        action_key: str = "action",
        obs_length: int = 2,
        ) -> List[List[Any]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths, dones = [], [], []

        for env_idx in range(self.num_processes):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]

                if len(recent) - j > obs_length:
                    lines.append(
                        f"Action {step_num}: {act}\nObservation {step_num}: ..."
                    )
                else:
                    lines.append(
                        f"Action {step_num}: {act}\nObservation {step_num}:\n{obs}"
                    )
                if 'dones' in rec.keys() and rec['dones']:
                    valid_len = step_num
                    break
            
            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths