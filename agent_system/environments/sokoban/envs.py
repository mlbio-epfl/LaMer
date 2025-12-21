import ray
import gym
from .sokoban import SokobanEnv
import numpy as np

@ray.remote(num_cpus=0.1)
class SokobanWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of SokobanEnv.
    """
    
    def __init__(self, mode, env_kwargs):
        """Initialize the Sokoban environment in this worker"""
        self.env = SokobanEnv(mode, **env_kwargs)
        ## Code for MetaRL ## To allow restart for MetaRL, we add an env_copy to allow returning to init_state
        self.env_copy = self.env.copy()
    
    def step(self, actions):
        """Execute a step in the environment
        - actions: List[List[int]], list of actions
        """
        wons = []
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            wons.append(info['won'])
        info['won'] = bool(np.any(wons))
        return obs, reward, done, info
    
    def reset(self, seed_for_reset):
        """Reset the environment with given seed"""
        obs, info = self.env.reset(seed=seed_for_reset)
        ## Code for MetaRL ##
        self.env_copy = self.env.copy()
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered

    ## Code for MetaRL ##
    def restart(self):
        '''Get back to init state of the game'''
        self.env = self.env_copy.copy()
        obs = self.env.render(self.env.mode) 
        info = {
            'won': False,
        }
        return obs, info



class SokobanMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the Sokoban environment.
    Each Ray actor creates an independent SokobanEnv instance.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self,
                 seed=0, 
                 env_num=1, 
                 group_n=1, 
                 mode='rgb_array',
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing SokobanEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.mode = mode
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors instead of processes
        self.workers = []
        for i in range(self.num_processes):
            worker = SokobanWorker.remote(self.mode, env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[List[int]], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode='text_with_row_numbers', env_idx=None):
        """
        Request rendering from Ray actor environments.
        Can specify env_idx to get render result from a specific environment,
        otherwise returns a list from all environments.
        """
        if env_idx is not None:
            future = self.workers[env_idx].render.remote(mode)
            return ray.get(future)
        else:
            futures = []
            for worker in self.workers:
                future = worker.render.remote(mode)
                futures.append(future)
            results = ray.get(futures)
            return results

    ## Code for MetaRL ##
    def restart(self):
        '''Get back to init state of the game'''
        futures = [worker.restart.remote() for worker in self.workers]
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        """
        Close all Ray actors
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_sokoban_envs(
        seed=0,
        env_num=1,
        group_n=1,
        mode='rgb_array',
        is_train=True,
        env_kwargs=None):
    return SokobanMultiProcessEnv(seed, env_num, group_n, mode, is_train, env_kwargs=env_kwargs)