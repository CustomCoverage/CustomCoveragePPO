import matplotlib.pyplot as plt
from collections import deque
from IPython import display
from collections import deque
import gymnasium as gym
from gymnasium import spaces

import random
import numpy as np
import cv2

def generate_digger_maze(width, height, fill_percent=0.6):
    maze = np.full((height, width), 1, dtype=np.float32)
    target_free_cells = int((width - 2) * (height - 2) * fill_percent)

    cx, cy = random.randint(1, width - 2), random.randint(1, height - 2)
    maze[cy, cx] = 0
    free_cells_count = 1
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while free_cells_count < target_free_cells:
        dx, dy = random.choice(directions)
        nx, ny = cx + dx, cy + dy

        if 1 <= nx < width - 1 and 1 <= ny < height - 1:
            cx, cy = nx, ny
            if maze[cy, cx] == 1:
                maze[cy, cx] = 0
                free_cells_count += 1
                
    return maze

class CoverageEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    def __init__(self, map_paths = [], local_r=5, scale_factor=3, 
                 window_len=20, up_threshold=0.9, down_threshold=0.3, 
                 r_view=1, reward_factor=0.92, is_tw_reward=True, is_frontier=True, np_stack=1,
                 render_mode="rgb_array"):
        super().__init__()
        self.r = local_r
        self.lvls = map_paths
        self.cur_lvl = 0
        self.configs = {
            0: {"size": 16, "fill": 1.0, "max_steps": 160},   
            1: {"size": 16, "fill": 0.9, "max_steps": 450},
            2: {"size": 20, "fill": 0.8, "max_steps": 418},
            3: {"size": 24, "fill": 0.8, "max_steps": 656},
            4: {"size": 32, "fill": 0.8, "max_steps": 1044},
            5: {"size": 40, "fill": 0.7, "max_steps": 2500},
        }
        self.manhaten = True
        self.np_stack = np_stack # 1
        self.np_stack_buffer = deque(maxlen=self.np_stack)
        self.change_lvl = False
        self.render_mode = render_mode
        self.record_countdown = 0
        self.shake_lvl = False
        self.cringe = False
        self.tt_counter = 0
        self.all_spaces = 0
        self.window_len=window_len
        self.episode_history = deque(maxlen=window_len)
        self.episode_history_steps = deque(maxlen=window_len)
        self.episode_history_rewards = deque(maxlen=window_len)
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.low_bound = 0.5
        self.high_bound = 1.5
        self.steps_threshold = 0.64
        self.video_counter = 0
        self.episodes_on_current_level = 0
        self.video_map = {}
        self.r_view = r_view
        self.reward_factor = reward_factor
        self.is_training = True
        self.load_file = False
        self.file_path = ""
        self.is_tw_reward = is_tw_reward # T
        self.is_frontier = is_frontier # T
        self.is_fr_reward = True
        self.target_size = 2*local_r+1
        self.num_scales = scale_factor
        self.map_count = 2
        if self.is_frontier:
            self.map_count += 1
        total_channels = self.np_stack * self.map_count * self.num_scales # 1 * 3 * 3
        self.reset()
        self._update_knowledge()
        
        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(total_channels, self.target_size, self.target_size), 
            dtype=np.float32
        )

    def _update_knowledge(self):
        h, w = self.map_data.shape
        queue = deque([(self.agent_pos[0], self.agent_pos[1], 0)]) # (y, x)
        visited_cells = set([(self.agent_pos[0], self.agent_pos[1])])
        self.opened[self.agent_pos[0], self.agent_pos[1]] = 0

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 
        if self.manhaten:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

        while queue:
            cy, cx, dist = queue.popleft()
            if dist >= self.r:
                continue
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    if self.map_data[ny, nx] == 0 and (ny, nx) not in visited_cells:
                        visited_cells.add((ny, nx))
                        queue.append((ny, nx, dist + 1))
                        self.opened[ny, nx] = 0
                    if self.map_data[ny, nx] == 1:
                        self.opened[ny, nx] = 1

    def _load_map_file(self, path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        mapping = {'#': 1, '.': 0, 'S': 0}
        grid = np.array([[mapping[char] for char in line] for line in lines])
        self.start_pos = np.argwhere(np.array([[c == 'S' for c in l] for l in lines]))[0]
        rng = np.random.default_rng()
        
        zero_indices = np.argwhere(grid == 0)
        random_coords = rng.choice(zero_indices)

        self.agent_pos = random_coords
        self.all_spaces = np.sum(grid == 0)
        self.max_steps = np.sum(grid == 0) * 2
        self.h, self.w = grid.shape
        self.current_step = 0
        self.opened = np.full(grid.shape, fill_value = -1)
        return grid

    def _load_map(self, cfg_json):
        grid = generate_digger_maze(cfg_json["size"], cfg_json["size"], cfg_json["fill"])
        #self.start_pos = np.argwhere(np.array([[c == 'S' for c in l] for l in lines]))[0]
        rng = np.random.default_rng()
        
        zero_indices = np.argwhere(grid == 0)
        random_coords = rng.choice(zero_indices)

        self.agent_pos = random_coords
        self.all_spaces = np.sum(grid == 0)
        self.max_steps = np.sum(grid == 0) * 2
        self.h, self.w = grid.shape
        self.current_step = 0
        self.opened = np.full(grid.shape, fill_value = -1)
        return grid
    
    def get_level(self):
        return self.cur_lvl

    def get_max_levels(self):
        return len(self.configs)

    def set_level(self, level):
        self.cur_lvl = level

    def _update_level_check(self):
        if len(self.episode_history) < self.window_len * self.low_bound or not self.is_training:
            self.map_data = self._load_map(self.configs[self.cur_lvl])
            return
        
        win_rate = np.mean(self.episode_history)
        steps_rate = np.mean(self.episode_history_steps)
        if (win_rate > self.up_threshold and self.all_spaces / steps_rate > self.steps_threshold):
            if (self.cur_lvl >= len(self.configs) - 1):
                self.map_data = self._load_map(self.configs[self.cur_lvl])
                return
            self.cur_lvl += 1
            self.episode_history.clear()
            self.episode_history_steps.clear()
            self.episode_history_rewards.clear()
            self.episodes_on_current_level = 0
            print(f"UP level: {self.cur_lvl}")
            if self.video_map.get(self.cur_lvl, 0) < 4:
                self.record_countdown = 2
                self.video_map[self.cur_lvl] = 1 + self.video_map.get(self.cur_lvl, 0)

        
        if win_rate < self.down_threshold and self.episodes_on_current_level > self.high_bound*self.window_len:
            self.cur_lvl -= 1
            if (self.cur_lvl >= 0):
                self.episode_history.clear()
                self.episode_history_steps.clear()
                self.episode_history_rewards.clear()
                self.episodes_on_current_level = 0
                print(f"Down level: {self.cur_lvl}")
                if self.video_map.get(self.cur_lvl, 0) < 4:
                    self.record_countdown = 2
                    self.video_map[self.cur_lvl] = 1 + self.video_map.get(self.cur_lvl, 0)
            self.cur_lvl = max(self.cur_lvl, 0)
        
        
        self.map_data = self._load_map(self.configs[self.cur_lvl])

    def reset(self, seed=None, options: dict | None = None):
        super().reset(seed=seed)
        if self.load_file:
            self.map_data = self._load_map_file(self.file_path)
        else:
            self._update_level_check()
        self._update_knowledge()
        self.change_lvl = False

        if self.tt_counter % 200 == 0:
            self.record_countdown = 2

        self.episodes_on_current_level += 1

        self.current_step = 0
        self.visited = np.zeros_like(self.map_data)
        self.visited_heat = np.zeros_like(self.map_data).astype(np.float32)
        self.visited[self.agent_pos[0], self.agent_pos[1]] = 1
        self.visited_heat[self.agent_pos[0], self.agent_pos[1]] = 1
        if self.record_countdown > 0:
            self.record_countdown -= 1

        cur_obs = self._get_obs()
        self.np_stack_buffer.append(cur_obs)
        self.np_stack_buffer.append(cur_obs)
        self.np_stack_buffer.append(cur_obs)
        return np.concatenate(self.np_stack_buffer), {}

    def compress_layer(self, layer, mode="min"):
        h, w = layer.shape
        h_new, w_new = h // 2, w // 2
        truncated = layer[:h_new*2, :w_new*2]
        
        b1 = truncated[0::2, 0::2].astype(int)
        b2 = truncated[0::2, 1::2].astype(int)
        b3 = truncated[1::2, 0::2].astype(int)
        b4 = truncated[1::2, 1::2].astype(int)
        
        if mode == "min": 
            return (b1 & b2 & b3 & b4).astype(np.float32)
        elif mode == "max": 
            return (b1 | b2 | b3 | b4).astype(np.float32)
        elif mode == "mean": 
            return ((b1 + b2 + b3 + b4) >= 2).astype(np.float32)
        
    def get_egocentric_compressed_obs(self, global_maps, level_scale, target_r):
        walls = global_maps[0]
        visited = global_maps[1]
        if self.is_frontier:
            frontier = global_maps[2]
        
        for _ in range(level_scale):
            walls = self.compress_layer(walls, mode="min")       
            visited = self.compress_layer(visited, mode="mean")   
            if self.is_frontier:
                frontier = self.compress_layer(frontier, mode="max")  
            
        rx_comp = self.agent_pos[1] // (2 ** level_scale)
        ry_comp = self.agent_pos[0] // (2 ** level_scale)
        
        half_size = target_r // 2
        
        ymin, ymax = ry_comp - half_size, ry_comp + (target_r - half_size)
        xmin, xmax = rx_comp - half_size, rx_comp + (target_r - half_size)
        
        h_comp, w_comp = walls.shape
        
        ego_obs = np.zeros((self.map_count, target_r, target_r), dtype=np.float32)
        ego_obs[0, :, :] = 1.0  
        
        src_ymin, src_ymax = max(0, ymin), min(h_comp, ymax)
        src_xmin, src_xmax = max(0, xmin), min(w_comp, xmax)
        
        dest_ymin = src_ymin - ymin
        dest_ymax = dest_ymin + (src_ymax - src_ymin)
        dest_xmin = src_xmin - xmin
        dest_xmax = dest_xmin + (src_xmax - src_xmin)
        
        if (src_ymax > src_ymin) and (src_xmax > src_xmin):
            ego_obs[0, dest_ymin:dest_ymax, dest_xmin:dest_xmax] = walls[src_ymin:src_ymax, src_xmin:src_xmax]
            if level_scale == 0:
                ego_obs[1, dest_ymin:dest_ymax, dest_xmin:dest_xmax] = self.visited_heat[src_ymin:src_ymax, src_xmin:src_xmax].copy()
            else:
                ego_obs[1, dest_ymin:dest_ymax, dest_xmin:dest_xmax] = visited[src_ymin:src_ymax, src_xmin:src_xmax]
            if self.is_frontier:
                ego_obs[2, dest_ymin:dest_ymax, dest_xmin:dest_xmax] = frontier[src_ymin:src_ymax, src_xmin:src_xmax]
            
        return ego_obs

    def _get_obs(self):
        global_walls = self.opened.copy()
        global_walls = np.where(global_walls == -1, 0, global_walls).astype(np.float32)
        global_visited = self.visited.copy()
        
        
        if self.is_frontier:
            global_frontier = self.get_frontier_map_cv2()
            global_maps = np.stack([global_walls, global_visited, global_frontier], axis=0)
        else:
            global_maps = np.stack([global_walls, global_visited], axis=0)
        
        scale_observations = []
        
        for k in range(self.num_scales):
            ego_window_scale_k = self.get_egocentric_compressed_obs(
                global_maps=global_maps,
                level_scale=k,
                target_r=self.target_size
            )
            scale_observations.append(ego_window_scale_k)
            
        final_observation = np.concatenate(scale_observations, axis=0)
        
        return final_observation.astype(np.float32)

    def get_frontier_map_cv2(self):
        unknown_mask = (self.opened == -1).astype(np.uint8)
        free_mask = (self.opened == 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(unknown_mask, kernel, iterations=1)

        frontier_map = cv2.bitwise_and(dilated_unknown, free_mask)
        return frontier_map.astype(np.uint8)
    
    def get_matrix_tv(self):
        masked_v = np.copy(self.visited).astype(np.int8)
        masked_v[self.map_data == 1] = 1.0
        
        diff_y = np.abs(masked_v[1:, :] - masked_v[:-1, :])
        diff_x = np.abs(masked_v[:, 1:] - masked_v[:, :-1])
        
        return np.sum(diff_y) + np.sum(diff_x)

    def set_visited(self, pos):
        r, c = pos[0], pos[1]  
        k = self.r_view        

        neighborhood = self.map_data[max(0, r-k):min(self.map_data.shape[0], r+k+1), 
                        max(0, c-k):min(self.map_data.shape[1], c+k+1)]
        neighborhood_v = self.visited[max(0, r-k):min(self.visited.shape[0], r+k+1), 
                        max(0, c-k):min(self.visited.shape[1], c+k+1)]

        combined_mask = (neighborhood == 0) & (neighborhood_v == 0)
        new_ones_count = np.sum(combined_mask)
        neighborhood_v[combined_mask] = 1

        self.visited_heat = self.visited_heat * 0.96
        self.visited_heat[(self.visited_heat < 0.1) & (self.visited_heat > 0.01)] = 0.09
        self.visited_heat[pos[0], pos[1]] = 1
        return new_ones_count
    
    def count_reward(self, tv_before, fr_before):
        tv_after = self.get_matrix_tv()
        fr_after = (self.opened == -1).sum()

        delta_tv = tv_after - tv_before
        delta_fr = fr_before - fr_after

        reward = 0
        c_tv = 0.1
        c_fr = 0.2
        tv_reward = -c_tv * (delta_tv / self.max_steps)
        if self.is_tw_reward:
            reward += tv_reward
        if self.is_fr_reward:
            reward += c_fr * (delta_fr / self.all_spaces)
        return reward

    def step(self, action):
        move = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}[action]
        new_pos = self.agent_pos + move

        reward = -0.05 / self.max_steps 
        terminated = False
        truncated = False
        info = {}

        if (0 <= new_pos[0] < self.h and 0 <= new_pos[1] < self.w):
            if self.map_data[new_pos[0], new_pos[1]] == 0:
                tv_before = self.get_matrix_tv()
                fr_before = (self.opened == -1).sum()
                
                reward += self.set_visited(new_pos) / self.all_spaces

                if self.visited_heat[new_pos[0], new_pos[1]] > 0.3:
                    reward -= (0.4 + 0.1*self.r_view) / self.all_spaces
                elif self.visited_heat[new_pos[0], new_pos[1]] < 0.3 and \
                    self.visited_heat[new_pos[0], new_pos[1]] > 0.01:
                    reward -= (0.05 + 0.02*self.r_view) / self.all_spaces

                self.agent_pos = new_pos
                self._update_knowledge()
                reward = self.count_reward(tv_before, fr_before)
            else:
                reward -= 1.5 / self.all_spaces
        else:
            reward -= 1.5 / self.all_spaces
        self.current_step += 1
        if np.sum(self.visited == 1) >= np.sum(self.map_data == 0) * self.reward_factor:
            reward += 0.1
            terminated = True
            self.episode_history.append(1)
            self.episode_history_steps.append(self.current_step)
            self.episode_history_rewards.append(reward)

        if self.current_step >= self.max_steps:
            truncated = True
            self.episode_history.append(0)
            self.episode_history_steps.append(self.current_step)
            self.episode_history_rewards.append(reward)
            
        if terminated or truncated:
            self.tt_counter += 1

        self.np_stack_buffer.append(self._get_obs())
        return np.concatenate(self.np_stack_buffer), reward, terminated, truncated, info

    def change_map(self):
        self.change_lvl = True
        if self.change_lvl and self.cur_lvl < len(self.lvls) - 1:
            self.cur_lvl += 1
            self.record_countdown = 2

    def render(self):
        scale = 20 
        img = np.ones((self.h * scale, self.w * scale, 3), dtype=np.uint8) * 255
        fr = self.get_frontier_map_cv2()
        for r in range(self.h):
            for c in range(self.w):
                if self.map_data[r, c] == 1:
                    color = [20, 20, 20]        
                elif self.visited[r, c] == 1:
                    color = [180, 255, 180]  
                elif fr[r, c] == 1:
                    color = [255, 180, 180]  
                else:
                    color = [255, 255, 255] 

                if self.opened[r, c] == -1:
                    color = [int(x * 0.6) for x in color]

                img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color

        r, c = self.agent_pos
        offset = scale // 4
        img[r*scale+offset:(r+1)*scale-offset, c*scale+offset:(c+1)*scale-offset] = [0, 0, 255]

        if self.render_mode == "rgb_array":
            return img