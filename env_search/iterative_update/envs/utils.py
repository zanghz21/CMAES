import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from env_search.competition.update_model.utils import Map
from env_search.utils.logging import get_current_time_str
from tqdm import tqdm
import logging
import json
# logging.getLogger('matplotlib').setLevel(logging.WARNING)


def draw_one_frame(comp_map: Map, agent_pos, task_pos, agents_color):
    fig, ax = plt.subplots(figsize=(comp_map.width//2, comp_map.height//2))
    ax.set_xlim(0, comp_map.width)
    ax.set_ylim(comp_map.height, 0)
    
    # Draw grid
    for x in range(comp_map.width + 1):
        ax.axhline(x, color='black', linewidth=0.5)
    for x in range(comp_map.height + 1):
        ax.axvline(x, color='black', linewidth=0.5)
    
    obstacle_pos = np.where(comp_map.graph==1.0)
    # Draw obstacles
    for pos in zip(obstacle_pos[0], obstacle_pos[1]):
        square = plt.Rectangle((pos[1], pos[0]), 1, 1, color="black")
        ax.add_patch(square)
        
    for pos in comp_map.end_points_ids:
        square = plt.Rectangle((pos[1], pos[0]), 1, 1, color=(0.9, 0.9, 0.9))
        ax.add_patch(square)
    
    for pos in comp_map.home_loc_ids:
        square = plt.Rectangle((pos[1], pos[0]), 1, 1, color=(0.0, 0.9, 0.9))
        ax.add_patch(square)
        
    
    for pos in comp_map.home_loc_ids:
        square = plt.Rectangle((pos[1], pos[0]), 1, 1, color=(0.0, 0.9, 0.9))
        ax.add_patch(square)
    # # Draw circles
    for pos, c in zip(agent_pos, agents_color):
        circle = plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.3, color=c, linewidth=2)
        ax.add_patch(circle)
        
    for pos, c in zip(task_pos, agents_color):
        circle = plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.4, edgecolor=c, facecolor='none', linewidth=3)
        ax.add_patch(circle)
    
    # Hide axes
    # ax.axis('off')
    return fig, ax


def parse_event(results_file, total_steps):
    with open(results_file, "r") as f:
        results_json = json.load(f)
    events = results_json["events"]
    tasks = results_json["tasks"]
    
    # for each agent, we should get the task on each step
    all_agents_tasks = []
    for aid in range(len(events)):
        a_events = events[aid]
        task_id_list = []
        for e in a_events:
            curr_task_id, t, msg = e
            if msg == "assigned":
                while len(task_id_list) <t:
                    task_id_list.append(task_id)
                task_id = curr_task_id
        while len(task_id_list) < total_steps:
            task_id_list.append(task_id)
        
        task_pos_list = []
        for task_id in task_id_list:
            _task_id, r, c = tasks[task_id]
            assert(task_id == _task_id)
            task_pos_list.append([r, c])
        
        all_agents_tasks.append(task_pos_list)
    return all_agents_tasks

        
    
def visualize_simulation(comp_map: Map, agent_pos_hist, results_file=None):
    timestr = get_current_time_str()
    video_dir = os.path.join("video_file", timestr)
    os.makedirs(video_dir, exist_ok=True)
    # Generate and save frames
    agent_pos_hist = np.asarray(agent_pos_hist)
    agent_pos_hist = np.moveaxis(agent_pos_hist, 1, 0)
    num_agents = agent_pos_hist.shape[1]
    agents_color = np.random.random(size=(num_agents, 3))
    
    filenames = []
    if results_file is not None:
        all_agents_tasks = parse_event(results_file, total_steps=len(agent_pos_hist))
        all_agents_tasks = np.array(all_agents_tasks)

    for i, agent_pos in tqdm(enumerate(agent_pos_hist)):
        task_pos = all_agents_tasks[:, i] if results_file is not None else []
        fig, ax = draw_one_frame(comp_map, agent_pos, task_pos, agents_color)
        filename = os.path.join(video_dir, f"frame_{i:04d}.png")
        filenames.append(filename)
        plt.title(f"timestep = {i}")
        plt.savefig(filename)
        plt.close(fig)

    # Compile frames into a video
    with imageio.get_writer(os.path.join(video_dir, "video.mp4"), fps=2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

if __name__ == "__main__":
    map_path = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"
    comp_map = Map(map_path)
    # visualize_simulation(comp_map)
    parse_event("large_files_new/results.json", total_steps=200)