import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
EPSILON = 0.80
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
GRID_SIZE = 3
EPISODES = 1000
IMAGE_PATH = 'sample_image.jpg'

# Rescale images to fit display
scale_factor = 5
# Initialize vector to store steps taken per episode
steps_per_episode = []

class PuzzleEnvironment:
    def __init__(self, image_path, grid_size=4):
        self.image = Image.open(image_path).convert('L')
        self.grid_size = grid_size
        self.chunk_size = min(self.image.size) // grid_size
        self.chunks = self._split_image()
        self.state = np.random.permutation(grid_size*grid_size).reshape(grid_size, grid_size)
        self.possible_swaps = [(i, j) for i in range(self.grid_size*self.grid_size) for j in range(i+1, self.grid_size*self.grid_size)]
        self.locked_positions = np.zeros((grid_size, grid_size), dtype=bool)
        self.visited_states = set()
        

    def get_current_image(self):
        """Constructs an image from the current puzzle state"""
        new_image = Image.new('L', self.image.size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                chunk_idx = self.state[i][j]
                chunk = self.chunks[chunk_idx]
                new_image.paste(chunk, (i * self.chunk_size, j * self.chunk_size))
        return new_image

    def _split_image(self):
        chunks = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = i * self.chunk_size
                upper = j * self.chunk_size
                right = left + self.chunk_size
                lower = upper + self.chunk_size
                chunk = self.image.crop((left, upper, right, lower))
                chunks.append(chunk)
        return chunks
    
    def step(self, action):
        i, j = self.possible_swaps[action]
        i_row, i_col = i // self.grid_size, i % self.grid_size
        j_row, j_col = j // self.grid_size, j % self.grid_size
        reward = 0
        # Check if the pieces involved in the swap are locked
        if self.locked_positions[i_row][i_col] or self.locked_positions[j_row][j_col]:
            reward += -100
        else:
            previous_correct_count = sum([(self.state[i][j] == self.grid_size*i + j) for i in range(self.grid_size) for j in range(self.grid_size)])

            # Swap pieces
            self.state[i_row][i_col], self.state[j_row][j_col] = self.state[j_row][j_col], self.state[i_row][i_col]
            
            current_correct_count = sum([(self.state[i][j] == self.grid_size*i + j) for i in range(self.grid_size) for j in range(self.grid_size)])
            
            # Check if the swap resulted in any progress
            if current_correct_count == previous_correct_count:
                reward += -10  # Penalize no progress
            else:
                reward = current_correct_count * 15  # Giving higher reward for each correct piece
                #reward = (1+current_correct_count)*10  # Giving higher reward for each correct piece

            # Lock the pieces that are in their correct positions
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.state[x][y] == self.grid_size * x + y:
                        self.locked_positions[x][y] = True

        # Convert the state to a tuple and check if it has been visited
        state_tuple = tuple(self.state.flatten())
        if state_tuple in self.visited_states:
            reward -= 20  # Penalize revisiting a state
        else:
            # Add the current state to the visited_states
            self.visited_states.add(state_tuple)

        done = np.array_equal(self.state, np.arange(self.grid_size*self.grid_size).reshape(self.grid_size, self.grid_size))
        return self.state, reward, done

    def reset(self):
        self.state = np.random.permutation(self.grid_size*self.grid_size).reshape(self.grid_size, self.grid_size)
        self.locked_positions.fill(False)

        # Clear the visited states set
        self.visited_states.clear()

        return self.state


class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=None):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size or input_size * (input_size - 1) // 2)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, input_size):
        self.model = DQNNetwork(input_size, output_size=input_size * (input_size - 1) // 2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = []

    def choose_action(self, state):
        if random.random() > EPSILON:
            state = torch.FloatTensor(state).view(1, -1)
            q_values = self.model(state)
            return torch.argmax(q_values).item()
        else:
            return random.randint(0, len(state)**2 - 2)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (GAMMA * next_q_values)

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #print(f"Episode: {episode}, Time in last episode: {elapsed_time} Training loss: {loss.item()}")

def shade_correct_pieces(image, grid_size, chunk_size, state):
    """Shades the correct pieces of the puzzle with a green tint."""
    overlay = image.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if state[i][j] == grid_size * i + j:
                left = i * chunk_size
                upper = j * chunk_size
                right = left + chunk_size
                lower = upper + chunk_size
                cv2.rectangle(overlay, (left, upper), (right, lower), (0, 255, 0), -1)  # Shade with green

    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def get_run_folder():
    """Find the next available run folder."""
    base_folder = 'runs'
    run_num = 1
    while True:
        folder_name = os.path.join(base_folder, str(run_num))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)  # Create the folder
            return folder_name
        run_num += 1

run_folder = get_run_folder()  # Call it once at the beginning to get the current run folder
low_res_image = Image.open(IMAGE_PATH).convert('L').resize((64, 64))
low_res_image.save('low_res_image.jpg')

env = PuzzleEnvironment('low_res_image.jpg', grid_size=GRID_SIZE)
agent = DQNAgent(GRID_SIZE*GRID_SIZE)

for episode in range(EPISODES):
    state = env.reset()
    done = False
    steps = 0  # Initialize steps count for this episode

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        steps += 1  # Increment step count
        if(steps%50==0):
            print(f"Current step: {steps}", end='\r')
        
        # Displaying the images side by side
        original_image = cv2.imread('low_res_image.jpg', cv2.IMREAD_GRAYSCALE)
        current_puzzle_image = np.array(env.get_current_image())
        current_puzzle_image = shade_correct_pieces(current_puzzle_image, GRID_SIZE, env.chunk_size, env.state)

        original_image = cv2.resize(original_image, (original_image.shape[1] * scale_factor, original_image.shape[0] * scale_factor))
        current_puzzle_image = cv2.resize(current_puzzle_image, (current_puzzle_image.shape[1] * scale_factor, current_puzzle_image.shape[0] * scale_factor))

        combined_image = np.hstack((original_image, current_puzzle_image))
        cv2.imshow("Original vs Current", combined_image)
        cv2.waitKey(1)  # Display it for a short duration. Change to higher value if you want longer pauses.
        
        agent.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
        agent.train()

        state = next_state
        
    steps_per_episode.append(steps)  # Append step count for this episode
    print(f"Episode: {episode + 1}, Epsilon: {EPSILON}, Steps taken: {steps}")  # Display steps taken in this episode

    plt.figure(figsize=(10,5))
    plt.plot(steps_per_episode, marker='o', color='b')
    plt.title('Steps Taken per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    chart_path = os.path.join(run_folder, "progress_chart.png")  # Save it in the current run folder
    plt.savefig(chart_path)
    plt.close()  # Close the plot to free up resources

cv2.destroyAllWindows()