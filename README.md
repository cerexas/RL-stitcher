# Puzzle Solver with Deep Q-Learning

This project provides an implementation of a Deep Q-Learning based agent to solve jigsaw puzzles using PyTorch. The agent uses a DQN to learn the optimal moves to arrange the shuffled pieces of an image to match the original image.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/cerexas/RL-stitcher.git
   cd RL-stitcher
    ```


2. Install the required packages:
   ```bash
pip install -r requirements.txt
    ```

## Usage

1. Place the image you'd like to use as the puzzle in the project directory and rename it to sample_image.jpg or modify the IMAGE_PATH variable in the code accordingly.

2. Run the learn.py file to train the model

3. I will update code to save and load weights.

## Configuration

Hyperparameters like BATCH_SIZE, LR, GAMMA, etc., can be adjusted at the top of the script.

The GRID_SIZE variable determines the granularity of the puzzle. For example, GRID_SIZE = 3 will divide the image into a 3x3 grid, resulting in 9 pieces.

The EPISODES variable determines the number of episodes the agent will train for.

## Note

Please ensure the image dimensions are evenly divisible by the chosen GRID_SIZE. Resize the image if necessary.


## Contributing

Feel free to fork this repository, make changes, and submit pull requests. Feedback and contributions are welcome

## License

Please see our [License file](docs/licence.md) for more details.
