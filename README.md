#  Stitching with Deep Q-Learning
The project uses a Deep Q-Learning based agent to solve puzzles made from user's images. I will look to improve the model to be a more generalized solver for any "puzzle" and use the algorithm to train an image stitcher agent.

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

2. Run the learn.py file to train the model.

3. I will update code to save and load weights.

## Configuration

### Hyperparameters
Hyperparameters can be adjusted at the top of the script:

```bash
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
EPSILON = 0.80
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
```
### GRID_SIZE and EPISODES
The GRID_SIZE variable determines the granularity of the puzzle. For example, GRID_SIZE = 3 will divide the image into a 3x3 grid, resulting in 9 pieces.

The EPISODES variable determines the number of episodes the agent will train for.

## Note

Please ensure the image dimensions are evenly divisible by the chosen GRID_SIZE. Resize the image if necessary.


## Contributing

Feel free to fork this repository, make changes, and submit pull requests. Feedback and contributions are welcome

## License

Please see our [license file](docs/licence.md) for more details.
