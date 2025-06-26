# Transformer-based Chess Position Evaluator
![Diagram](icon.png)
## 1. Overview

This project is an experimental implementation of a neural network designed to evaluate chess positions. The model, `BOT`, leverages a simple Transformer-based architecture (specifically, a Multi-Head Attention layer) to process the board state and output a scalar value representing the positional advantage.

The primary goal is to train this model to distinguish between strong and weak moves from a given position. The training process utilizes a dataset of chess puzzles from Lichess, attempting to teach the model to assign a high score to the puzzle's solution path and a low score to alternative legal moves.

The implementation uses PyTorch for the neural network, `python-chess` for board representation and move generation, and Pandas for data handling.

## 2. Model Architecture

The core of the project is the `BOT` class, a `torch.nn.Module`.

-   **Input Representation**: A chess board (`chess.Board`) is converted into a 64-element tensor. Each square is represented by an integer token corresponding to the piece on it (1-6 for White's pieces, 7-12 for Black's, 0 for empty).

-   **Embedding Layer**: An `nn.Embedding` layer maps each of the 13 possible piece tokens into a 64-dimensional vector space. This allows the model to learn a dense representation for each piece type.

-   **Attention Layer**: A `nn.MultiheadAttention` layer is applied to the sequence of 64 embedded vectors. The intention is for the model to learn the relationships and interactions between pieces on the board. The layer uses 16 attention heads.

-   **Feed-Forward Network (MLP)**: The output of the attention layer is flattened and passed through a Multi-Layer Perceptron (MLP) with ReLU activations. This MLP acts as the final processing stage to map the complex features from the attention output to a single evaluation score.

-   **Output**: The model produces a single, unbounded scalar value. A positive value is intended to represent an advantage for White, while a negative value represents an advantage for Black.

## 3. Training Methodology

-   **Dataset**: The training data is derived from the `lichess_db_puzzle.csv` file. The script processes each puzzle, which consists of a starting position (FEN) and a sequence of correct moves.

-   **Training Objective & Loss Function**: The training objective is unconventional. For each puzzle, the script iterates through all legal moves from a given position.
    -   If a move is part of the puzzle's official solution, its resulting board state is assigned a target evaluation of `+10.0` (if it's White's turn to find the winning move) or `-10.0` (if it's Black's).
    -   All other legal moves result in a board state with a target evaluation of `0.0`.

    The loss is calculated using Mean Squared Error (`nn.MSELoss`) between the model's output and this target value.

-   **Optimization**: The model parameters are optimized using the Adam optimizer with a learning rate of `1e-4`. The model state (`booty.pth`) is saved after processing each puzzle.

## 4. How to Run

1.  **Dependencies**: Install the required Python libraries:
    ```bash
    pip install torch pandas python-chess berserk
    ```
2.  **Dataset**: Download the Lichess puzzle database and place `lichess_db_puzzle.csv` in the same directory as the script.
3.  **Execution**: Run the script from your terminal:
    ```bash
    python chessboot.py
    ```
    The script will attempt to load model weights from `booty.pth` if it exists and will save its progress back to the same file.

## 5. Potential Research Avenues & Critique

This implementation serves as a basic proof-of-concept but has several areas for improvement from a research perspective:

-   **Training Objective**: The current objective function is a heuristic. Assigning an arbitrary score of +/-10 to "correct" moves and 0 to others is a simplification. A more standard approach would be to train the model on a large dataset of grandmaster games, using the game outcome (Win/Loss/Draw) as the target label, possibly with a sigmoid or tanh activation on the output layer and a Binary Cross-Entropy or similar loss function.
-   **Data Utilization**: Using puzzles is an interesting approach to focus on tactical sequences. However, the model only learns to evaluate the position *after* the move is made. It does not learn a policy to select the best move from the set of legal moves, which would be a more common reinforcement learning setup (e.g., AlphaZero).
-   **Efficiency**: The current training loop iterates through every legal move for every step in every puzzle, which is computationally expensive and slow. A more efficient pipeline would batch process positions and updates.
-   **Architectural Enhancements**: While attention is powerful, the model could benefit from incorporating more explicit spatial information. Hybrid architectures combining convolutional layers (to capture board patterns) with attention mechanisms could yield stronger results.
