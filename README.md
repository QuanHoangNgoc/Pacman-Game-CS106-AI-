# **Pacman Game Evaluation Function**

A sophisticated evaluation function for the Pacman game utilizing Minimax, AlphaBeta, and Expectimax algorithms.

---
## What is it?

This project presents a comprehensive evaluation function designed for the Pacman game, focusing on optimizing decision-making through advanced adversarial search algorithms. By implementing Minimax, AlphaBeta, and Expectimax strategies, we aim to enhance the gameplay experience by enabling Pacman to make informed moves against ghost opponents.

---
## Why do we do it?

The Pacman game serves as an excellent platform to explore adversarial search algorithms, which are crucial in artificial intelligence. Our goal is to develop an evaluation function that not only maximizes Pacman's score but also ensures survival against ghost adversaries. This project addresses the need for intelligent decision-making in games, providing insights into game theory and AI strategies.

---
## Who is the user?

Our primary users include:

- **Game Developers**: Looking to implement or improve AI in games.
- **AI Researchers**: Interested in studying adversarial search algorithms and their applications.
- **Students**: Learning about game theory and AI through practical examples.

### Demo and Results

The evaluation function has been tested across various maps, yielding impressive results. For example, in a series of experiments, our proposal evaluation function achieved a mean average score of **710.33** and a win rate of **70%**, outperforming traditional evaluation functions.

**[Demo in content file]()**

---
## How did we do it?

The development process involved several key steps:

1. **Theoretical Foundation**: We began by researching adversarial games and search algorithms to establish a solid theoretical background.
2. **Feature Design**: We identified critical features affecting Pacman's decision-making, such as distances to food, capsules, and ghosts.
3. **Evaluation Function Development**: We created a proposal evaluation function that balances short-term goals with long-term strategies.
4. **Algorithm Implementation**: We implemented Minimax, AlphaBeta, and Expectimax algorithms to evaluate the game states effectively.
5. **Testing and Tuning**: Rigorous testing was conducted to fine-tune the evaluation function and optimize performance.

---
## Frameworks and Tools Used

- **Python**: is the primary programming language used for implementing algorithms and evaluation functions due to its simplicity and extensive libraries.
- **Algorithms**: Minimax, AlphaBeta, and Expectimax were chosen for their effectiveness in adversarial search scenarios.
- **Dijkstra's Algorithm**: Utilized for calculating shortest paths, essential for evaluating distances in the game.

These tools were selected for their robustness, ease of use, and relevance to the project objectives.

---
## What did you learn?

Through this project, we learned the importance of balancing short-term and long-term goals in decision-making processes. We also gained insights into the complexities of adversarial search and the significance of feature selection in evaluation functions. The experience highlighted the challenges of optimizing AI behavior in dynamic environments.

---
## Achievements

- Developed a state-of-the-art evaluation function for the Pacman game.
- Achieved a mean average score of **710.33** and a win rate of **70%** in various test scenarios.
- Successfully implemented and compared multiple adversarial search algorithms.

---
## How to Install and Run the Project

To set up and run this project, follow these steps:

1. **Clone the repository:**
2. **Install dependencies:**
3. **Run the project:**

You can run the Pacman game with various options as follows:

### 1. Interactive Control
To start the game and control Pacman using your keyboard, run:
```bash
python pacman.py
```

### 2. Interactive Control with Custom Layout
To play with a specific layout, use the `-l` option. For example:
```bash
python pacman.py -l trickyClassic
```

### 3. Run a Specific AI Algorithm
To run a specific AI algorithm, specify it using the `-p` option. For example:
```bash
python pacman.py -l mediumClassic -p ExpectimaxAgent
```

### 4. Set Depth for Game Tree
To run an algorithm with a specified depth in the game tree (the number of steps to look ahead), use the `-a` option. For example:
```bash
python pacman.py -l mediumClassic -p MinimaxAgent -a depth=3
```

### 5. Run with Evaluation Function
You can run an algorithm with a specified depth and an evaluation function. For example:
```bash
python pacman.py -l mediumClassic -p MinimaxAgent -a depth=3,evalFn=betterEvaluationFunction
```

### 6. Set Random Seed
To ensure reproducibility, you can set a random seed using the `-s` option. For example:
```bash
python pacman.py -l mediumClassic -p MinimaxAgent -a depth=3,evalFn=betterEvaluationFunction -s 22520000 
```

### 7. Adjust Simulation Time
You can use the `--frameTime` option to shorten the simulation time. For example:
```bash
python pacman.py -l mediumClassic -p MinimaxAgent -a depth=3,evalFn=betterEvaluationFunction --frameTime 0
```

### 8. Additional References

For more detailed usage instructions and options, please refer to the official [Pacman Project Documentation](https://inst.eecs.berkeley.edu/~cs188/fa19/project2/).

Happy gaming!

---
## How to Use the Project

- Launch the game by running the provided script.
- Use the arrow keys to control Pacman.
- Observe how the evaluation function influences Pacman's decision-making against ghost opponents.

---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

