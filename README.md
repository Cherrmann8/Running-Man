# Running-Man

Using TensorFlow 1.0, I've implemented Q-Learning and temporal difference algorithms to train an actor in a grid-world environment. The actor can move left or right but must stay on the bottom row of the grid. Each tick, "rocks" fall down the grid row by row while new "rocks" appear randomly on the top row. The game continues until the actor gets hit by one of the rocks. You can change the grid size and how often a new rock falls. I've included some graphs in this repo to show performance of the Q-learning algorithm only. unfortunately, the first couple graphs are not labeled. The x axis is time (each point is the avg of 100 games). The blue line is avg loss, the green is avg time alive, and the red is avg reward.

