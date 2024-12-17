# Load the required package
library(neuralnet)

# Set the seed for reproducibility
set.seed(1234567890)

# Generate data
Var <- runif(500, 0, 10)             # Random 500 points in [0, 10]
mydata <- data.frame(Var, Sin=sin(Var)) # Create data frame with sine values

# Split data into training and testing sets
tr <- mydata[1:25, ]   # First 25 points as training data
te <- mydata[26:500, ] # Remaining points as test data

# Random initialization of weights
winit <- runif(1, -1, 1)  # Random initial weights in [-1, 1]

# Train a neural network with one hidden layer and 10 hidden units
nn <- neuralnet(Sin ~ Var,            # Formula: Predict Sin from Var
                data = tr,            # Training data
                hidden = 10,          # One hidden layer with 10 units
                threshold = 0.0001,     # Convergence threshold
                startweights = rep(winit, 31), # Initialize weights
                stepmax = 1e6)        # Maximum training steps

# Plot training and test data
plot(tr, cex=2, main="Training Data (Black), Test Data (Blue), Predictions (Red)")
points(te, col = "blue", cex=1)                   # Test data points in blue
points(te[,1], predict(nn, te), col="red", cex=1) # NN predictions in red
