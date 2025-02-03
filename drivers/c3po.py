from random import *

# Init your variables here

# Put your name Here
name = "C3PO"  # FirstName LastName

BRAKE = 0
ACCELERATE = 1
LEFT5 = 2
RIGHT5 = 3
NOTHING = 4


# This function will be called at the beginning
# it will allow you to laod your model for example
def setup():
    global name
    print(name, "driver setup...")

    # put your initialization code here
    # ....
    return 0


# C3PO strategy : Return the first element in the available cells.


# c1, c2, c3, c4, c5 are five 2D points where the car could collided, updated in every frame
# d1, d2, d3, d4, d5 are distances from the car to those points, updated every frame too and used as the input for the NN
def drive(d1, d2, d3, d4, d5, velocity, acceleration):

    # d1  front
    # d2  mid left
    # d3  mid right
    # d4  left
    # d5  right
    # velocity : corrent velocity of the car

    # List of possible actions to return
    # BRAKE
    # ACCELERATE
    # LEFT5
    # RIGHT5

    # PUT YOUR CODE HERE

    return NOTHING

    if abs(d4 - d5) < 8:
        return ACCELERATE
    elif d4 < d5:
        return LEFT5
    else:
        return RIGHT5
