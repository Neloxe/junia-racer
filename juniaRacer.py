import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class JuniaRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super(JuniaRacerEnv, self).__init__()

        # Initialize Pygame
        pygame.init()
        self.size = self.width, self.height = 1400, 800
        self.gameDisplay = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()

        # Load images
        self.white_small_car = pygame.image.load("Images/Sprites/white_small.png")
        self.bg = pygame.image.load("bg73.png")
        self.bg4 = pygame.image.load("bg43.png")

        # Initialize car object
        self.car = Car([6, 6, 4], self.bg4)

        # Define the action space and observation space
        self.action_space = spaces.Discrete(
            4
        )  # Actions: LEFT5, RIGHT5, ACCELERATE, BRAKE
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(6,),
            dtype=np.float32,  # Observation: 6-dimensional
        )

        self.prev_angle = self.car.angle  # Store previous angle for U-turn detection
        self.start_time = None  # Initialize start time for the timer
        self.elapsed_time = 0  # Initialize elapsed time
        self.best_lap_time = 99999  # Initialize best lap time

        # Define start/end line coordinates
        self.start_line = (70, 470, 210, 480)  # Example coordinates (x1, y1, x2, y2)
        self.prev_car_position = (self.car.x, self.car.y)  # Store previous car position

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation and info.
        """
        super().reset(seed=seed)  # Properly handle Gym seed for reproducibility
        self.car.resetPosition()
        self.car.update()
        self.start_time = pygame.time.get_ticks()  # Start the timer
        self.elapsed_time = 0  # Reset elapsed time
        self.prev_car_position = (self.car.x, self.car.y)  # Reset previous car position

        # Return both observation and info as required by Gymnasium
        return self._get_obs(), {}  # Ensures a tuple (obs, info) is returned

    def step(self, action):
        """
        Takes a step in the environment based on the given action.
        """
        # Perform action based on discrete space
        if action == 0:  # LEFT5
            self.car.rotate(-5)
        elif action == 1:  # RIGHT5
            self.car.rotate(5)
        elif action == 2:  # ACCELERATE
            self.car.set_accel(0.2)
        elif action == 3:  # BRAKE
            self.car.set_accel(-0.2)
        else:  # NOTHING
            self.car.set_accel(0)

        self.car.update()
        done = self.car.collision()

        # Calculate reward
        reward = 0

        # Collision penalty
        if done:
            reward = -2000

        # # Check if the car made a U-turn or is going the wrong way
        # angle_diff = abs(self.car.angle - self.prev_angle)
        # if angle_diff > 180:  # U-turn detected, reward for not turning around
        #     reward -= 50  # Apply penalty for making a sharp U-turn

        # Encourage forward motion (positive velocity) and penalize negative velocity
        reward += self.car.velocity * 2

        # Penalize for getting too close to the walls
        if min(self.car.d1, self.car.d2, self.car.d3, self.car.d4, self.car.d5) < 10:
            reward -= 100

        # Update previous angle for next step
        self.prev_angle = self.car.angle

        # Calculate elapsed time for the turn
        current_time = pygame.time.get_ticks()
        self.elapsed_time = (current_time - self.start_time) / 1000.0

        # Check if the car crosses the start/end line
        if self._crossed_start_line():
            self.start_time = current_time  # Reset the timer for the next lap
            if (
                self.best_lap_time is None
                or self.elapsed_time < self.best_lap_time
                and self.elapsed_time > 4
            ):
                self.best_lap_time = self.elapsed_time
                # reward += 50  # Reward for completing a lap
            self.elapsed_time = 0  # Reset elapsed time for the next lap

        # Update previous car position
        self.prev_car_position = (self.car.x, self.car.y)

        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        """
        Renders the current state of the environment.
        """
        self.gameDisplay.blit(self.bg, (0, 0))
        self.car.draw(self.gameDisplay)

        # Draw the start/end line
        pygame.draw.line(
            self.gameDisplay,
            (0, 255, 0),
            (self.start_line[0], self.start_line[1]),
            (self.start_line[2], self.start_line[3]),
            5,
        )

        # Display the timer
        font = pygame.font.Font(None, 36)
        text = font.render(f"Time: {self.elapsed_time:.2f} s", True, (255, 255, 255))
        self.gameDisplay.blit(text, (10, 10))

        # Display the best lap time
        lap_text = font.render(
            f"Best Lap: {self.best_lap_time:.2f} s", True, (255, 255, 255)
        )
        self.gameDisplay.blit(lap_text, (10, 50))

        pygame.display.update()

        pygame.time.Clock().tick(120)

    def _get_obs(self):
        """
        Retrieves the current observation from the environment, normalized to [0.0, 255.0].
        """
        # Normalize distances and angle to fit the observation space bounds
        return np.array(
            [
                self._normalize_value(self.car.d1, 0, 1000, 0, 255),
                self._normalize_value(self.car.d2, 0, 1000, 0, 255),
                self._normalize_value(self.car.d3, 0, 1000, 0, 255),
                self._normalize_value(self.car.d4, 0, 1000, 0, 255),
                self._normalize_value(self.car.d5, 0, 1000, 0, 255),
                self._normalize_value(self.car.velocity, 0, 10, 0, 255),
            ],
            dtype=np.float32,
        )

    def _normalize_value(
        self, value, original_min, original_max, target_min, target_max
    ):
        """
        Normalizes a value from an original range to a target range.
        """
        return ((value - original_min) / (original_max - original_min)) * (
            target_max - target_min
        ) + target_min

    def _crossed_start_line(self):
        """
        Checks if the car has crossed the start/end line.
        """
        x1, y1, x2, y2 = self.start_line
        car_x, car_y = self.car.x, self.car.y

        if car_x >= x1 and car_x <= x2:
            if car_y >= y1 and car_y <= y2:
                return True
        return False

    def close(self):
        """
        Properly closes the environment and quits Pygame.
        """
        pygame.quit()


class Car:
    def __init__(self, sizes, bg4):
        self.bg4 = bg4
        self.sizes = sizes
        self.x = 120
        self.y = 480
        self.velocity = 0
        self.acceleration = 0
        self.angle = 180
        self.car_image = pygame.image.load("Images/Sprites/white_small.png")

        # Sensors and collision detection variables
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0
        self.d5 = 0

    def set_accel(self, accel):
        self.acceleration = accel

    def rotate(self, rot):
        self.angle += rot
        self.angle %= 360

    def update(self):
        """
        Updates the car's position and sensors.
        """
        # Update velocity
        self.velocity += self.acceleration
        self.velocity = max(0, min(self.velocity, 10))  # Clamp velocity to [0, 10]

        # Update position
        self.x, self.y = move((self.x, self.y), self.angle, self.velocity)

        # Update sensors
        self.d1 = self.get_distance(self.angle)
        self.d2 = self.get_distance(self.angle + 45)
        self.d3 = self.get_distance(self.angle - 45)
        self.d4 = self.get_distance(self.angle + 90)
        self.d5 = self.get_distance(self.angle - 90)

    def get_distance(self, angle):
        """
        Calculates the distance to the nearest obstacle at a given angle.
        """
        sensor_x, sensor_y = move((self.x, self.y), angle, 10)
        while self.bg4.get_at((int(sensor_x), int(sensor_y))).a != 0:
            sensor_x, sensor_y = move((sensor_x, sensor_y), angle, 10)
        return calculateDistance(self.x, self.y, sensor_x, sensor_y)

    def draw(self, display):
        """
        Draws the car and its sensors on the display.
        """
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle - 180)
        rect_rotated_image = rotated_image.get_rect(center=(self.x, self.y))
        display.blit(rotated_image, rect_rotated_image)

    def collision(self):
        """
        Detects collision with the background.
        """
        return self.bg4.get_at((int(self.x), int(self.y))).a == 0

    def resetPosition(self):
        """
        Resets the car to its initial position and angle.
        """
        self.x = 120
        self.y = 480
        self.angle = 180
        self.velocity = 0


def calculateDistance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def move(point, angle, unit):
    """
    Moves a point in a given direction by a specified unit.
    """
    x, y = point
    rad = math.radians(-angle % 360)
    x += unit * math.sin(rad)
    y += unit * math.cos(rad)
    return x, y
