import numpy as np
import threading
import serial
import random
import time

from capture_frame import find_sticker_and_save_coordinates
from ArmGym import ArmGym

lock = threading.Lock()

def update_positions():
    global red_positions, blue_positions
    while True:
        lock.acquire()
        try:
            red_positions, blue_positions = find_sticker_and_save_coordinates()
        finally:
            lock.release()
        time.sleep(0.5)

class QLAgent:
    def __init__(self, low_state, high_state, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_size = tuple([(high - low) - 1 for low, high in zip(low_state, high_state)])

        self.q_table = np.zeros(self.state_size + (action_size,))

    def state_to_index(self, state, low_state, high_state):
        index = [int((state[i] - low_state[i])) for i in range(len(state))]
        return tuple(index)

    def choose_action(self, state, low_state, high_state):
        state_index = self.state_to_index(state, low_state, high_state)

        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_index])

    def learn(self, state, action, reward, next_state, done, low_state, high_state):
        state_index = self.state_to_index(state, low_state, high_state)
        next_state_index = self.state_to_index(next_state, low_state, high_state)
        predict = self.q_table[state_index + (action,)]
        target = reward

        if not done:
            target += self.gamma * np.max(self.q_table[next_state_index])

        self.q_table[state_index + (action,)] += self.alpha * (target - predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


ser = serial.Serial('COM3', 9600)

# 아두이노로 각도 보내기
def send_angles_to_arduino(angles):
    ser.write(','.join(str(angle) for angle in angles) + '\n')


# red_positions = [(100, 100), (150, 150), (200, 200)]
# blue_positions = [(150, 100), (200, 150), (250, 200)]

# 환경 생성
def q_learning(red_positions, blue_positions):
    env = ArmGym(red_positions, blue_positions)

    num_episodes = 1000
    num_steps = 50

    low_state = [0, 0, 0, 0, 0, 0, 0, 0]
    high_state = [640, 480, 640, 480, 150, 150, 150, 150]
    agent = QLAgent(low_state, high_state, env.action_space.shape[0])

    for episode in range(num_episodes):
        lock.acquire()
        try:
            red_positions_copy = list(red_positions)
            blue_positions_copy = list(blue_positions)
        finally:
            lock.release()
        
        # state 초기화
        state = env.reset()
        # total_reward 초기화
        total_reward = 0

        for step in range(num_steps):
            action = agent.choose_action(state, env.observation_space.low, env.observation_space.high)
            next_state, reward, done, _ = env.step(env.action_space.sample())
            agent.learn(state, action, reward, next_state, done, env.observation_space.low, env.observation_space.high)
            send_angles_to_arduino(action[:4]) #서보모터 각도 값을 아두이노로 전송
            total_reward += reward
            state = next_state

            # 현재 상태, 행동 및 보상 출력
            print(f'Step: {step}, State: {state}, Action: {action}, Reward: {reward}')

        if (episode+1) % 100 == 0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward}")

position_thread = threading.Thread(target=update_positions)
position_thread.start()
q_learning_thread = threading.Thread(target=q_learning, args=(red_positions, blue_positions))
q_learning_thread.start()