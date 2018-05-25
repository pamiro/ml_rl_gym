import numpy as np
import matplotlib.pyplot as plt
import uuid
from enum import Enum


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(cls.__class__, cls) \
                .__new__(cls, *args, **kwargs)
        return cls._instance


class Actions(Enum):
    NONE = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    PICK_UP_ASSET = 5
    DROP_OFF_ASSET = 6


ActionMoves = [Actions.MOVE_FORWARD, Actions.MOVE_BACKWARD,
               Actions.TURN_LEFT, Actions.TURN_RIGHT]


class DLT(Singleton):
    GENESIS = 0

    def __init__(self):
        self.accounts = {'': 0}  # genesis
        self.transactions = []
        self.contracts = []

    def make_account(self):
        acc = uuid.uuid4()
        return acc

    def balance(self, acc):
        if not (acc in self.accounts.keys()):
            self.accounts[acc] = 0
        return self.accounts[acc]

    def transaction(self, sender, receiver, amount):
        self.transactions.append([sender, receiver, amount])
        s = self.balance(sender)
        self.accounts[sender] = s - amount
        r = self.balance(receiver)
        self.accounts[receiver] = r + amount

    @staticmethod
    def finanсial_objective(balance, cashflow):
        return 0

    @staticmethod
    def location_contract(agent, state, action):
        """localisation contract, registering own location+orientation for collision avoidance"""
        reward = 0.0
        costs = 0.1  # costs/rewards of registering own location
        goal = False

        position = state['position']
        world = state['world']
        [[local_x, local_y]] = np.argwhere(world == agent)

        change_x = 0
        change_y = 0
        if action == Actions.MOVE_FORWARD:
            [change_x, change_y] = agent.orientation
        elif action == Actions.MOVE_BACKWARD:
            [change_x, change_y] = -agent.orientation
        elif action == Actions.TURN_LEFT:
            agent.set_orientation(np.asarray(
                np.matrix([[0, -1], [1, 0]]) * np.matrix(agent.orientation).T).T
                                  .flatten())
        elif action == Actions.TURN_RIGHT:
            agent.set_orientation(np.asarray(
                np.matrix([[0, 1], [-1, 0]]) * np.matrix(agent.orientation).T).T
                                  .flatten())

        if action in ActionMoves:
            position['x'] += change_x
            position['y'] += change_y
            local_x += change_x
            local_y += change_y

            # do not cross borders of the world
            if local_x < 0 or local_y < 0 or \
                    local_x >= world.shape[0] or local_y >= world.shape[1]:
                reward = -1
                goal = True
            else:
                # collision
                if action in [Actions.MOVE_BACKWARD, Actions.MOVE_FORWARD]:
                    if (world[local_x, local_y] is not None) \
                            and (agent.load is not world[local_x, local_y]):
                        reward = -1
                        goal = True

            if not goal:
                agent.set_position(position)
                state['orientation'] = agent.orientation

            DLT().transaction(agent.wallet, DLT.GENESIS, costs)
        return reward, costs, goal

    @staticmethod
    def delivery_contract(agent, state, action):
        reward = 0.0
        costs = 0.0  # costs/rewards picking/moving/dropping assets
        goal = False

        world = state['world']
        [[local_x, local_y]] = np.argwhere(world == agent)
        orientation = agent.orientation

        [next_x, next_y] = [local_x, local_y] + orientation;
        if action == Actions.PICK_UP_ASSET and (not state['loaded']):
            # reward for picking up asset
            # punish for attempt of picking up action in improper circumstances
            o = world[next_x, next_y]
            if isinstance(o, Asset):
                agent.set_load(o)
            else:
                # punishment for not picking up assets
                reward = -1
                goal = True

        elif action == Actions.DROP_OFF_ASSET and state['loaded']:
            # reward for picking up asset
            # punish for attempt of picking up action in improper circumstances
            o = world[next_x, next_y]
            if o is None:
                agent.set_load(None)
            else:
                reward = -1
                goal = True

        if state['loaded']:
            if action in [Actions.TURN_LEFT, Actions.TURN_RIGHT]:

                # collision
                if next_x < 0 or next_y < 0 or \
                        next_x >= world.shape[0] or next_y >= world.shape[1]:
                    reward = -1
                    goal = True
                elif (world[next_x, next_y] is not None) \
                        and (world[next_x, next_y] is not agent):
                    reward = -1
                    goal = True

                if not goal:
                    change = np.array([next_x, next_y]) - np.array([local_x, local_y])
                    position = agent.get_position();
                    position['x'] += change[0]
                    position['y'] += change[1]
                    agent.load.set_position(position)

            if action in [Actions.MOVE_BACKWARD, Actions.MOVE_FORWARD]:
                if action == Actions.MOVE_FORWARD:
                    [next_x, next_y] = [next_x, next_y] + orientation;
                elif action == Actions.MOVE_BACKWARD:
                    [next_x, next_y] = [next_x, next_y] - orientation;

                if next_x < 0 or next_y < 0 or \
                        next_x >= world.shape[0] or next_y >= world.shape[1]:
                    reward = -1
                    goal = True
                    print

                if not goal:
                    o = world[next_x, next_y]
                    if (o is not None) and (o is not agent):
                        reward = -1
                        goal = True

                    position = agent.get_position()
                    position['x'] += orientation[0]
                    position['y'] += orientation[1]
                    agent.load.set_position(position)

        DLT().transaction(agent.wallet, DLT.GENESIS, costs)
        return reward, costs, goal

    def locate_contracts(self, agent):
        return iter([
            self.location_contract,
            self.delivery_contract
        ])


class Agent:
    type = -1

    def __init__(self, coordinates, size, intensity, name):
        self.pos = coordinates
        self.name = name
        self.intensity = intensity
        self.wallet = DLT().make_account()

    def set_position(self, position):
        self.pos = [position['x'], position['y']]

    def get_position(self):
        return {'x': self.pos[0], 'y': self.pos[1]}

    def get_state(self):
        return {
            'type': self.type,
            'position': self.get_position(),
            'balance': DLT().balance(self.wallet)
        }


class Obstacle(Agent):
    def __init__(self, coordinates):
        super(Obstacle, self).__init__(coordinates, 1, [100, 100, 100], '')


# charge for pass
class Asset(Agent):
    id = 0
    type = 2

    def __init__(self, coordinates, destination):
        self.picked_up = False
        self.delivered = False
        self.destination = destination
        super(Asset, self).__init__(coordinates, 1, [100, 0, 100], f'package {Asset.id}')
        Asset.id = Asset.id + 1

    def get_state(self):
        state = super().get_state()
        state['dest'] = (self.load is not None)
        return state


class DeliveryRobot(Agent):
    id = 0
    type = 1

    def __init__(self, coordinates):
        self.load = None
        self.orientation = np.array([0, 1])
        super(DeliveryRobot, self).__init__(coordinates, 1, [100, 0, 00], f'drobot {DeliveryRobot.id}')
        DeliveryRobot.id = DeliveryRobot.id + 1

    def set_orientation(self, orientation):
        self.orientation = orientation

    def get_orientation(self):
        return self.orientation

    def get_orientation_angle(self):
        if np.allclose(self.orientation, [1, 0]):
            return 0
        if np.allclose(self.orientation, [0, 1]):
            return 90
        if np.allclose(self.orientation, [-1, 0]):
            return 180
        if np.allclose(self.orientation, [0, -1]):
            return 270

    def set_load(self, load):
        self.load = load

    def get_state(self):
        state = super().get_state()
        state['loaded'] = (self.load is not None)
        state['orientation'] = self.orientation
        return state


class Environment:
    def __init__(self, envsize=(10, 10), horizont=2):
        self.horizont = horizont
        self.sizeX = envsize[0]
        self.sizeY = envsize[1]
        self.reset()
        # plt.imshow(a,interpolation="nearest")

    def add(self, agent: Agent):
        self.agents.append(agent)

    def reset(self):
        self.map = []
        self.agents = []
        state = self.renderEnv()
        return state

    def renderEnv(self, plot=True):
        self.map = np.array([None] * (self.sizeY * self.sizeX)).reshape((self.sizeX, self.sizeY))
        for agent in self.agents:
            self.map[agent.pos[0], agent.pos[1]] = agent;

        if plot:
            img = [a.intensity if a is not None else [0.0, 0.0, 0.0] for a in self.map.flatten(order='F')]
            img = np.array(img, dtype=np.float).reshape((self.sizeX, self.sizeY, 3))
            plt.imshow(img, interpolation="nearest")
        return self.map

    def localMap(self, agent, plot=False):
        self.renderEnv(False)
        horizont = self.horizont
        left = agent.pos[0] - horizont - 1
        if left < 0:
            left = 0
        right = agent.pos[0] + horizont + 1
        if right >= self.sizeX:
            right = self.sizeX
        top = agent.pos[1] - horizont - 1
        if top < 0:
            top = 0
        bottom = agent.pos[1] + horizont + 1
        if bottom >= self.sizeY:
            bottom = self.sizeY

        map = self.map[left:right, top:bottom]

        if plot:
            img = [a.intensity if a is not None else [0.0, 0.0, 0.0] for a in map.flatten(order='F')]
            img = np.array(img, dtype=np.float).reshape((bottom - top, right - left, 3))
            plt.imshow(img, interpolation="nearest")

        return map

    def step(self, agent, proved_action):
        # the end goal reward for emergent the collaborative behaviour

        if not isinstance(agent, DeliveryRobot):
            raise RuntimeError('Unsupported')

        dlt = DLT()

        # compile state of the world around/near us
        new_state = agent.get_state()
        new_state['assets'] = [self.agents]  # show boxes around us
        new_state['agents'] = [self.agents]  # show agents around
        new_state['world'] = self.localMap(agent)

        total_reward = 0.0
        cashflow = 0.0
        completed = False

        # fetch relevant contracts and update state
        for contract in dlt.locate_contracts(agent):
            (reward, costs, goal) = contract(agent, new_state, proved_action)
            cashflow += costs
            total_reward += reward
            completed = completed or goal

        total_reward += dlt.finanсial_objective(dlt.balance(agent.wallet), cashflow)

        # compile state of the world around/near us
        new_state = agent.get_state()
        new_state['assets'] = [self.agents]  # show boxes around us
        new_state['agents'] = [self.agents]  # show agents around
        new_state['world'] = self.localMap(agent)
        return new_state, total_reward, completed

# if __name__ == '__main__':
#     env = Environment((10,10))
#     # plt.ion()
#     # plt.show()
#
#     plt.subplot(211)
#     env.add(Obstacle([2, 2]))
#     env.add(Asset([1, 2], [10, 10]))
#
#     env.add(Asset([0, 0], [9, 9]))
#     robot = DeliveryRobot([1, 1])
#     env.add(robot)
#
#     # print(env.step(robot, Actions.TURN_RIGHT))
#     # print(env.step(robot, Actions.TURN_RIGHT))
#     # print(env.step(robot, Actions.TURN_RIGHT))
#     # print(env.step(robot, Actions.TURN_RIGHT))
#
#     # print(env.step(robot, Actions.MOVE_FORWARD))
#     print(env.step(robot, Actions.PICK_UP_ASSET))
#     print(env.step(robot, Actions.MOVE_FORWARD))
#     print(env.step(robot, Actions.MOVE_FORWARD))
#     print(env.step(robot, Actions.TURN_RIGHT))
#
#     env.renderEnv()
#
#     plt.subplot(212)
#     env.localMap(robot, plot=True)
#     # print(new_state)
#     # print(total_reward)
#     # print(completed)
#     plt.show()
