import numpy as np
import matplotlib.pyplot as plt
import uuid
from enum import Enum, IntEnum

# TODO Overspending punishment
LOCATION_CHANGE_COST = 0.1

class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(cls.__class__, cls) \
                .__new__(cls, *args, **kwargs)
            cls._instance.init()
        return cls._instance


class Actions(IntEnum):
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

    # def __init__(self):
    #     pass

    def init(self):
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
        costs = LOCATION_CHANGE_COST  # costs/rewards of registering own location
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
                #state['orientation'] = agent.orientation

            DLT().transaction(agent.wallet, DLT.GENESIS, costs)
        return reward, costs, goal

    @staticmethod
    def delivery_contract(agent, state, action):
        reward = 0.0
        costs = 0.0  # costs/rewards picking/moving/dropping assets
        goal = False

        def reward_for_moving_closer_to_dest(old_pos, new_pos, dest):
            rew = 0.0
            dist_old = (old_pos['x'] - dest[0]) ** 2 + (old_pos['y'] - dest[1]) ** 2
            dist_new = (new_pos['x'] - dest[0]) ** 2 + (new_pos['y'] - dest[1]) ** 2
            if dist_old > dist_new:
                rew = 1.0
            return rew

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
                o.picked_up = True
                reward = 1
            else:
                # punishment for not picking up assets
                reward = -1
                goal = True

        elif action == Actions.DROP_OFF_ASSET:
            if state['loaded']:
                # reward for picking up asset
                # punish for attempt of picking up action in improper circumstances

                if agent.load.get_position()['x'] == agent.load.destination[0] and \
                        agent.load.get_position()['y'] == agent.load.destination[1]:
                    agent.load.delivered = True
                    reward = 1
                    goal = True
                agent.load.picked_up = False
                agent.set_load(None)
            else:
                # punishment for not picking up assets
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
                    reward += reward_for_moving_closer_to_dest(agent.get_position(), position, agent.load.destination)
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

                if not goal:
                    o = world[next_x, next_y]
                    if (o is not None) and (o is not agent):
                        reward = -1
                        goal = True

                    position = agent.get_position()
                    position['x'] += orientation[0]
                    position['y'] += orientation[1]
                    reward += reward_for_moving_closer_to_dest(agent.get_position(), position, agent.load.destination)
                    agent.load.set_position(position)

        elif action in [Actions.MOVE_BACKWARD, Actions.MOVE_FORWARD]:
            # reward for moving closer to delivery
            old_pos = state['position']
            new_pos = agent.get_position()
            assets = state['assets']
            for asset in assets:
                apos = asset['position']
                dist_old = (old_pos['x'] -  apos['x'])**2 + (old_pos['y'] - apos['y'])**2
                dist_new = (new_pos['x'] - apos['x'])**2 + (new_pos['y'] - apos['y'])**2
                if dist_old > dist_new:
                    reward = 1.0
                    break

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
        super(Obstacle, self).__init__(coordinates, 1, [0.5, 0.5, 0.5], '')


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
        state['picked'] = self.picked_up
        state['destination'] = self.destination
        state['delivered'] = self.delivered
        return state


class DeliveryRobot(Agent):
    id = 0
    type = 1

    def __init__(self, coordinates):
        self.load = None
        self.orientation = np.array([0, 1])

        for i in range(0,np.random.randint(10)):
            self.set_orientation(np.asarray(
                np.matrix([[0, -1], [1, 0]]) * np.matrix(self.orientation).T).T
                                  .flatten())

        super(DeliveryRobot, self).__init__(coordinates, 1, [100, 0, 00], f'drobot-{self.id}')
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

    def set_load(self, load: Asset):
        self.load = load

    def get_state(self):
        state = super().get_state()
        state['loaded'] = (self.load is not None)
        state['position'] = self.get_position()
        state['orientation'] = self.orientation
        state['destination'] = [-1, -1]
        if self.load is not None:
            state['destination'] = self.load.destination
        return state


class Environment:
    def __init__(self, envsize=(10, 10), horizon=2):
        self.horizon = horizon
        self.sizeX = envsize[0]
        self.sizeY = envsize[1]
        # self.reset()
        # plt.imshow(a,interpolation="nearest")

    def local_map_size(self):
        return (self.horizon + 1) * 2 + 1

    def add(self, agent: Agent):
        self.agents.append(agent)

    def reset(self):
        self.map = []
        self.agents = []
        state = self.renderEnv()
        return state

    def renderEnv(self, plot=False):
        self.map = np.array([None] * (self.sizeY * self.sizeX)).reshape((self.sizeX, self.sizeY))
        for agent in self.agents:
            if (agent.pos[0] >= 0) and (agent.pos[1] >= 0) and \
                    (agent.pos[0] < self.map.shape[1]) and \
                    (agent.pos[1] < self.map.shape[1]):
                self.map[agent.pos[0], agent.pos[1]] = agent;

        if plot:
            img = [a.intensity if a is not None else [0.0, 0.0, 0.0] for a in self.map.flatten(order='F')]
            img = np.array(img, dtype=np.float).reshape((self.sizeX, self.sizeY, 3))
            plt.imshow(img, interpolation="nearest")
        return self.map

    def localMap(self, agent, plot=False):
        self.renderEnv(False)
        horizont = self.horizon
        left = agent.pos[0] - horizont - 1
        oleft = 0
        if left < 0:
            oleft = -left
            left = 0

        right = agent.pos[0] + horizont + 2
        oright = 0
        if right >= self.sizeX:
            oright = right - self.sizeX
            right = self.sizeX

        top = agent.pos[1] - horizont - 1
        otop = 0
        if top < 0:
            otop = -top
            top = 0

        bottom = agent.pos[1] + horizont + 2
        obottom = 0
        if bottom >= self.sizeY:
            obottom = bottom - self.sizeY
            bottom = self.sizeY

        map = self.map[left:right, top:bottom]
        if oleft > 0 or oright > 0 or obottom > 0 or otop > 0:
            newmap = np.zeros(((2 * (horizont + 1) + 1), (2 * (horizont + 1) + 1)), dtype=map.dtype)
            newmap.fill(Obstacle([0, 0]))
            oright = -oright
            if oright == 0:
                oright = None
            obottom = -obottom
            if obottom == 0:
                obottom = None
            newmap[oleft:oright, otop:obottom] = map
            # print(oleft, oright, otop, obottom)
            map = newmap

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
        current_state = self.get_compound_state(agent)

        total_reward = 0.0
        cashflow = 0.0
        completed = False

        # fetch relevant contracts and update state
        for contract in dlt.locate_contracts(agent):
            (reward, costs, goal) = contract(agent, current_state, proved_action)
            cashflow += costs
            total_reward += reward
            completed = completed or goal

        total_reward += dlt.finanсial_objective(dlt.balance(agent.wallet), cashflow)

        new_state = self.get_compound_state(agent)
        return new_state, total_reward, completed

    def get_compound_state(self, agent):
        # compile state of the world around/near us
        new_state = agent.get_state()
        new_state['id'] = agent.name
        new_state['assets'] = \
            map(lambda x: x.get_state(),
                sorted(
                    filter(lambda x: isinstance(x, Asset) and x is not agent.load, self.agents),
                    key=lambda a: (a.pos[0] - agent.pos[0]) ** 2 + (
                            a.pos[1] - agent.pos[1]) ** 2))  # show boxes around us
        new_state['robots'] = \
            [r.get_state() for r in sorted(
                filter(lambda x: isinstance(x, DeliveryRobot) and x != agent, self.agents),
                key=lambda a: (a.pos[0] - agent.pos[0]) ** 2 + (a.pos[1] - agent.pos[1]) ** 2)]
        new_state['world'] = self.localMap(agent)
        return new_state
