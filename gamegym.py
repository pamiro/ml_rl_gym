import gc
from collections import deque
from functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.graphics.context_instructions import PushMatrix, Rotate, PopMatrix, Scale
from kivy.graphics.vertex_instructions import Line
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
import numpy as np

from world import DeliveryRobot, Agent, Actions, Obstacle, Asset, DLT
from world import Environment
import learning
import tensorflow as tf
import matplotlib.pyplot as plt

BASIS_INCOME = 1.0

A_KEY = ord('a')
R_KEY = ord('r')
D_KEY = ord('d')
P_KEY = ord('p')
I_KEY = ord('i')
S_KEY = ord('s')
Q_KEY = ord('q')
T_KEY = ord('t')

LEFT_KEY = 276
RIGHT_KEY = 275
BACKWARD_KEY = 274
FORWARD_KEY = 273

NASSETS = 3
NOBSTACLES = 5
NDELIVERY_AGENTS = 2


class WidgetWrapper(Image):
    def __init__(self, *args, **kwargs):
        self.agent = kwargs['agent']
        self.last_state = None
        self.selected = False
        del kwargs['agent']
        super(WidgetWrapper, self).__init__(*args, **kwargs)

        self.bind(parent=self.on_update)
        self.bind(size=self.on_update)

    def on_update(self, *args, **kwargs):
        if self.parent is None:
            return

        cell_x = self.parent.size[0] / self.parent.sizeX
        cell_y = self.parent.size[1] / self.parent.sizeY

        agent_pos = self.agent.get_position();

        (ppos_x, ppos_y) = self.parent.pos;
        self.pos = (int(agent_pos['x'] * cell_x) + ppos_x,
                    int(agent_pos['y'] * cell_y) + ppos_y)

        self.size_hint = (1 / self.parent.sizeX, 1 / self.parent.sizeY)
        self.source = 'images/icons8-truck-50.png'

        self.canvas.before.clear()
        self.canvas.after.clear()

        with self.canvas.before:
            if self.selected:
                self.draw_selected()

            PushMatrix()
            if self.agent.get_orientation_angle() == 180:
                pass
            Rotate(angle=self.agent.get_orientation_angle(), origin=self.center)

        with self.canvas.after:
            PopMatrix()

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            self.selected = True
        else:
            self.selected = False
        self.on_update()

    def on_touch_move(self, touch):
        pass

    def on_touch_up(self, touch):
        pass

    def draw_selected(self):
        # with self.canvas:
        Color(0, 1, 0, 1)
        Rectangle(size=self.size,
                  pos=self.pos)


class GenericWrapper(Image):
    def __init__(self, *args, **kwargs):
        self.agent = kwargs['agent']
        del kwargs['agent']
        super(GenericWrapper, self).__init__(*args, **kwargs)
        self.bind(parent=self.on_update,
                  pos=self.on_update,
                  size=self.on_update)

    def on_update(self, *args):
        if self.parent is None:
            return

        self.size_hint = (1 / self.parent.sizeX, 1 / self.parent.sizeY)
        cell_x = self.parent.size[0] / self.parent.sizeX
        cell_y = self.parent.size[1] / self.parent.sizeY

        agent_pos = self.agent.get_position();
        (ppos_x, ppos_y) = self.parent.pos
        self.pos = (int(agent_pos['x'] * cell_x) + ppos_x,
                    int(agent_pos['y'] * cell_y) + ppos_y)

        self.canvas.before.clear()
        with self.canvas.before:
            # with self.canvas:
            Color(*self.agent.intensity)
            Rectangle(size=self.size,
                      pos=self.pos)


class EnvironmentScreen(FloatLayout, Environment):
    def __init__(self, *args, **kwargs):
        super(EnvironmentScreen, self).__init__(*args, **kwargs)

        self.bind(parent=self.on_update)
        self.bind(size=self.on_update)
        self.bind(pos=self.on_update)

    def on_update(self, *args, **kwargs):

        cellX = self.size[0] / self.sizeX
        cellY = self.size[1] / self.sizeY

        self.canvas.before.clear()

        with self.canvas.before:
            Color(0, 1, 1, 0.1)
            Rectangle(pos=self.pos, size=self.size)
            self.render_grid(cellX, cellY)

        self._trigger_layout()

    def render_grid(self, cellX, cellY):
        xL = 0
        yL = 0
        while xL < self.size[0]:
            Line(points=[self.pos[0] + xL, self.pos[1], self.pos[0] + xL, self.pos[1] + self.size[1]], width=1)
            xL += cellX
        while yL < self.size[1]:
            Line(points=[self.pos[0], self.pos[1] + yL, self.pos[0] + self.size[0], self.pos[1] + yL], width=1)
            yL += cellY

    def on_touch_down(self, touch):
        for child in self.children:
            child.selected = False
        super(EnvironmentScreen, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        super(EnvironmentScreen, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        super(EnvironmentScreen, self).on_touch_up(touch)

    def add(self, agent: Agent):
        super(EnvironmentScreen, self).add(agent)
        if isinstance(agent, DeliveryRobot):
            self.add_widget(WidgetWrapper(agent=agent))
            return

        if isinstance(agent, Obstacle):
            self.add_widget(GenericWrapper(agent=agent, size=[0, 0]))
            return

        if isinstance(agent, Asset):
            self.add_widget(GenericWrapper(agent=agent, source='images/icons8-cardboard-box-50.png'))
            return
        raise RuntimeError("unknown type")

    def reset(self):
        while (len(self.children) > 0):
            self.remove_widget(self.children[0])
        super(EnvironmentScreen, self).reset()


class RobotViewWidget(Widget):
    def __init__(self, **kwargs):
        super(RobotViewWidget, self).__init__(**kwargs)
        self.bind(parent=self.on_update)
        self.bind(size=self.on_update)
        self.bind(pos=self.on_update)

    def on_update(self, *args, **kwargs):
        self.canvas.clear()

    def draw(self, map, horizon, robot):
        self.canvas.clear()
        ms = (2 * (horizon + 1) + 1)
        cell_s = min(self.size) / ms
        offs_x = offs_y = 0
        # if map.shape[0] < min(self.size):
        offs_x = (ms - map.shape[0]) * cell_s / 2
        # if map.shape[1] < min(self.size):
        offs_y = (ms - map.shape[1]) * cell_s / 2
        with self.canvas:
            Color(0, 0, 0, 1.0)
            Rectangle(pos=(self.pos[0] + offs_x, self.pos[1] + offs_y),
                      size=(map.shape[0] * cell_s, map.shape[1] * cell_s))
            for x in range(map.shape[0]):
                for y in range(map.shape[1]):
                    if map[x, y] is not None:
                        if map[x, y] == robot:
                            Color(0, 1, 0)
                        else:
                            Color(*map[x, y].intensity)
                        Rectangle(pos=(self.pos[0] + x * cell_s + offs_x, self.pos[1] + y * cell_s + offs_y),
                                  size=(cell_s, cell_s))


class Visualization:
    # reward
    # loss
    # actions distributions

    MAX_M = 100

    def __init__(self):
        self.rq_cnt = 0
        self.rq = deque(np.zeros(self.MAX_M), maxlen=self.MAX_M)
        self.tq = deque(np.zeros(self.MAX_M), maxlen=self.MAX_M)
        self.hist = {}
        self.build()

    def add_reward(self, r):
        self.rq.append(r)
        self.rq_cnt += 1

    def add_train_loss(self, l):
        self.tq.append(l)

    def add_action(self, a):
        if not int(a) in self.hist.keys():
            self.hist[int(a)] = 1
        else:
            self.hist[int(a)] += 1

    def build(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        self.tq_line, = self.ax2.plot(self.tq)
        # self.ax1.title = 'rewards'
        # self.ax2.title = 'loss training'
        plt.subplots_adjust(hspace=1.0)

    def update(self):
        self.ax1.clear()
        self.ax3.set_title('reward')
        self.ax1.plot(np.arange(self.rq_cnt-100, self.rq_cnt), self.rq)

        self.ax3.set_title('loss train.')
        self.tq_line.set_ydata(self.tq)
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.ax3.clear()
        self.ax3.set_title('actions')
        self.ax3.bar(self.hist.keys(), self.hist.values())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class GameGymApp(App):
    environment = EnvironmentScreen(envsize=[10, 10]);
    robot_local_world = RobotViewWidget(size_hint=(1, 1))

    left_panel = Builder.load_string('''
StackLayout:
    orientation: 'tb-lr'
    padding: [5, 5, 5, 5]
    Label:
        id: state
        color: [0,0,0,1]
        text: 'Status/Wallet'
        markup: True
        size_hint: (1.0, 0.6)
        size: self.texture_size
        text_size: self.size
        halign: 'left'
        valign: 'top'
    Label:
        id: second_line
        color: [0,0,0,1]
        markup: True
        size_hint: (1.0, 0.4)
        size: self.texture_size
        text_size: self.size
        halign: 'left'
        valign: 'top'
    ''')

    help = """
HELP
click on robot for control
left/right - rotate robot
forward/backwards - move robot
p - pickup a box
d - drop a box
r - reset
a - activate
i - deposit universal basis income
s - simulation/training start
q - simulation/training abort
"""

    def build(self):
        Window.size = (800, 800)
        Window.clearcolor = (1, 1, 1, 1)
        root = BoxLayout(orientation='vertical')
        root.add_widget(self.environment)
        root.add_widget(self.robot_local_world)
        Window.bind(on_key_down=self.key_action)
        Window.bind(on_touch_down=self.on_touch_down)
        self.environment.reset()

        super_root = BoxLayout(orientation='horizontal')
        super_root.add_widget(self.left_panel)
        super_root.add_widget(root)

        self.set_status_text('')

        return super_root

    def set_status_text(self, text):
        self.left_panel.ids.second_line.text = text + self.help

    def key_action(self, key, scancode=None, codepoint=None, modifier=None, *args):
        print("got a key event: %s" % scancode)

        if scancode == S_KEY:
            self.start_sim()

        if scancode == Q_KEY:
            if self.sim is not None:
                self.sim.cancel()

        if scancode == T_KEY:
            pass

        if scancode == R_KEY:
            self.environment.reset()
            self.set_status_text('reset')

        # activate
        if scancode == A_KEY:
            self.activate_random()

        if len(self.environment.children) == 0:
            return

        robot = None
        for child in self.environment.children:
            if isinstance(child, WidgetWrapper) and child.selected:
                robot = child.agent
                break

        current_action = None
        if scancode == FORWARD_KEY:
            current_action = Actions.MOVE_FORWARD

        if scancode == BACKWARD_KEY:
            current_action = Actions.MOVE_BACKWARD

        if scancode == RIGHT_KEY:
            current_action = Actions.TURN_RIGHT

        if scancode == LEFT_KEY:
            current_action = Actions.TURN_LEFT

        if scancode == P_KEY:
            current_action = Actions.PICK_UP_ASSET

        if scancode == D_KEY:
            current_action = Actions.DROP_OFF_ASSET

        if scancode == I_KEY:
            self.deposit_basis_income()

        if current_action is not None and robot is not None:
            if (child.last_state is None) or \
                    (child.last_state is not None and
                     not child.last_state[2]):
                child.last_state = (self.environment.step(robot, current_action))

                (new_state, reward, goal) = child.last_state
                if goal:
                    self.set_status_text('Terminated, agent reward {r}\n'.format(r=reward))

                self.display_agent_state(new_state)

                # continue execution
                if goal and reward > 0:
                    child.last_state = (new_state, reward, False)

                    self.reestate_asset()

            for child in self.environment.children:
                child.on_update()

        if robot is not None:
            self.robot_local_world.draw(self.environment.localMap(robot),
                                        self.environment.horizon, robot)

    def activate_random(self):
        # warning add robot without collision
        self.add_agents(NDELIVERY_AGENTS, DeliveryRobot)
        self.add_agents(NOBSTACLES, Obstacle)
        cnt = 0
        while cnt < NASSETS:
            if self.add_asset():
                cnt += 1

    def deposit_basis_income(self):
        for agent in self.environment.agents:
            if isinstance(agent, DeliveryRobot):
                DLT().transaction(0, agent.wallet, BASIS_INCOME)

    def reestate_asset(self):
        for chld in self.environment.children:
            if isinstance(chld.agent, Asset) and chld.agent.delivered:
                self.environment.agents.remove(chld.agent)
                self.environment.remove_widget(chld)
                while not self.add_asset():
                    pass
                break

    def add_asset(self):
        pos = (np.random.randint(self.environment.sizeX),
               np.random.randint(self.environment.sizeY))
        dest = (np.random.randint(1, self.environment.sizeX - 1),
                np.random.randint(1, self.environment.sizeY - 1))
        env = self.environment.renderEnv(plot=False)
        if env[pos] is None and \
                (env[dest] is None or isinstance(env[dest], Obstacle)):
            self.environment.add(Asset(list(pos), list(dest)))
            return True
        return False

    def add_agents(self, nR, cls):
        cnt = 0
        while cnt < nR:
            pos = (np.random.randint(self.environment.sizeX),
                   np.random.randint(self.environment.sizeY))
            if self.environment.renderEnv(plot=False)[pos] is None:
                self.environment.add(cls(list(pos)))
                cnt += 1

    def display_agent_state(self, new_state):
        rsum = '\n'
        for r in new_state['robots']:
            rsum += f'[{r["position"]},{r["loaded"]}]\n'
        asum = '\n'
        for r in new_state['assets']:
            asum += str(r) + '\n'
        self.left_panel.ids.state.text = 'ID: {id}\n' \
                                         'Loaded : {loaded}\n' \
                                         'Balance: {balance}\n' \
                                         'Position: {position}\n' \
                                         'Destination: {destination}\n' \
                                         'Assets: {asum}' \
                                         'Other Robots: {rsum}'.format(**new_state,
                                                                       asum=asum,
                                                                       rsum=rsum)

    def on_touch_down(self, touch, *args):
        def fun(dt):
            for child in self.environment.children:
                if child.selected:
                    robot = child.agent
                    self.display_agent_state(self.environment.get_compound_state(robot))
                    self.robot_local_world.draw(self.environment.localMap(robot),
                                                self.environment.horizon,
                                                robot)

        Clock.schedule_once(fun, 0.1)

    def encode_state(self, o, wstate):
        state = {}

        def _maprep(e):
            if e == o:
                return 1.0;
            if isinstance(e, Obstacle):
                return 1.0
            if isinstance(e, DeliveryRobot):
                return 1.0
            if isinstance(e, Asset):
                return -1
            return 0.0

        maprepr = np.vectorize(_maprep)
        hsize = self.environment.local_map_size()
        state['map'] = maprepr(np.array(wstate['world'])).reshape((1, hsize, hsize))

        mspace = np.array([self.environment.sizeX, self.environment.sizeY])

        def norm_coords(coord):
            return coord / mspace

        def to_list(i):
            return np.array([i['x'], i['y']]) / mspace

        state['coord'] = to_list(o.get_position()).reshape((1, 2))
        state['orient'] = o.get_orientation().reshape((1, 2))

        dest = wstate['destination']
        if wstate['loaded']:
            dest = norm_coords(dest)
        state['dest'] = np.array(dest).reshape((1, 2))
        state['loaded'] = np.array([1 if wstate['loaded'] else 0]).reshape((1, 1))

        state['assets'] = np.zeros((1, 3, 2, 2));
        idx = 0
        for ass in wstate['assets']:
            state['assets'][0, idx, :, :] = np.array([to_list(ass['position']),
                                            norm_coords(ass['destination'])])
            idx += 1

        return state

    def sim_reset(self):

        self.environment.reset()
        self.set_status_text('reset')
        self.activate_random()

        for w in self.environment.children:
            if isinstance(w.agent, DeliveryRobot):
                w.q_state = learning.DoubleQN.get_new_rnn_state()
                w.experience = learning.ExperienceBuffer()
        gc.collect()

    def start_sim(self):
        path = 'gamegym-train'
        tf.reset_default_graph()
        self.q = learning.DoubleQN(self.environment.horizon)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        totalExperience = learning.ExperienceBuffer()

        self.vis = Visualization()
        self.sim_reset()

        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is not None:
           print("Loading model")
           saver.restore(sess, ckpt.model_checkpoint_path)
        else:
           sess.run(init)

        def simulation_routine(count, dt):
            completed = False
            for w in self.environment.children:
                if isinstance(w.agent, DeliveryRobot):
                    self.q.set_rnn_state(w.q_state)
                    enc_state = self.encode_state(w.agent,
                                                  self.environment.get_compound_state(
                                                      w.agent))
                    action, new_rnn_state, _ = self.q.predict(sess, enc_state, epsilon=0.05)
                    w.q_state = new_rnn_state

                    new_state, total_reward, completed = self.environment.step(w.agent, action)

                    w.experience.add_to_episode([
                        enc_state['map'].reshape((7,7)),  # BATCH_INDEX_MAP
                        enc_state['coord'],  # BATCH_INDEX_COORD
                        enc_state['orient'],  # BATCH_INDEX_ORIEN
                        enc_state['dest'],  # BATCH_INDEX_DEST
                        enc_state['loaded'],  # BATCH_INDEX_LOAD
                        np.array([action]),  # BATCH_INDEX_ACTION
                        np.array([1 if completed else 0]),  # BATCH_INDEX_SUCCESS_FLAG
                        np.array([total_reward])  # BATCH_INDEX_REWARD
                    ], new_episode=(count == 0))

                    self.vis.add_action(action)
                    self.vis.add_reward(total_reward)

                    if completed:
                        # print(new_state, total_reward, completed)
                        break

                    self.display_agent_state(new_state)
                    self.robot_local_world.draw(new_state['world'],
                                                self.environment.horizon,
                                                w.agent)
                    w.on_update()

                    if count % 100 == 0:
                        self.vis.update()

            if completed:
                totalExperience.add(w.experience.buffer[-1])
                self.sim_reset()
                count = -1

                if (totalExperience.cnt > 0) and (totalExperience.cnt % 100 == 0):
                    loss = self.q.train(sess, totalExperience, 100, 10, y=0.1)
                    self.vis.add_train_loss(loss)
                    saver.save(sess, path + '/model-' + str(1) + '.cptk')

            self.sim = Clock.schedule_once(partial(simulation_routine, count + 1), 0)

        simulation_routine(0, 0)


if __name__ == '__main__':
    tf.reset_default_graph()
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    GameGymApp().run()
