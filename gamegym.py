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

BASIS_INCOME = 1.0

A_KEY = ord('a')
R_KEY = ord('r')
D_KEY = ord('d')
P_KEY = ord('p')
I_KEY = ord('i')
LEFT_KEY = 276
RIGHT_KEY = 275
BACKWARD_KEY = 274
FORWARD_KEY = 273

NASSETS = 3
NOBSTACLES = 5
NDELIVERY_AGENTS = 4


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

        if scancode == R_KEY:
            self.environment.reset()
            self.set_status_text('reset')

        # activate
        if scancode == A_KEY:
            # warning add robot without collision
            self.add_agents(NDELIVERY_AGENTS, DeliveryRobot)
            self.add_agents(NOBSTACLES, Obstacle)

            cnt = 0
            while cnt < NASSETS:
                if self.add_asset():
                    cnt += 1

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
                                        self.environment.horizont, robot)

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
                                                self.environment.horizont,
                                                robot)

        Clock.schedule_once(fun, 0.1)


if __name__ == '__main__':
    GameGymApp().run()
