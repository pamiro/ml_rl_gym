from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.graphics.context_instructions import PushMatrix, Rotate, PopMatrix
from kivy.graphics.vertex_instructions import Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget

from world import DeliveryRobot, Agent, Actions
from world import Environment


##
# why am i doing this?!
# to visualize simulation ?!
# я хочу проанализировать - смогу ли я достичь более высокоуровнего
#  поведения за счёт введения экономических отношений между агентами

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
        # rebuilding grafical stack
        cellX = self.parent.size[0] / self.parent.sizeX;
        cellY = self.parent.size[1] / self.parent.sizeY;

        agent_pos = self.agent.get_position();

        (pposX, pposY) = self.parent.pos;
        self.pos = (int(agent_pos['x'] * cellX) + pposX,
                    int(agent_pos['y'] * cellY) + pposY)

        self.size_hint = (1 / self.parent.sizeX, 1 / self.parent.sizeY)
        self.source = 'images/icons8-truck-50.png'

        self.canvas.before.clear()
        self.canvas.after.clear()

        with self.canvas.before:
            if self.selected:
                self.draw_selected()

            PushMatrix()
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
        # print('actor move', touch)
        pass

    def on_touch_up(self, touch):
        # print('actor up', touch)
        pass

    def draw_selected(self):
        # with self.canvas:
        Color(0, 1, 0, 1)
        Rectangle(size=self.size,
                  pos=self.pos)


class EnvironmentScreen(FloatLayout, Environment):
    def __init__(self, *args, **kwargs):
        super(EnvironmentScreen, self).__init__(*args, **kwargs)
        print(Window.size)

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
        else:
            raise RuntimeError("unknown type")


class RobotViewWidget(Widget):
    def __init__(self, **kwargs):
        super(RobotViewWidget, self).__init__(**kwargs)
        self.bind(parent=self.on_update)
        self.bind(size=self.on_update)
        self.bind(pos=self.on_update)

    def on_update(self, *args, **kwargs):
        print(self.pos, self.size)
        self.canvas.clear()
        with self.canvas:
            # Color(0, 1, 0, 1)
            # Rectangle(size=self.size,
            #          pos=self.pos)
            pass


class GameApp(App):
    env = EnvironmentScreen(size=[100, 100]);
    rview = RobotViewWidget(size=[40, 40])

    def build(self):
        Window.size = (600, 600)
        Window.clearcolor = (1, 1, 1, 1)
        root = BoxLayout(orientation='vertical')
        root.add_widget(self.env)
        root.add_widget(self.rview)

        Window.bind(on_key_down=self.key_action)
        self.env.reset()
        return root

    def key_action(self, key, scancode=None, codepoint=None, modifier=None, *args):
        print("got a key event: %s" % scancode)

        if scancode == 114:
            print('reset')
            self.env.reset()

        if scancode == 97:
            # warning add robot without collistion
            self.env.add(DeliveryRobot([1, 1]))
            self.env.add(DeliveryRobot([2, 2]))
            self.env.add(DeliveryRobot([3, 3]))

        if len(self.env.children) == 0:
            return

        robot = None

        for child in self.env.children:
            if child.selected:
                robot = child.agent
                break

        current_action = None
        if scancode == 273:
            current_action = Actions.MOVE_FORWARD

        if scancode == 274:
            current_action = Actions.MOVE_BACKWARD

        if scancode == 275:
            current_action = Actions.TURN_RIGHT

        if scancode == 276:
            current_action = Actions.TURN_LEFT

        if current_action is not None and robot is not None:
            if (child.last_state is None) or (child.last_state is not None and not child.last_state[2]):
                child.last_state = (self.env.step(robot, current_action))
                print(robot.pos)
                print(child.last_state)
            child.on_update()


if __name__ == '__main__':
    GameApp().run()
