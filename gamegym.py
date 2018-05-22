from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image

from world import DeliveryRobot, Agent, Actions
from world import Environment

##
# why am i doing this?!
# to visualize simulation ?!

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
        cellX = self.parent.size[0] / self.parent.sizeX;
        cellY = self.parent.size[1] / self.parent.sizeY;

        agent_pos = self.agent.get_position();
        self.pos = (int(agent_pos['x'] * cellX + cellX / 2),
                    int(agent_pos['y'] * cellY + cellY / 2))
        self.size_hint = (1 / self.parent.sizeX, 1 / self.parent.sizeY)
        self.source = 'images/icons8-truck-50.png'
        self.canvas.before.clear()
        if self.selected:
            self.draw_selected()

    def on_touch_down(self, touch):
        print("actor down " + self.agent.name, touch)
        self.selected = True
        self.on_update()

    def on_touch_move(self, touch):
        #print('actor move', touch)
        pass

    def on_touch_up(self, touch):
        #print('actor up', touch)
        pass

    def draw_selected(self):
        with self.canvas.before:
            Color(0, 1, 0, 1)
            Rectangle(size=self.size,
                      pos=self.pos)


class EnvironmentScreen(FloatLayout, Environment):
    def __init__(self, *args, **kwargs):
        super(EnvironmentScreen, self).__init__(*args, **kwargs)

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


class GameApp(App):

    env = EnvironmentScreen(size=[100, 100]);

    def build(self):
        Window.size = (600, 600)
        Window.clearcolor = (1, 1, 1, 1)
        root = BoxLayout(orientation='vertical')
        root.add_widget(self.env)
        # root.add_widget(Button(pos=(0,0), size_hint=(0.1,0.1)))
        # root.add_widget(Button(pos=(10,10), size_hint=(0.1,0.1)))
        # .add_widget(self.actor)
        Window.bind(on_key_down=self.key_action)

        self.env.reset()
        return root

    def key_action(self, key, scancode=None, codepoint=None, modifier=None, *args):
        print("got a key event: %s" % scancode)

        # self.actor.pos[0]+=5;
        # self.actor.source = 'images/icons8-truck-filled-50.png'

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

        a = None
        if scancode == 273:
            a = Actions.MOVE_FORWARD

        if scancode == 274:
            a = Actions.MOVE_BACKWARD

        if scancode == 275:
            a = Actions.TURN_LEFT

        if scancode == 276:
            a = Actions.TURN_RIGHT

        if a is not None and robot is not None:
            if (child.last_state is None) or (child.last_state is not None and not child.last_state[2]):
                child.last_state = (self.env.step(robot, a))
                print(robot.pos)
                print(child.last_state)
            child.on_update()

if __name__ == '__main__':
    GameApp().run()