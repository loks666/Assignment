# GameLoop.py
import ctypes
import pygame
import sys

from Controller import Controller
from DataCollection import DataCollection
from EventHandler import EventHandler
from GameLogic import GameLogic
from Lander import Lander
from MainMenu import MainMenu
from NeuralNetHolder import NeuralNetHolder
from ResultMenu import ResultMenu
from Surface import Surface
from Vector import Vector


class GameLoop:

    def __init__(self):
        self.controller = Controller()
        self.Handler = EventHandler(self.controller)
        self.object_list = []
        self.game_logic = GameLogic()
        self.fps_clock = pygame.time.Clock()
        self.fps = 60
        self.neuralnet = NeuralNetHolder()
        self.version = "v1.01"
        self.prediction_cycle = 0
        self.lander = None
        self.surface = None

    def init(self, config_data):
        # 初始化 pygame 库
        pygame.init()

        # 读取配置文件中的全屏设置
        if config_data["FULLSCREEN"].upper() == "TRUE":
            user32 = ctypes.windll.user32
            config_data['SCREEN_HEIGHT'] = int(user32.GetSystemMetrics(1))
            config_data['SCREEN_WIDTH'] = int(user32.GetSystemMetrics(0))
            self.screen = pygame.display.set_mode(
                (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']),
                pygame.FULLSCREEN
            )
        else:
            # 窗口模式，通过配置文件设置屏幕尺寸
            config_data['SCREEN_HEIGHT'] = int(config_data['SCREEN_HEIGHT'])
            config_data['SCREEN_WIDTH'] = int(config_data['SCREEN_WIDTH'])
            self.screen = pygame.display.set_mode(
                (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT'])
            )  # 窗口模式

        pygame.display.set_caption('CE889 Assignment Template')
        pygame.display.set_icon(pygame.image.load(config_data['LANDER_IMG_PATH']))

    def score_calculation(self):
        score = 1000.0 - (self.surface.centre_landing_pad[0] - self.lander.position.x)
        angle = self.lander.current_angle
        if angle == 0:
            angle = 1
        if angle > 180:
            angle = abs(angle - 360)
        score = score / angle
        velocity = 500 - (self.lander.velocity.x + self.lander.velocity.y)
        score += velocity

        print("Lander difference: ", self.surface.centre_landing_pad[0] - self.lander.position.x)
        print("SCORE: ", score)

        return score

    def main_loop(self, config_data):
        pygame.font.init()  # 初始化字体模块
        myfont = pygame.font.SysFont('Comic Sans MS', 30)

        # 创建用于渲染的精灵组
        sprites = pygame.sprite.Group()

        # 游戏状态的布尔值
        on_menus = [True, False, False]  # 主菜单，赢，输
        game_start = False

        # 游戏模式：播放游戏，数据收集，神经网络，退出
        game_modes = [False, False, False, False]

        # 背景图片
        background_image = pygame.image.load(config_data['BACKGROUND_IMG_PATH']).convert_alpha()
        background_image = pygame.transform.scale(
            background_image,
            (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT'])
        )

        # 数据收集器和菜单
        # 初始时 all_data 根据 config_data["ALL_DATA"] 设置
        data_collector = DataCollection(config_data["ALL_DATA"])
        main_menu = MainMenu((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        result_menu = ResultMenu((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        score = 0

        # 初始化
        while True:
            # 处理菜单和退出
            if game_modes[-1]:  # 如果选择退出
                pygame.quit()
                sys.exit()

            # 如果游戏开始，初始化所有对象
            if game_start:
                self.controller = Controller()
                self.Handler = EventHandler(self.controller)
                sprites = pygame.sprite.Group()
                self.game_start(config_data, sprites)
                game_start = False  # 防止重复初始化

            # 处理菜单显示
            if any(on_menus):
                if on_menus[1] or on_menus[2]:  # 赢或输菜单
                    result_menu.draw_result_objects(self.screen, on_menus[1], score)
                else:  # 主菜单
                    main_menu.draw_buttons(self.screen)
                    # 绘制版本号
                    textsurface = myfont.render(self.version, False, (0, 0, 0))
                    self.screen.blit(textsurface, (10, 10))

                for event in pygame.event.get():
                    if on_menus[0]:  # 主菜单
                        main_menu.check_hover(event)
                        button_clicked = main_menu.check_button_click(event)
                        if button_clicked > -1:
                            game_modes[button_clicked] = True
                            on_menus[0] = False
                            game_start = True
                            # 根据选择的游戏模式，设置 data_collector 的 all_data
                            if button_clicked == 1:  # 数据收集模式
                                data_collector.all_data = True
                            elif button_clicked == 2:  # 神经网络模式
                                data_collector.all_data = False
                    elif on_menus[1] or on_menus[2]:  # 赢或输菜单
                        result_menu.check_hover(event)
                        back_to_main = result_menu.check_back_main_menu(event)
                        if back_to_main:
                            on_menus[0] = True
                            on_menus[1] = False
                            on_menus[2] = False

            else:
                # 游戏进行中
                self.Handler.handle(pygame.event.get())

                # 检查是否激活神经网络控制模式（假设 game_modes[2] 是神经网络模式）
                if game_modes[2]:
                    self.prediction_cycle += 1
                    # 每隔一定帧进行一次预测（例如，每2帧）
                    if self.prediction_cycle % 2 == 0:
                        input_row = data_collector.get_input_row(self.lander, self.surface, self.controller)
                        print(f"获取到的 input_row: {input_row}")  # 添加调试输出
                        nn_prediction = self.neuralnet.predict(input_row)
                        # 重置控制器
                        self.controller.set_up(False)
                        self.controller.set_left(False)
                        self.controller.set_right(False)

                        # 根据预测调整控制器状态
                        if self.lander.velocity.y > nn_prediction[1]:
                            self.controller.set_up(True)

                        if self.lander.velocity.x < nn_prediction[0]:
                            self.controller.set_right(True)
                        elif self.lander.velocity.x > nn_prediction[0]:
                            self.controller.set_left(True)

                        # 限制最大角度
                        if 30 < self.lander.current_angle < 330:
                            ang_val = (self.lander.current_angle - 30) / (330 - 30)
                            ang_val = round(ang_val)
                            if ang_val == 0:
                                self.lander.current_angle = 30
                            else:
                                self.lander.current_angle = 330

                        # 调试输出
                        print("Current controller status:", self.controller.up, self.controller.left, self.controller.right)
                        print("Lander velocity:", self.lander.velocity.x, self.lander.velocity.y)
                        print("Predicted velocity:", nn_prediction)

                # 渲染背景
                self.screen.blit(background_image, (0, 0))

                # 更新游戏对象
                self.update_objects()

                # 数据收集模式
                if game_modes[1]:
                    input_row = data_collector.get_input_row(self.lander, self.surface, self.controller)
                    data_collector.save_current_status(input_row, self.lander, self.surface, self.controller)

                # 绘制精灵
                sprites.draw(self.screen)

                # 检查碰撞和游戏状态
                if self.lander.landing_pad_collision(self.surface):
                    score = self.score_calculation()
                    on_menus[1] = True
                    if game_modes[1]:  # 数据收集模式下，保存数据
                        data_collector.write_to_file()
                        data_collector.reset()
                elif self.lander.surface_collision(self.surface) or self.lander.window_collision(
                        (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT'])):
                    on_menus[2] = True
                    data_collector.reset()

            # 更新显示和控制帧率
            pygame.display.flip()
            self.fps_clock.tick(self.fps)

    def update_objects(self):
        # 更新游戏对象的速度和位置
        self.game_logic.update(0.2)

    def setup_lander(self, config_data):
        lander = Lander(
            config_data['LANDER_IMG_PATH'],
            [config_data['SCREEN_WIDTH'] / 2, config_data['SCREEN_HEIGHT'] / 2],
            Vector(0, 0),
            self.controller
        )
        self.game_logic.add_lander(lander)
        return lander

    def game_start(self, config_data, sprites):
        # 创建着陆器对象
        self.lander = self.setup_lander(config_data)
        self.surface = Surface((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        sprites.add(self.lander)
        sprites.add(self.surface)
