# DataCollection.py
class DataCollection:
    def __init__(self, all_data):
        self.data_file = open("ce889_dataCollection.csv", "a")
        self.data_file.close()
        self.buffer = []
        self.all_data = (all_data == "TRUE")

    def get_input_row(self, lander, surface, controller):
        """
        获取输入行，包含 [X_dist, Y_dist, Vx, Vy]
        :param lander: Lander 对象
        :param surface: Surface 对象
        :param controller: Controller 对象
        :return: 包含四个数值的列表
        """
        # 计算飞船与着陆点的距离
        X_dist = surface.centre_landing_pad[0] - lander.position.x
        Y_dist = surface.centre_landing_pad[1] - lander.position.y

        # 获取当前速度
        Vx = lander.velocity.x
        Vy = lander.velocity.y

        # 返回四个数值的列表
        input_row = [X_dist, Y_dist, Vx, Vy]
        print(f"Generated input_row: {input_row}")  # Debug print
        return input_row

    def save_current_status(self, input_row, lander, surface, controller):
        """
        保存当前状态到缓冲区
        :param input_row: 包含 [X_dist, Y_dist, Vx, Vy] 的列表
        :param lander: Lander 对象
        :param surface: Surface 对象
        :param controller: Controller 对象
        """
        # 输出
        thrust = 1 if controller.is_up() else 0
        new_vel_y = lander.velocity.y
        new_vel_x = lander.velocity.x

        turning = [0, 0]
        if controller.is_left():
            turning = [1, 0]
        elif controller.is_right():
            turning = [0, 1]

        new_angle = lander.current_angle

        if self.all_data:
            # 额外特征：current_speed, current_angle, dist_to_surface
            current_speed = lander.velocity.length()
            dist_to_surface = surface.polygon_rect.topleft[1] - lander.position.y

            # 构建包含所有特征的行
            status_row = ",".join(map(str, [current_speed] + input_row)) + "," + \
                         str(thrust) + "," + \
                         str(new_vel_y) + "," + \
                         str(new_vel_x) + "," + \
                         str(new_angle) + "," + \
                         str(turning[0]) + "," + str(turning[1]) + "\n"
        else:
            # 仅包含四个特征
            status_row = ",".join(map(str, input_row)) + "," + \
                         str(new_vel_y) + "," + \
                         str(new_vel_x) + "\n"

        # 保存到缓冲区
        self.buffer.append(status_row)
        print(f"Saved status_row: {status_row}")  # Debug print

    def write_to_file(self):
        """
        将缓冲区的数据写入文件
        """
        self.data_file = open("ce889_dataCollection.csv", "a")
        for row in self.buffer:
            self.data_file.write(row)
        self.data_file.close()
        print("Data written to file.")  # Debug print

    def reset(self):
        """
        重置缓冲区
        """
        self.buffer = []
        print("Buffer reset.")  # Debug print
