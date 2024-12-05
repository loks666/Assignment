# main.py
# Main Runnable file for the CE889 Assignment
# Project built by Lewis Veryard and Hugo Leon-Garza
from GameLoop import GameLoop

def importConfigFile():
    keys = []
    values = []
    with open("Files/Config.con", 'r') as file:
        for line in file:
            line_split = line.strip().split(',')
            for individual in line_split:
                individual = individual.replace(" ", "")
                content = individual.split('=')
                keys.append(content[0])
                values.append(content[1])
    return dict(zip(keys, values))

def start_game_window():
    game = GameLoop()
    game.init(config_data)
    game.main_loop(config_data)

if __name__ == "__main__":
    config_data = importConfigFile()
    start_game_window()
