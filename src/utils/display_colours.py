import random

def get_colors_to_use(colors):
    unique_colors = set(list(colors.values()))
    colors_to_use = {c:get_random_color() for c in unique_colors if c is not None}
    node_colors_list = []
    for u in colors.keys():
        if colors[u] is not None:
            node_colors_list.append(colors_to_use[colors[u]])
        else:
            node_colors_list.append('#FFFFFF')
    return node_colors_list

def get_random_color():
    hexadecimal = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])][0]
    return hexadecimal