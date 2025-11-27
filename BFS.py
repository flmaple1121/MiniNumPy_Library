import matplotlib.pyplot as plt
from collections import deque


def bfs_shortest_path(maze, start, end):
    """
    使用广度优先搜索 (BFS) 寻找迷宫从起点到终点的最短路径。
    :param maze: 二维列表表示的迷宫，0为通路，1为墙壁
    :param start: 起点坐标 (行, 列)
    :param end: 终点坐标 (行, 列)
    :return: 包含路径坐标的列表，如果未找到路径则返回 None
    """
    # 定义四个移动方向：上、下、左、右
    # (-1, 0)表示行减1（向上），(1, 0)表示行加1（向下），以此类推
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    rows, cols = len(maze), len(maze[0])

    # 初始化队列。队列中的每个元素是一个元组：(当前坐标, 到达该坐标的路径列表)
    # 初始时，队列中只有起点及其路径
    queue = deque([(start, [start])])

    # 使用集合记录已访问过的节点，防止重复搜索和死循环
    visited = set()
    visited.add(start)

    while queue:
        # 从队列左侧取出一个节点进行扩展
        print(queue)
        (x, y), path = queue.popleft()

        # 检查是否到达终点
        if (x, y) == end:
            return path  # 找到终点，返回完整路径

        # 尝试向四个方向移动
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            # 检查新坐标是否有效：
            # 1. 在迷宫边界内
            # 2. 不是墙壁 (maze[new_x][new_y] == 0)
            # 3. 没有被访问过
            if 0 <= new_x < rows and 0 <= new_y < cols and \
                    maze[new_x][new_y] == 0 and (new_x, new_y) not in visited:
                # 标记为已访问，避免后续重复加入队列
                visited.add((new_x, new_y))

                # 构建到达新节点的路径
                new_path = path + [(new_x, new_y)]

                # 将新节点加入队列，等待后续搜索
                queue.append(((new_x, new_y), new_path))

    return None  # 队列为空仍未找到终点，说明无路可通


def draw_maze(maze, path=None):
    """
    使用 matplotlib 绘制迷宫和路径。
    :param maze: 二维列表表示的迷宫
    :param path: 路径坐标列表（可选）
    """
    rows, cols = len(maze), len(maze[0])

    # 创建画布和坐标轴
    fig, ax = plt.subplots()

    # 1. 绘制迷宫背景（墙壁和通路）
    for x in range(rows):
        for y in range(cols):
            # 计算绘图时的坐标。
            # 矩阵坐标 (x, y) -> 绘图坐标 (y, -x-1)
            # 这样做的目的是让矩阵的 (0,0) 显示在图形的左上角
            if maze[x][y] == 0:
                # 通路绘制为白色方块
                ax.add_patch(plt.Rectangle((y, -x - 1), 1, 1, facecolor="white"))
            else:
                # 墙壁绘制为灰色方块
                ax.add_patch(plt.Rectangle((y, -x - 1), 1, 1, facecolor="gray"))

    # 2. 绘制找到的最短路径
    if path:
        for x, y in path:
            # 路径绘制为绿色方块
            ax.add_patch(plt.Rectangle((y, -x - 1), 1, 1, facecolor="green"))

    # 3. 设置坐标轴属性，使显示更美观
    ax.set_aspect("equal")  # 保持纵横比 1:1，确保格子是正方形

    # 设置坐标轴范围
    ax.set_xlim(0, cols)
    ax.set_ylim(-rows, 0)

    # 隐藏刻度标签，只显示网格线
    ax.set_xticks(range(cols))
    ax.set_yticks(range(-rows, 0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.grid(True, color="black", linewidth=1)  # 显示网格线
    plt.show()


# --- 主程序 ---

# 定义迷宫结构：0 表示可通行，1 表示障碍物
maze = [
    [0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 0]
]

start = (0, 0)  # 起点坐标
end = (1, 4)  # 终点坐标

# 1. 计算路径
path = bfs_shortest_path(maze, start, end)

# 2. 打印结果（可选）
if path:
    print(f"找到最短路径，步数: {len(path)}")
    print(f"路径坐标: {path}")
else:
    print("未找到路径")

# 3. 绘制迷宫图
draw_maze(maze, path)