# coding = UTF-8

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from operator import itemgetter, attrgetter

cross_points = []
trace_data = []
cars = {}
range_x = 100
range_y = 100
now_point = 1
start_time = 1493852331
end_time = 1494982799
traffic_data = []

'''点：x，y'''
class point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

'''路口：标号，路口x，路口y，四角点坐标'''
class CrossPoint:
    def __init__(self, index, point_x, point_y, right_down, left_down, right_up, left_up):
        self.index = index
        self.point_x = float(point_x)
        self.point_y = float(point_y)
        self.right_down = right_down
        self.left_down = left_down
        self.right_up = right_up
        self.left_up = left_up
    def get_center_point(self):
        return ((self.right_down.x + self.left_down.x) / 2 + (self.right_up.x + self.left_up.x) / 2) / 2, \
        ((self.right_down.y + self.left_down.y) / 2 + (self.right_up.y + self.left_up.y) / 2) / 2

'''轨迹数据结构：标号，时间戳，x，y，车速，载客'''
class Trace:
    def __init__(self, index, time, x, y, speed, state):
        self.index = index
        self.time = time
        self.x = x
        self.y = y
        self.speed = speed
        self.state = state

'''交通流：标号，开始时间，经过的相位'''
class Traffic:
    def __init__(self, index, start_time):
        self.index = index
        self.start_time = start_time
        self.phase = []

    def append_phase(self, phase_data):
        self.phase.append(phase_data)

    def print_traffic(self):
        print self.index, self.start_time
        for i in self.phase:
            i.print_phase()

'''相位：标号，方向，时间戳序列'''
class Phase:
    def __init__(self, index, directions, time):
        self.index = index
        self.directions = directions
        self.time = time

    def print_phase(self):
        print self.index, self.directions, self.time

'''？？？'''
class AddPhase:
    def __init__(self, index, directions, start_time, end_time):
        self.index = index
        self.directions = directions
        self.start_time = start_time
        self.end_time = end_time

'''路口：每个路口的四角点坐标'''
class Section:
    def __init__(self, right_down, left_down, right_up, left_up):
        self.right_down = right_down
        self.left_down = left_down
        self.right_up = right_up
        self.left_up = left_up

    def check_point(self, x, y, speed):
        if speed != 0.0 and x <= self.right_down.x and x >= self.left_down.x and x <= self.right_up.x and x >= self.left_up.x \
                and y <= self.right_up.y and y <= self.left_up.y and y >= self.left_down.y and y >= self.right_down.y:
            return True
        else:
            return False

    def print_section(self):
        print self.right_down.x, self.right_down.y
        print self.right_up.x, self.right_up.y
        print self.left_down.x, self.left_down.y
        print self.left_up.x, self.left_up.y

'''数据给定的路口点坐标'''
def draw_cross_points():
    points_x = []
    points_y = []
    for cross_point in cross_points:
        points_x.append(cross_point.point_x)
        points_y.append(cross_point.point_y)
    return points_x, points_y

'''路口i附近的车轨迹图数据点-可以用来确定每个路口四角坐标'''
def draw_car_trace(i):
    time_trace = sorted(trace_data, key=attrgetter('time', 'index'))
    draw_x = []
    draw_y = []
    temp_cross_point = cross_points[i]
    rangex = [temp_cross_point.point_x - range_x, temp_cross_point.point_x + range_x]
    rangey = [temp_cross_point.point_y - range_y, temp_cross_point.point_y + range_y]
    for k in time_trace:
        if (k.x <= rangex[1]) and (k.x >= rangex[0]) and (k.y >= rangey[0]) and (k.y <= rangey[1]):
            draw_x.append(k.x)
            draw_y.append(k.y)
    return draw_x, draw_y

'''所有轨迹数据形成的车道图数据点'''
def draw_all_trace():
    time_trace = sorted(trace_data, key=attrgetter('time', 'index'))
    draw_x = []
    draw_y = []
    for k in time_trace:
        draw_x.append(k.x)
        draw_y.append(k.y)
    return draw_x, draw_y

'''画出所有轨迹数据形成的车道图，以及路口红点'''
def draw_all_data():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_x, draw_y = draw_all_trace()
    ax.scatter(draw_x, draw_y)
    cm = plt.get_cmap("RdYlGn")
    col = cm(20)
    points_x, points_y = draw_cross_points()
    ax.scatter(points_x, points_y, c=col)
    plt.show()

'''画出路口i附近的车轨迹图，以及路口红点'''
def draw_one_data(i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_x, draw_y = draw_car_trace(i)
    ax.scatter(draw_x, draw_y)
    cm = plt.get_cmap("RdYlGn")
    col = cm(20)
    point_x = [cross_points[i].point_x]
    point_y = [cross_points[i].point_y]
    ax.scatter(point_x, point_y, c=col)
    plt.show()

'''计算相位i的车辆数（流量）'''
def count_cars(i):
    print i
    temp_traffic = traffic_data[i]
    cross_point = cross_points[i]
    sections = []
    sections.append(Section(point(cross_point.right_down.x + range_x, cross_point.right_down.y - range_y), cross_point.right_down,
                            point(cross_point.right_up.x + range_x, cross_point.right_up.y + range_y), cross_point.right_up))
    sections.append(Section(point(cross_point.right_down.x + range_x, cross_point.right_down.y - range_y),
                            point(cross_point.left_down.x-range_x, cross_point.left_down.y - range_y),
                            cross_point.right_down, cross_point.left_down))
    sections.append(Section(cross_point.left_down, point(cross_point.left_down.x - range_x, cross_point.left_down.y - range_y),
                            cross_point.left_up, point(cross_point.left_up.x - range_x, cross_point.left_up.y + range_y)))
    sections.append(Section(cross_point.right_up, cross_point.left_up,
                            point(cross_point.right_up.x + range_x, cross_point.right_up.y + range_y),
                            point(cross_point.left_up.x - range_x, cross_point.left_up.y + range_y)))
    time_trace = sorted(trace_data, key=attrgetter('time', 'index'))
    draw_x = []
    draw_y = []
    # draw_x.append(sections[0].right_down.x)
    # draw_x.append(sections[0].left_down.x)
    # draw_x.append(sections[0].right_up.x)
    # draw_x.append(sections[0].left_up.x)
    # draw_y.append(sections[0].right_down.y)
    # draw_y.append(sections[0].left_down.y)
    # draw_y.append(sections[0].right_up.y)
    # draw_y.append(sections[0].left_up.y)
    # return draw_x, draw_y
    # for k in time_trace:
    #     for section in sections:
    #         if section.check_point(k.x, k.y, 1.0):
    #             print k.index, k.x, k.y, k.time, k.speed
    #             draw_x.append(k.x)
    #             draw_y.append(k.y)
    # return draw_x, draw_y
    temp_start_time = int(temp_traffic.start_time)
    count_phase = {}
    # while 1:
    #     if temp_start_time > end_time:
    #         break
    #     for phase in temp_traffic.phase:
    #         temp_end_time = temp_start_time + phase.time
    #         index = phase.index
    #         count = 0
    #         for direction in phase.directions:
    #             if direction == -1:
    #                 continue
    #             temp_section = sections[direction]
    #             for trace in time_trace:
    #                 if trace.time > temp_end_time:
    #                     break
    #                 if trace.time > temp_start_time:
    #                     if temp_section.check_point(trace.x, trace.y, trace.speed):
    #                         count += 1
    #         if count > count_phase.get(index, -1):
    #             count_phase[index] = count
    #             print count
    #         temp_start_time = temp_end_time
    result = {}
    result_avg = {}
    count_cycle = 0
    # 计算每一个相位对应区域的车辆流量
    while 1:
        if len(time_trace) == 0:
            break
        temp_end_time = temp_start_time + 200
        counts = []
        while 1:
            try:
                temp_trace = time_trace[0]
                if temp_trace.time < temp_end_time:
                    counts.append(temp_trace)
                    time_trace.pop(0)
                else:
                    break
            except Exception, e:
                break
        count_cycle += 1
        temp_phase_start_time = temp_start_time
        temp_phase_end_time = temp_start_time
        for phase in temp_traffic.phase:
            temp_phase_start_time = temp_phase_end_time
            temp_phase_end_time = temp_phase_start_time + phase.time
            temp_cars = set()
            for direction in phase.directions:
                for trace in counts:
                    if direction == -1:
                        break
                    if trace.time > temp_phase_end_time:
                        break
                    temp_section = sections[direction]
                    if trace.time > temp_phase_start_time:
                        if temp_section.check_point(trace.x, trace.y, trace.speed):
                            temp_cars.add(trace.index)
            if result_avg.get(phase.index, -1) == -1:
                result_avg[phase.index] = len(temp_cars)
            else:
                result_avg[phase.index] += len(temp_cars)
            if len(temp_cars) > result.get(phase.index, -1):
                result[phase.index] = len(temp_cars)
        temp_start_time = temp_end_time
    print i, count_cycle
    for phase in temp_traffic.phase:
        print phase.index, float(1.0 * result_avg[phase.index] / (count_cycle * phase.time)), float(1.0 * result[phase.index] / phase.time)
    for k in result_avg:
        print k, result_avg[k]
    for k in result:
        print k, result[k]

'''计算协调相位i上行/下行流量'''
def count_up_down(i):
    print i
    temp_traffic = traffic_data[i]
    cross_point = cross_points[i]
    sections = []
    sections.append(Section(point(cross_point.right_down.x + range_x, cross_point.right_down.y - range_y),
                            point(cross_point.left_down.x - range_x, cross_point.left_down.y - range_y),
                            point(cross_point.right_up.x + range_x, cross_point.right_up.y + range_y),
                            point(cross_point.left_up.x - range_x, cross_point.left_up.y + range_y)))
    time_trace = sorted(trace_data, key=attrgetter('time', 'index'))
    temp_start_time = int(temp_traffic.start_time)
    result_up_max = -1
    result_down_max = -1
    result_up_speed_max = -1
    result_down_speed_max = -1
    result_up_sum = 0
    result_down_sum = 0
    result_up_speed_sum = 0
    result_down_speed_sum = 0
    result_up_traces_sum = 0
    result_down_traces_sum = 0
    count_cycle = 0
    phase_time = 0
    start_phase = 0
    if i == 0:
        phase_time = temp_traffic.phase[0].time
    else:
        phase_time = temp_traffic.phase[1].time
        start_phase = temp_traffic.phase[0].time
    while 1:
        if len(time_trace) == 0:
            break
        temp_end_time = temp_start_time + 200
        counts = []
        while 1:
            try:
                temp_trace = time_trace[0]
                if temp_trace.time < temp_end_time:
                    counts.append(temp_trace)
                    time_trace.pop(0)
                else:
                    break
            except Exception, e:
                break
        count_cycle += 1
        temp_phase_start_time = temp_start_time + start_phase
        temp_phase_end_time = temp_phase_start_time + phase_time
        temp_traces = {}
        up_traces = []
        down_traces = []
        for trace in counts:
            if trace.time > temp_phase_end_time:
                break
            temp_section = sections[0]
            if trace.time > temp_phase_start_time:
                if temp_section.check_point(trace.x, trace.y, trace.speed):
                    if temp_traces.get(trace.index, -1) == -1:
                        temp_traces[trace.index] = []
                        temp_traces[trace.index].append(trace)
                    else:
                        temp_traces[trace.index].append(trace)
        for index in temp_traces:
            traces = temp_traces[index]
            traces = sorted(traces, key=attrgetter('time'))
            flag = True
            try:
                if traces[1].y > traces[0].y:
                    flag = False
            except Exception,e:
                pass
            for trace in traces:
                if flag:
                    up_traces.append(trace)
                else:
                    down_traces.append(trace)
        up_cars = set()
        down_cars = set()
        for trace in up_traces:
            up_cars.add(trace.index)
            if trace.speed == 0.0:
                up_traces.remove(trace)
                continue
            result_up_speed_sum += trace.speed
            if trace.speed > result_up_speed_max:
                result_up_speed_max = trace.speed
        for trace in down_traces:
            down_cars.add(trace.index)
            if trace.speed == 0.0:
                down_traces.remove(trace)
                continue
            result_down_speed_sum += trace.speed
            if trace.speed > result_down_speed_max:
                result_down_speed_max = trace.speed
        result_up_sum += len(up_cars)
        result_down_sum += len(down_cars)
        result_up_traces_sum += len(up_traces)
        result_down_traces_sum += len(down_traces)
        if len(up_cars) > result_up_max:
            result_up_max = len(up_cars)
        if len(down_cars) > result_down_max:
            result_down_max = len(down_cars)
            # if result_avg.get(phase.index, -1) == -1:
            #     result_avg[phase.index] = len(temp_cars)
            # else:
            #     result_avg[phase.index] += len(temp_cars)
            # if len(temp_cars) > result.get(phase.index, -1):
            #     result[phase.index] = len(temp_cars)
        temp_start_time = temp_end_time

    print float(1.0 * result_up_max / phase_time), float(1.0 * result_up_sum / (phase_time * count_cycle)), float(1.0 * result_up_speed_sum / result_up_traces_sum), result_up_speed_max
    print float(1.0 * result_down_max / phase_time), float(1.0 * result_down_sum / (phase_time * count_cycle)), float(1.0 * result_down_speed_sum / result_down_traces_sum), result_down_speed_max

'''计算点之间的距离'''
def cal_distance(point1, point2):
    import math
    return math.sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y))

if __name__ == '__main__':
    '''调用cal_distance计算相邻路口间的距离'''
    # with open('cross_point.txt', 'r') as f:
    #     while 1:
    #         line = f.readline()
    #         if not line:
    #             break
    #         paras = line.split(';')
    #         index = paras[0]
    #         point_x = paras[1]
    #         point_y = paras[2]
    #         right_down = point(paras[3].split(',')[0], paras[3].split(',')[1])
    #         left_down = point(paras[4].split(',')[0], paras[4].split(',')[1])
    #         right_up = point(paras[5].split(',')[0], paras[5].split(',')[1])
    #         left_up = point(paras[6].split(',')[0], paras[6].split(',')[1])
    #         temp_point = CrossPoint(index, point_x, point_y, right_down, left_down, right_up, left_up)
    #         cross_points.append(temp_point)
    # centers = []
    # for cross in cross_points:
    #     temp_x, temp_y = cross.get_center_point()
    #     temp_point = point(temp_x, temp_y)
    #     centers.append(temp_point)
    # for i in range(0,6):
    #     print cal_distance(centers[i], centers[i+1])


    count = 0
    count_car = 1
    with open('trace_data.txt', 'r') as f:
        while 1:
            line = f.readline()
            count += 1
            if count == 1:
                continue
            if not line:
                break
            paras = line.split(',')
            index = paras[0].strip()
            if cars.get(index, -1) == -1:
                cars[index] = count_car
                count_car += 1
            time = int(paras[1].strip())
            x = float(paras[2].strip())
            y = float(paras[3].strip())
            speed = float(paras[4].strip())
            state = paras[5].strip()
            temp_trace = Trace(cars.get(index), time, x, y, speed, state)
            trace_data.append(temp_trace)
    time_trace = sorted(trace_data, key=attrgetter('time', 'index'))
    temp_traffic = None
    with open('xiangwei.txt', 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            paras = line.split(';')
            if len(paras) == 2:
                traffic = Traffic(index=paras[0], start_time=int(paras[1]))
                traffic_data.append(traffic)
                temp_traffic = traffic
            else:
                temp_directions = []
                temp_directions.append(int(paras[1]))
                temp_directions.append(int(paras[2]))
                temp_directions.append(int(paras[3]))
                temp_directions.append(int(paras[4]))
                temp_phase = Phase(index=int(paras[0]), directions=temp_directions, time=int(paras[5]))
                temp_traffic.append_phase(temp_phase)
    for i in range(0,7):
        count_up_down(i)
    draw_x, draw_y = count_cars(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(draw_x, draw_y)
    cm = plt.get_cmap("RdYlGn")
    col = cm(20)
    points_x, points_y = draw_cross_points()
    ax.scatter(points_x, points_y, c=col)
    plt.show()