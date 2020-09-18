"""

Grid based Dijkstra planning

author: Atsushi Sakai(@Atsushi_twi)

"""

import matplotlib.pyplot as plt
import math
from math import *
import cv2
import numpy as np

show_animation = False


class Dijkstra:

    def __init__(self,resolution, robot_radius, map):
        """
        Initialize map for a star planning


        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.min_x = 0
        self.min_y = 0
        self.max_x = 512
        self.max_y = 1024
        self.x_width = int(self.max_x/resolution)
        self.y_width = int(self.max_y/resolution)
        self.map = map

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.motion = None

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while 1:
            # TODO 选择A*还是贪心法
            #c_id = min(open_set, key=lambda o: open_set[o].cost)
            c_id = min(open_set,key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,open_set[o]))

            # for i in open_set.keys():
            #     print('cost1:  ',open_set[i].cost,'  cost2: ',self.calc_heuristic(goal_node,open_set[i]))
            # print()

            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                # if len(closed_set.keys()) % 10 == 0:
                #     plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id] #TODO
            #open_set.clear()

            # Add it to the closed set
            closed_set[c_id] = current

            # update motion cost
            self.motion = self.cal_motion_model_curr(current.x, current.y)
            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 70.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False

        return True


    def cal_motion_model_curr(self, cul_x, cul_y):
        # dx, dy, cost
        l_step = self.resolution
        motion = [[1, 0, float('inf')],
                  [0, 1, float('inf')],
                  [-1, 0, self.map[l_step*(cul_x-1),l_step*cul_y]],
                  [0, -1, float('inf')],
                  [-1, -1, self.map[l_step*(cul_x-1),l_step*(cul_y-1)]],
                  [-1, 1, self.map[l_step*(cul_x-1),l_step*(cul_y+1)]],
                  [1, -1, float('inf')],
                  [1, 1, float('inf')]]
        # print(l_step*(cul_x-1),l_step*cul_y,self.map[l_step*(cul_x-1),l_step*cul_y])
        # print(l_step*(cul_x-1),l_step*(cul_y-1),self.map[l_step*(cul_x-1),l_step*(cul_y-1)])
        # print(l_step*(cul_x-1),l_step*(cul_y+1),self.map[l_step*(cul_x-1),l_step*(cul_y+1)])
        return motion

# 建立一个欧氏距离的mask
mask = np.array([[10.63, 9.90, 9.21, 8.60, 8.06, 7.62, 7.28, 7.07, 7, 7.07, 7.28, 7.62, 8.06, 8.60, 9.21, 9.90, 10.63],
                 [10.00, 9.21, 8.49, 7.81, 7.21, 6.71, 6.32, 6.08, 6, 6.08, 6.32, 6.71, 7.21, 7.81, 8.49, 9.21, 10.00],
                 [9.43, 8.60, 7.81, 7.07, 6.40, 5.83, 5.39, 5.10, 5, 5.10, 5.39, 5.83, 6.40, 7.07, 7.81, 8.60, 9.43],
                 [8.94, 8.06, 7.21, 6.40, 5.66, 5.00, 4.47, 4.12, 4, 4.12, 4.47, 5.00, 5.66, 6.40, 7.21, 8.06, 8.94],
                 [8.54, 7.62, 6.71, 5.83, 5.00, 4.24, 3.61, 3.16, 3, 3.16, 3.61, 4.24, 5.00, 5.83, 6.71, 7.62, 8.54],
                 [8.25, 7.28, 6.32, 5.39, 4.47, 3.61, 2.83, 2.24, 2, 2.24, 2.83, 3.61, 4.47, 5.39, 6.32, 7.28, 8.25],
                 [8.06, 7.07, 6.08, 5.10, 4.12, 3.16, 2.24, 1.41, 1, 1.41, 2.24, 3.16, 4.12, 5.10, 6.08, 7.07, 8.06],
                 [8.00, 7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00, 0, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
                 [8.06, 7.07, 6.08, 5.10, 4.12, 3.16, 2.24, 1.41, 1, 1.41, 2.24, 3.16, 4.12, 5.10, 6.08, 7.07, 8.06],
                 [8.25, 7.28, 6.32, 5.39, 4.47, 3.61, 2.83, 2.24, 2, 2.24, 2.83, 3.61, 4.47, 5.39, 6.32, 7.28, 8.25],
                 [8.54, 7.62, 6.71, 5.83, 5.00, 4.24, 3.61, 3.16, 3, 3.16, 3.61, 4.24, 5.00, 5.83, 6.71, 7.62, 8.54],
                 [8.94, 8.06, 7.21, 6.40, 5.66, 5.00, 4.47, 4.12, 4, 4.12, 4.47, 5.00, 5.66, 6.40, 7.21, 8.06, 8.94],
                 [9.43, 8.60, 7.81, 7.07, 6.40, 5.83, 5.39, 5.10, 5, 5.10, 5.39, 5.83, 6.40, 7.07, 7.81, 8.60, 9.43],
                 [10.00, 9.21, 8.49, 7.81, 7.21, 6.71, 6.32, 6.08, 6, 6.08, 6.32, 6.71, 7.21, 7.81, 8.49, 9.21, 10.00],
                 [10.63, 9.90, 9.21, 8.60, 8.06, 7.62, 7.28, 7.07, 7, 7.07, 7.28, 7.62, 8.06, 8.60, 9.21, 9.90, 10.63]],dtype=np.float16)

def foothold_score(rectangle):
    alpha = 0.05
    shape = rectangle.shape

    mid = [shape[0]//2,shape[1]//2]
    score_dis = mask[8-mid[0]:8-mid[0]+shape[0],9-mid[1]:9-mid[1]+shape[1]]
    score = rectangle + alpha * (10 - score_dis)
    return score

def max_index(rectangle):
    rectangle = foothold_score(rectangle)
    row = np.argmax(rectangle) // rectangle.shape[1]
    col = np.argmax(rectangle) % rectangle.shape[1]
    return [row, col]

def foothold_selection( img, path, grid_size):
    for i in range(len(path[0])-1):

        s = grid_size
        y1, y2 = path[1][i], path[1][i+1]
        if y1 > y2: angle = -45
        elif y1 < y2: angle = 45
        else: angle = 0
        pt1_alt = [path[0][i+1]-s*sin(angle),path[1][i+1]-s*cos(angle)]
        pt2_alt = [path[0][i], path[1][i]]
        pt3_alt = [path[0][i+1], path[1][i+1]]
        pt4_alt = [path[0][i]+s*sin(angle),path[1][i]+s*cos(angle)]

        # print(pt1,'  ', pt2)
        # print(pt3,' ',pt4)
        height = img.shape[0]  # 原始图像高度
        width = img.shape[1]  # 原始图像宽度
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        # heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        # widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
        # rotateMat[0, 2] += (widthNew - width) / 2
        # rotateMat[1, 2] += (heightNew - height) / 2
        # imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
        imgRotation = cv2.warpAffine(img, rotateMat, (width, height), borderValue=(255, 255, 255))
        # cv2.imshow('rotateImg2', imgRotation)
        # cv2.waitKey(0)
        # pt1 = pt1 + [-height / 2, -width / 2]
        # pt2 = pt2 + [-height / 2, -width / 2]
        # pt3 = pt3 + [-height / 2, -width / 2]
        # pt4 = pt4 + [-height / 2, -width / 2]

        # 计算旋转后图像的四点坐标
        pt1, pt2, pt3, pt4 = pt1_alt.copy(), pt2_alt.copy(), pt3_alt.copy(), pt4_alt.copy()
        [[pt1[1]], [pt1[0]]] = np.dot(rotateMat, np.array([[pt1_alt[1]], [pt1_alt[0]], [1]]))
        [[pt2[1]], [pt2[0]]] = np.dot(rotateMat, np.array([[pt2_alt[1]], [pt2_alt[0]], [1]]))
        [[pt3[1]], [pt3[0]]] = np.dot(rotateMat, np.array([[pt3_alt[1]], [pt3_alt[0]], [1]]))
        [[pt4[1]], [pt4[0]]] = np.dot(rotateMat, np.array([[pt4_alt[1]], [pt4_alt[0]], [1]]))
        # print(pt1,'  ', pt2)
        # print(pt3,'  ',pt4)
        # pt1 = pt1 + [height / 2, width / 2]
        # pt2 = pt2 + [height / 2, width / 2]
        # pt3 = pt3 + [height / 2, width / 2]
        # pt4 = pt4 + [height / 2, width / 2]
        # cv2.circle(imgRotation, (int(pt2[1]),int(pt2[0])), 5, (0, 0, 255))
        # cv2.circle(imgRotation, (int(pt2[1]),int(pt2[0])), 5, (0, 0, 255))
        # rect1 = imgRotation[int(pt1[0]):int(pt2[0]), int(pt1[1]):int(pt2[1])]
        # rect2 = imgRotation[int(pt3[0]):int(pt4[0]), int(pt3[1]):int(pt4[1])]
        #cv2.rectangle(imgRotation, (int(pt1[1]),int(pt1[0])),(int(pt2[1]),int(pt2[0])),(255,0,0),1)
        #cv2.rectangle(imgRotation, (int(pt3[1]),int(pt3[0])),(int(pt4[1]),int(pt4[0])),(255,0,0),1)
        # 计算四个矩形的顶点的坐标，并且截取出矩形
        rect_w, rect_h = s, int(pt2[0]-pt3[0])//2
        pt1[0], pt1[1], pt3[0], pt3[1] = int(pt1[0]), int(pt1[1]), int(pt3[0]), int(pt3[1])
        pt2[0], pt2[1] = int(pt2[0]), int(pt2[1])
        rect_lf = imgRotation[pt1[0]:pt1[0]+rect_h, pt1[1]:pt1[1]+rect_w, 0]
        rect_rf = imgRotation[pt3[0]:pt3[0] + rect_h, pt3[1]:pt3[1] + rect_w, 0]
        rect_lb = imgRotation[pt1[0]+rect_h:pt1[0]+2*rect_h, pt1[1]:pt1[1]+rect_w, 0]
        rect_rb = imgRotation[pt3[0]+rect_h:pt3[0]+2*rect_h, pt3[1]:pt3[1]+rect_w, 0]
        # rect_lf = imgRotation[pt2[0]-2*rect_h:pt2[0]-rect_h, pt2[1]-rect_w:pt2[1], 0]
        # rect_rf = imgRotation[pt2[0]-2*rect_h:pt2[0] - rect_h, pt2[1]:pt2[1] + rect_w, 0]
        # rect_lb = imgRotation[pt2[0]-rect_h:pt2[0], pt2[1]-rect_w:pt2[1], 0]
        # rect_rb = imgRotation[pt2[0]-rect_h:pt2[0], pt2[1]:pt2[1] + rect_w, 0]
        # 找出最大的哪个值对应的位置 TODO： 这里需要考虑更多的因素:通行性+距离

        foot_lf, foot_rf, foot_lb, foot_rb = max_index(rect_lf),max_index(rect_rf),max_index(rect_lb),max_index(rect_rb)
        foot_lf = [foot_lf[0]+pt1[0], foot_lf[1]+pt1[1]]
        foot_rf = [foot_rf[0]+pt3[0], foot_rf[1]+pt3[1]]
        foot_lb = [foot_lb[0] + pt1[0]+rect_h, foot_lb[1] + pt1[1]]
        foot_rb = [foot_rb[0] + pt3[0]+rect_h, foot_rb[1] + pt3[1]]


        # 旋转回去
        if angle!=0:
            rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1)
            [[foot_lf[1]], [foot_lf[0]]] = np.dot(rotateMat, np.array([[foot_lf[1]], [foot_lf[0]], [1]]))
            [[foot_rf[1]], [foot_rf[0]]] = np.dot(rotateMat, np.array([[foot_rf[1]], [foot_rf[0]], [1]]))
            [[foot_lb[1]], [foot_lb[0]]] = np.dot(rotateMat, np.array([[foot_lb[1]], [foot_lb[0]], [1]]))
            [[foot_rb[1]], [foot_rb[0]]] = np.dot(rotateMat, np.array([[foot_rb[1]], [foot_rb[0]], [1]]))
        cv2.circle(img, (int(foot_lf[1]), int(foot_lf[0])), 3, (0, 255, 255), -3)
        cv2.circle(img, (int(foot_rf[1]), int(foot_rf[0])), 3, (255, 255, 0), -3)
        cv2.circle(img, (int(foot_lb[1]), int(foot_lb[0])), 3, (0, 255, 255), -3)
        cv2.circle(img, (int(foot_rb[1]), int(foot_rb[0])), 3, (255, 255, 0), -3)
        imgRotation = cv2.warpAffine(imgRotation, rotateMat, (width, height), borderValue=(255, 255, 255))

        # cv2.rectangle(imgRotation, ((pt1[1]),(pt1[0])),((pt1[1]+rect_w),(pt1[0]+rect_h)),(255,255,255),1)
        # cv2.rectangle(imgRotation, ((pt3[1]),(pt3[0])),((pt3[1] + rect_w),(pt3[0] + rect_h)),(255,255,255),1)
        # cv2.rectangle(imgRotation, ((pt1[1]), (pt1[0]+rect_h)), ((pt1[1]+rect_w), (pt1[0]+2*rect_h)),
        #               (255, 0, 0), 1)
        # cv2.rectangle(imgRotation, ((pt3[1]), (pt3[0]+rect_h)), ((pt3[1] + rect_w), (pt3[0] + 2*rect_h)),
        #               (255, 0, 0), 1)
    return  img


def path_planning(map, img):
    # start and goal position
    sx = 512  # [m]
    sy = 512  # [m]
    gx = 256  # [m]
    gy = 512  # [m]
    grid_size = 16  # [m]
    robot_radius = 4 # [m]

    size = map.shape
    map = cv2.resize(map,(size[1]//grid_size,size[0]//grid_size))
    map = cv2.resize(map,(size[1],size[0]),interpolation=cv2.INTER_NEAREST)

    map = 255 - map

    if show_animation:  # pragma: no cover

        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(grid_size, robot_radius, map)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)
    rx = rx[::-1]
    ry = ry[::-1]

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = foothold_selection(img, [rx, ry], grid_size)
    cv2.circle(img,(sy,sx),7,(0,0,0),-7)
    cv2.circle(img,(gy,gx),7,(0,0,0),-7)
    for i in range(len(rx)-1):
        cv2.line(img, (ry[i],rx[i]),(ry[i+1],rx[i+1]),(80,80,255),1)
    return img


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 512  # [m]
    sy = 512  # [m]
    gx = 256  # [m]
    gy = 512  # [m]
    grid_size = 16  # [m]
    robot_radius = 4 # [m]

    img = cv2.imread("test_img/test.png")
    size = img.shape
    # img = cv2.resize(img,(size[1]//grid_size,size[0]//grid_size))
    # img = cv2.resize(img,(size[1],size[0]),interpolation=cv2.INTER_NEAREST)

    map = 255 - img[:,:,0]

    if show_animation:  # pragma: no cover

        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(grid_size, robot_radius, map)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)
    rx = rx[::-1]
    ry = ry[::-1]
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()

    img = foothold_selection(img, [rx, ry], grid_size)

    cv2.circle(img,(sy,sx),5,(0,0,255))
    cv2.circle(img,(gy,gx),5,(0,0,255))
    for i in range(len(rx)-1):
        cv2.line(img, (ry[i],rx[i]),(ry[i+1],rx[i+1]),(80,80,255),1)
    cv2.imwrite('test_img/result.png', img)
    # cv2.imshow('test',img)
    # cv2.waitKey()

if __name__ == '__main__':
    main()
