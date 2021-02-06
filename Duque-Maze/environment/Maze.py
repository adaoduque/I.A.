import os
import sys
import tkinter as tk
import numpy as np
import time
import copy
import cv2

from PIL import Image

class Maze(tk.Tk):
    def __init__(self, config, mode):
        tk.Tk.__init__(self)
        self.root   =  None
        self.frame  =  None
        self.config =  config
        self.mode   =  mode
        self.shape  =  self.config["environment"].shape

        self.init    = [0, 0, 0]
        self.current = [0, 0, 0]
        self.ids     = []
        self.lines   = 0
        self.columns = 0

        self.canvas = None
        self.colorAnimate  = 'mediumOrchid1'
        self.paletteInit   = ('white', 'white', 'white', 'white')
        self.paletteLocked = ('gray10', 'gray10', 'gray10', 'gray10',)
        self.paletteUnlocked = ('white', 'white', 'white', 'white')
        self.paletteCurrent  = ('gray80', 'gray80', 'gray80', 'gray80')
        self.palettePositiveReward = ('lime green', 'lime green', 'lime green', 'lime green')
        self.paletteNegativeReward = ('firebrick1', 'firebrick1', 'firebrick1', 'firebrick1')

        self.run()

    def run(self):
        self.frame = tk.Frame(self)
        self.frame.pack()

        # Draw canvas
        self.draw_canvas()

    def reset(self):

        # Get current state
        group, subgroup, item = self.current

        # Get color for current state
        color = self.get_color(self.config["environment"][group][subgroup][0])

        # Update current state with your default color
        self.update_color(group, subgroup, color)

        # Update current with initial state
        self.current = copy.deepcopy(self.init)

        # Get initial state
        group, subgroup, _ = self.current

        # Update initial state
        self.update_color(group, subgroup, self.paletteCurrent)

    def set_current(self, group, subgroup):
        self.current[0] = group
        self.current[1] = subgroup

    def step(self, action):
        # Get current state
        group, subgroup, _ = self.current

        # Check action
        if action == 0:
            # Top
            group -= 1
            return self.exec_action(action, group, subgroup)
        elif action == 1:
            # right
            subgroup += 1
            return self.exec_action(action, group, subgroup)
        elif action == 2:
            # Down
            group += 1
            return self.exec_action(action, group, subgroup)
        elif action == 3:
            # left
            subgroup -= 1
            return self.exec_action(action, group, subgroup)
        else:
            # Nothing
            print("Action out of range")
            sys.exit("-1")

    def exec_action(self, action, group, subgroup):
        # Get current state
        oGroup, oSubgroup, _ = self.current

        # Animate action before execute
        self.animate_action(self.ids[oGroup][oSubgroup][action], self.paletteCurrent[action])

        # Get default palette color for prev action
        color = self.get_color(self.config["environment"][oGroup][oSubgroup][0])

        if 0 <= group < self.lines and 0 <= subgroup < self.columns:
            square = self.config["environment"][group][subgroup][0]
            if square == ' ' or square == 'I':
                # Set current action
                self.set_current(group, subgroup)

                # Update current action
                self.update_color(group, subgroup, self.paletteCurrent)

                # Update prev action
                self.update_color(oGroup, oSubgroup, color)

                return self.get_observable(), self.config['rewardEachStep'], False
            elif square == '+':
                # You Won
                return self.get_observable(), self.config['rewardPositive'], True
            elif square == '-':
                # You Lose

                # Set current action
                self.set_current(group, subgroup)

                # Update current action
                self.update_color(group, subgroup, self.paletteCurrent)

                # Update prev action
                self.update_color(oGroup, oSubgroup, color)

                return self.get_observable(), self.config['rewardNegative'], True
        return self.get_observable(), self.config['rewardInvalidStep'], False

    def update_color(self, group, subgroup, color):
        c1, c2, c3, c4 = color
        id0, id1, id2, id3 = self.ids[group][subgroup]
        tk.Canvas.itemconfig(self.canvas, tagOrId=id0, fill=c1)
        tk.Canvas.itemconfig(self.canvas, tagOrId=id1, fill=c2)
        tk.Canvas.itemconfig(self.canvas, tagOrId=id2, fill=c3)
        tk.Canvas.itemconfig(self.canvas, tagOrId=id3, fill=c4)

        # Update GUI
        self.update()

    def get_color(self, param):
        if param == " ":
            return self.paletteUnlocked
        elif param == "X":
            return self.paletteLocked
        elif param == "+":
            return self.palettePositiveReward
        elif param == "I":
            return self.paletteInit
        elif param == "-":
            return self.paletteNegativeReward

    def animate_action(self, rectangle_id, c1):
        # Animate this action ?
        if self.config["animate"]:
            for i in range(0, 2):
                tk.Canvas.itemconfig(self.canvas, tagOrId=rectangle_id, fill=self.colorAnimate)
                self.update()
                time.sleep(0.1)
                tk.Canvas.itemconfig(self.canvas, tagOrId=rectangle_id, fill=c1)
                self.update()
                time.sleep(0.1)

    def check_valid_action(self, group, subgroup):
        action = 0
        if 0 <= group < self.lines and 0 <= subgroup < self.columns:
            square = self.config["environment"][group][subgroup][0]
            if square == ' ' or square == 'I' or square == '+':
                action = 1
        return action

    def get_observable(self):
        state  =  None
        if self.mode == 'train-qtable':
            state = self.get_sequential_position()
        elif self.mode == 'train-state':
            group, subgroup, item = self.current
            state    =  np.array([0, 0, 0, 0, 0])

            # Current position in array (Considering a dimension)
            state[4] = self.get_sequential_position()

            # Top
            state[0]  =  self.check_valid_action(group-1, subgroup)

            # Right
            state[1]  =  self.check_valid_action(group, subgroup+1)

            # Down
            state[2]  =  self.check_valid_action(group+1, subgroup)

            # Left
            state[3]  =  self.check_valid_action(group, subgroup-1)

            # Reshape array
            state     =  state.reshape(-1, 5)
        elif self.mode == 'train-image':
            width, height, channel  =  self.config["image_dim"]

            # Save current observable
            img_id = self.save_observable_as_png(width, height)

            # Open image
            img  =  Image.open('states/state'+img_id+'.png')

            # Image to numpy array
            ar_img  =  np.array(img)

            # Close file
            img.close()

            # Reshape array
            state = ar_img.reshape(-1, width, height, 2)

        return state

    def save_observable_as_png(self, width, height):

        group, subgroup, item = self.current

        # IMG id
        img_id = str(group)+str(subgroup)

        if not os.path.isfile('states/state'+img_id+'.png'):
            # save postscipt image
            self.canvas.postscript(file='states/state.eps')

            # Use PIL to convert to PNG
            eps_img  =  Image.open('states/state.eps')

            # Image to numpy array
            img  =  np.array(eps_img)

            # Resize image
            img  =  self.image_resize(img, width=width, height=height)

            # Convert ndarray to image PIL
            ar_img  =  Image.fromarray(img)

            # Convert to grayscale and save
            ar_img.convert('LA').save('states/state'+img_id+'.png', 'png')

            # Close file
            ar_img.close()

            # Close file
            eps_img.close()

        return img_id

    def get_sequential_position(self):
        group, subgroup, item = self.current
        state = 0
        for i in range(len(self.config["environment"])):
            for j in range(len(self.config["environment"][i])):
                if i == group and j == subgroup:
                    return state
                state += 1

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def draw_canvas(self):
        canvas_height = self.config["height"]
        canvas_width  = self.config["width"]
        self.canvas   = tk.Canvas(self.frame, width=canvas_width, height=canvas_height, background='gray75')
        self.canvas.pack()

        y = 0
        for i in self.config["environment"]:
            x = 0
            self.columns = 0
            for j in i:
                if j[0] == ' ':
                    self.draw(x=x, y=y, color=self.paletteUnlocked)
                elif j[0] == '+':
                    self.draw(x=x, y=y, color=self.palettePositiveReward)
                elif j[0] == '-':
                    self.draw(x=x, y=y, color=self.paletteNegativeReward)
                elif j[0] == 'I':
                    self.draw(x=x, y=y, color=self.paletteInit)
                    self.init[0] = self.lines
                    self.init[1] = self.columns
                    self.current = copy.deepcopy(self.init)
                else:
                    self.draw(x=x, y=y, color=self.paletteLocked)
                x += self.config["widthSquares"]*2
                self.columns += 1
            y += self.config["widthSquares"]*2
            self.lines += 1

        self.ids    =  self.ids.astype(int)
        self.ids    =  self.ids.reshape(self.shape[0], self.shape[1], 4)

    def draw(self, x, y, color):

        c1, c2, c3, c4 = color

        x1 = x / 2
        y1 = y / 2
        x2 = (self.config["widthSquares"] + x) / 2
        y2 = (self.config["widthSquares"] + y) / 2
        x3 = self.config["widthSquares"] + x / 2
        y3 = y / 2
        points = [x1, y1, x2, y2, x3, y3]
        id0 = self.canvas.create_polygon(points, fill=c1)

        x1 = self.config["widthSquares"] + x / 2
        y1 = self.config["widthSquares"] + y / 2
        x2 = (self.config["widthSquares"] + x) / 2
        y2 = (self.config["widthSquares"] + y) / 2
        x3 = self.config["widthSquares"] + x / 2
        y3 = y / 2
        points = [x1, y1, x2, y2, x3, y3]
        id1 = self.canvas.create_polygon(points, fill=c2)

        x1 = self.config["widthSquares"] + x / 2
        y1 = self.config["widthSquares"] + y / 2
        x2 = (self.config["widthSquares"] + x) / 2
        y2 = (self.config["widthSquares"] + y) / 2
        x3 = x / 2
        y3 = self.config["widthSquares"] + y / 2
        points = [x1, y1, x2, y2, x3, y3]
        id2 = self.canvas.create_polygon(points, fill=c3)

        x1 = x / 2
        y1 = y / 2
        x2 = (self.config["widthSquares"] + x) / 2
        y2 = (self.config["widthSquares"] + y) / 2
        x3 = x / 2
        y3 = self.config["widthSquares"] + y / 2
        points = [x1, y1, x2, y2, x3, y3]
        id3 = self.canvas.create_polygon(points, fill=c4)

        self.ids = np.append(self.ids, [id0, id1, id2, id3])
