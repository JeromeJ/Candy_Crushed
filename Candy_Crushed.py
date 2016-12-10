#!/usr/bin/python3
# -*- coding: utf-8 -*-

# TODO : Test with a grid that has a different number of rows and cols.

__author__ = 'JeromeJ'
__website__ = 'http://www.olissea.com/'

# Resource:
# https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching

print('Candy Crushed')
print()
print('Loading the modules can take up to a few seconds...')

from collections import namedtuple
import enum
import itertools
import operator
import time
import re

import pyscreenshot
import cv2
import numpy

import pdb

print('Modules loaded successfuly!')
print()

catch_all = True
take_screenshot = False
display_grid = False

# width, height
standard_cell = (30, 30)  # Later changed to a StandardCell namedtuple

candy_book = {
    'blue':         ('bluecandy',       .7),
    'green':        ('greencandy',      .8),
    'orange':       ('orangecandy',     .6),
    'red':          ('redcandy',        .7),
    'yellow':       ('yellowcandy',     .85),
    'purple':       ('purplecandy',     .7),
    
    'black':        ('Black',           .7),
    'multicolor':   ('multicolorcandy', .7),
}

# x,y are synomyms of width, height so they can be obtained using axis.coord_from_axis(standard_cell)
standard_cell = namedtuple('StandardCell', 'x y width height')(*(standard_cell*2))

class Candy:
    # Candy can't be a namedtuple anymore because tuples are immutable
    # and we need to set col, row later on only
    def __init__(self, type, x, y, col=None, row=None):
        self.type = re.sub('(.*)candy$', '\\1', type)  # rtrim of "candy"
        self.x = x
        self.y = y
        self.col = col
        self.row = row

    def __repr__(self):
        return "Candy(type='{type}', x={x}, y={y}, col={col}, row={row})".format(**vars(self))
        # return "{name}Candy(x={x}, y={y}, col={col}, row={row})".format(name=self.type.title(), **vars(self))

    __str__ = __repr__
    
GridItem = namedtuple('GridItem', 'type')

get_x_coord = lambda candy: candy.x
get_y_coord = lambda candy: candy.y

class AllowedKeyboardInterrupt(Exception):
    pass

class Grid:
    def __init__(self, candies_pos):
        self.candies_pos = candies_pos
        self.candies = []
        self._map_candies_to_grid()

    def _get_first_colrow(self, get_coord_func, candies) -> iter:
        lowest_candy_xy = get_coord_func(
            min(candies, key=get_coord_func)
        )

        standard_cell_xy = get_coord_func(standard_cell)

        # Only keep the candies situated between lowest_candy.x and lowest_candy.x + cell_width
        # Respectively, lowest_candy.y and lowest_candy.y + cell_height
        
        # a <= x <= b
        # Is the same as
        # x - min(a, b) <= abs(a-b)

        # a <= x <= b
        # good_candies = filter(lambda candy: lowest_candy_xy <= get_coord_func(candy) <= lowest_candy_xy + standard_cell_xy, candies)

        # x - min(a-b) <= abs(a-b)
        good_candies = filter(lambda candy: get_coord_func(candy) - lowest_candy_xy <= standard_cell_xy, candies)
            
        return good_candies
    
    def _map_candies_to_axis(self, get_coord_func) -> ('candies_axis_index', 'axis_length'):
        colrow = get_coord_func(Candy('unknown', x='col', y='row'))
        
        candies = self.candies_pos[:]
        candies_axis_index = {}
            
        for axis_index in itertools.count():
            if not candies:  # No more candies to work on? We're over.
                break

            # Creating one col/row at a time
            #   (Note: This NEEDS to be a list because we modify in-place:
            #       self._get_first_colrow returns a filter object linked to candies BUT, in the
            #       loop below, we modify candies while iterating over that linked filter object.
            #   )
            axis_candies = list(self._get_first_colrow(get_coord_func, candies))

            # Removing the candies found in that col/row from the rest of the candies
            for candy in axis_candies:
                setattr(candy, colrow, axis_index)
                candies_axis_index[candy] = axis_index
                candies.remove(candy)

        return (candies_axis_index, axis_index)
    
    def _map_candies_to_grid(self):
        # Alternatively colrow_length can be calculated with max(colrow_coords.values())
        col_coords, cols_length = self._map_candies_to_axis(get_x_coord)
        row_coords, rows_length = self._map_candies_to_axis(get_y_coord)
        
        self.candies = [[None]*cols_length for _ in range(rows_length)]
        for candy in self.candies_pos:
            candy_x = row_coords[candy]
            candy_y = col_coords[candy]
            # self.candies[candy_x][candy_y] = GridItem(candy.type)
            self.candies[candy_x][candy_y] = candy

    def __getitem__(self, slice):
        return self.candies[slice]

    def __iter__(self):
        return iter(self.candies)

    def __repr__(self):
        return '\n'.join(
            ''.join(
                candy.type[0] if candy else '?'
                for candy in rows
            )
            for rows in self.candies
        )


# http://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def PIL2cv(PIL_img):
    return cv2.cvtColor(numpy.array(PIL_img), cv2.COLOR_RGB2BGR)


def catch_candy(surface, candy_type, threshold=.7):
    candy = cv2.imread('candies/{}.png'.format(candy_type))
    
    try:
        # .shape returns (rows, cols, channels)
        w, h, _ = candy.shape
    except AttributeError:
        # TODO Find a better error
        raise AttributeError('Invalid argument: can\'t load candies/{}.png'.format(candy_type))

    candies = cv2.matchTemplate(surface, candy, cv2.TM_CCOEFF_NORMED)
    loc = numpy.where(candies >= threshold)
    loc = list(zip(*loc[::-1]))  # zip(*list) switches columns and rows

    def is_valid(points, pt):
        pt_x, pt_y = pt
        for p_x, p_y in points:
            if 0 <= abs(p_x - pt_x) <= w and 0 <= abs(p_y - pt_y) <= h:
                return False
        return True
    
    # Remove duplicates
    valid_candies = []
    for pt in loc:
        if is_valid(valid_candies, pt):
            valid_candies.append(pt)

    candies = map(lambda pt: Candy(candy_type, *pt), valid_candies)
    return candies


def catch_candies(surface, candy_list):
    candies = []
    for name, threshold in candy_list:
        candies.extend(catch_candy(surface, name, threshold))
    return candies


def show_screenshot(screenshot, title='CandyCrushed Screenshot'):
    cv2.imshow(title, screenshot)

    try:
        cv2.waitKey(1)
    except KeyboardInterrupt:
        raise AllowedKeyboardInterrupt


def save(filename, img):
    cv2.imwrite(filename, img)


def show_grid(screenshot, candies):
    screenshot = screenshot[:]
    
    cluster_threshold_x = 70
    cluster_threshold_y = 63
    
    x_corr = candies[0][0]-15
    y_corr = candies[0][1]-10

    for i in range(20):
        for j in range(1, 20):
            cv2.rectangle(
                screenshot,
                (x_corr + cluster_threshold_x*i, y_corr + cluster_threshold_y*i),
                (x_corr + cluster_threshold_x*j, y_corr + cluster_threshold_y*j),
                (0, 0, 255),
                2
            )

    show_screen('The Grid', screenshot)


def highlight_candies(screenshot, candies:(Candy or ('x', 'y')), color=(255, 0, 0)):
    screenshot = screenshot[:]
    for candy in candies:
        try:
            x,y = candy.x, candy.y
        except AttributeError:
            x,y = candy
        
        cv2.rectangle(
            screenshot,
            (x, y),
            (x + standard_cell.width, y + standard_cell.height),
            color,
            2
        )
    return screenshot


if __name__ == '__main__':
    if take_screenshot:
        input('Ready for the screenshot?')
        time.sleep(1)
        print()
    
        screenshot = PIL2cv(pyscreenshot.grab())
    else:
        screenshot = cv2.imread('candy-crush-saga-screenshot-01.jpg')

    # Must never be written over: always copied before used
    blank_screenshot = screenshot.copy()
    
    if catch_all:
        print('catch_all is ON')
        
        candies = catch_candies(screenshot, candy_book.values())

        grid = Grid(candies)
        print()
        print(grid)
        print()
        
        for y, row in enumerate(grid):
            #           y-1
            # (----->)  y
            #           y+1
            #
            # Getting row by row
            
            for x, candy in enumerate(row):
                # candy1
                # ^
                # |
                # ----->
                #  |
                #  v
                #  candy2
                #
                # Iterating over the cols of that row
                
                print(candy)
                # row = grid[y]  # Already set
                col = list(zip(*grid))[x]
                
                if candy is None:
                    unknown_x = list(filter(None, col))[0].x
                    unknown_y = list(filter(None, row))[0].y
                    # No need to take the most top/left
                    # min(list(filter(None, col)), key=get_x_coord).x,  # Get the most left x coord on current col
                    # min(list(filter(None, row)), key=get_y_coord).y,  # Get the most top y coord on current row
                    
                    show_screenshot(
                        highlight_candies(
                            blank_screenshot.copy(),
                            [(unknown_x, unknown_y)],
                            (0,0,255)
                        )
                    )

                    # If we, somehow, can keep track of what is where without needing to take another screenshot,
                    # then we don't actually need to capture this again once we know what it is and where it goes/went.

                    # Valid names:
                    # - i   (stands for ignore)
                    # - bg  (stands for background)
                    # - h_purple (for horizontal stripped purple candy) (v_* for vertical)
                    # - l_purple (for locked purple candy)
                    # - cherry
                    # - nut
                    # - ...
                    
                    new_things = {
                        'i':   '?',
                        'bg':       ' background',  # Intended white-space at the beginning
                        'cherry':   'Cherry',       # Intended capital
                        'nut':      'Nut',          # Intended capital
                    }
                    
                    special_candy = '(l_)?([hv]_)?(purple|blue|green|yellow|red|orange)'

                    new_candy = None
                    while new_candy is None:
                        new_candy = input("Unknown thingy detected!! What is it? ")
                        
                        if new_candy in new_things:
                            new_candy = new_things[new_candy]
                        elif re.match(special_candy, new_candy) is None:
                            print('Name not recognized... Try again!')
                            new_candy = None

                    grid[y][x] = Candy(new_candy, unknown_x, unknown_y, col, row)

                    if new_candy != new_things['bg']:
                        # Highlight the new candy if it isn't the background
                        highlight_candies(screenshot, [grid[y][x]])
                    
                else:
                    # show_screenshot(highlight_candies(screenshot.copy(), [candy]))
                    show_screenshot(highlight_candies(screenshot, [candy]))
        
        print()
        print(grid)

        # # screenshot = highlight_candies(screenshot, candies)  # Useless as screenshot as been modified in-place
        show_screenshot(screenshot)
        
        save('result.png', screenshot)
    else:
        try:
            while True:
                try:
                    candy = input('Candy name: ')
                except KeyboardInterrupt:
                    raise AllowedKeyboardInterrupt
                
                if candy:
                    show_candies(screenshot, candy_book.get(candy, (candy, .7)))
        except AllowedKeyboardInterrupt:
            print()
            print('Bye bye~')
        except KeyboardInterrupt:
            print()
            print('ERROR: Script interrupted abruptely in the middle of a process! '
                  'Not in between two stages. Did a step take too long to execute?')
