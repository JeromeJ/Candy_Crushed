#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'JeromeJ'
__website__ = 'http://www.olissea.com/'

# Resource:
# https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching

print('Candy Crushed')
print()
print('Loading the modules can take up to a few seconds...')

from collections import namedtuple
import os
import pathlib
import itertools
import time
import re

import pyscreenshot
import cv2
import numpy

print('Modules loaded successfuly!')
print()

# width, height
standard_cell_size = (30, 30)

candy_book = {
    'blue':         ('bluecandy',       70),
    'green':        ('greencandy',      80),
    'orange':       ('orangecandy',     60),
    'red':          ('redcandy',        70),
    'yellow':       ('yellowcandy',     85),
    'purple':       ('purplecandy',     70),
    
    'black':        ('Black',           70),
    'multicolor':   ('multicolorcandy', 70),
}

# Valid names:
# - i   (stands for ignore)
# - bg  (stands for background)
# - h_purple (for horizontal stripped purple candy) (v_* for vertical)
# - l_purple (for locked purple candy)
# - cherry
# - nut
# - ...

new_things = {
    'i': '?',
    'bg': ' background',  # Intended white-space at the beginning
    'cherry': 'Cherry',  # Intended capital
    'nut': 'Nut',  # Intended capital
}

special_candy = '(l_)?([hv]_)?(purple|blue|green|yellow|red|orange)'

# ## CONSTANTS ##

RED = (0, 0, 255)
BLUE = (255, 0, 0)

# ## TOOLS ##


def nice_repr(obj):
    """
    Decorator to bring namedtuple's __repr__ behavior to regular classes.
    
    Source: http://stackoverflow.com/a/41600946/1524913
    """
    
    def _nice_repr(self):
        v = vars(self)
        
        # Prevent infinite looping if `vars` happens to include `self`.
        del(v['self'])
        
        return repr(namedtuple(type(self).__name__, v)(**v))

    obj.__repr__ = _nice_repr
    obj.__str__ = _nice_repr

    return obj


w, h = standard_cell_size

# (x,y) are synonyms of (w, h) so they can be obtained using axis.coord_from_axis(standard_cell)
standard_cell = namedtuple('StandardCell', 'x y width height w h')(
    w, h,
    w, h,
    w, h
)


@nice_repr
class Candy:
    # Candy can't be a namedtuple anymore because tuples are immutable
    # and we need to set col, row later on only
    
    # (1) Alternatively: type = re.sub('candy$', '', type)
    # (2) Same as `self.type = type` and so on.
    
    def __init__(self, type, x, y, col=None, row=None):
        # (1)
        if type.endswith('candy'):
            type = type[:-5]
        
        # (2)
        for k, v in vars().items():
            setattr(self, k, v)


def get_x_coord(candy):
    return candy.x


def get_y_coord(candy):
    return candy.y


class AllowedKeyboardInterrupt(Exception):
    pass


class Grid:
    def __init__(self, candies_pos):
        self.candies_pos = candies_pos
        self.candies = []
        
        self._candies2grid()

    @staticmethod
    def _get_first_colrow(f_get_coord, candies) -> iter:
        lowest_candy_xy = f_get_coord(
            min(candies, key=f_get_coord)
        )

        standard_cell_xy = f_get_coord(standard_cell)

        # Only keep the candies situated between lowest_candy.x and lowest_candy.x + cell_width
        # Respectively, lowest_candy.y and lowest_candy.y + cell_height
        
        # a <= x <= b
        # Is the same as
        # x - min(a, b) <= abs(a-b)

        # a <= x <= b
        # good_candies = filter(
        #   lambda candy: lowest_candy_xy <= f_get_coord(candy) <= lowest_candy_xy + standard_cell_xy, candies
        # )

        # x - min(a-b) <= abs(a-b)
        good_candies = filter(lambda candy: f_get_coord(candy) - lowest_candy_xy <= standard_cell_xy, candies)

        return good_candies
    
    def _candies2axis(self, f_get_coord):
        """Returns `(candies_axis_index, axis_length)`."""

        colrow = f_get_coord(Candy('unknown', x='col', y='row'))
        
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
            axis_candies = list(self._get_first_colrow(f_get_coord, candies))

            # Removing the candies found in that col/row from the rest of the candies
            for candy in axis_candies:
                setattr(candy, colrow, axis_index)
                candies_axis_index[candy] = axis_index
                candies.remove(candy)

        return candies_axis_index, axis_index
    
    def _candies2grid(self):
        # Alternatively colrow_length can be calculated with max(colrow_coords.values())
        col_coords, cols_length = self._candies2axis(get_x_coord)
        row_coords, rows_length = self._candies2axis(get_y_coord)
        
        self.candies = [[None]*cols_length for _ in range(rows_length)]
        for candy in self.candies_pos:
            candy_x = row_coords[candy]
            candy_y = col_coords[candy]
            self.candies[candy_x][candy_y] = candy

    def __getitem__(self, slice_obj):
        return self.candies[slice_obj]

    def __iter__(self):
        return iter(self.candies)

    def __repr__(self):
        return '\n'.join(
            ''.join(
                re.sub('^dynamic/', '', candy.type)[0] if candy else '?'
                for candy in rows
            )
            for rows in self.candies
        )


# http://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def pil2cv(pil_img):
    return cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGB2BGR)


def catch_candy(surface, candy_type, threshold=70):
    threshold /= 100

    candy = cv2.imread('{}{}.png'.format(CANDY_DIR, candy_type))
    
    try:
        # .shape returns (rows, cols, channels)
        w, h, _ = candy.shape
    except AttributeError:
        raise AttributeError('Invalid argument: can\'t load {}{}.png'.format(CANDY_DIR, candy_type))

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

    show_screenshot(screenshot, 'The Grid')


def highlight_candies(screenshot, candies:(Candy or ('x', 'y')), color=BLUE, permanent_ink=False):
    if not permanent_ink:
        screenshot = screenshot.copy()
    
    for candy in candies:
        try:
            x,y = candy.x, candy.y
        except AttributeError:
            x,y = candy
        
        cv2.rectangle(
            img=screenshot,
            pt1=(x, y),
            pt2=(x + standard_cell.width, y + standard_cell.height),
            color=color,
            thickness=2
        )
    
    return screenshot


def main(candy_book:dict, take_screenshot=True, catch_all=True, use_extended_candybook=True):
    """
    take_screenshot:        if False, opens a default screenshot.
    catch_all:              if False, you're prompted to chose which candy to highlight.
    use_extended_candybook: if True, let the AI try to learn and remember unknown candies. [experimental]
    """

    if take_screenshot:
        input('Ready for the screenshot?')
        time.sleep(1)
        print()
    
        screenshot = pil2cv(pyscreenshot.grab())
    else:
        screenshot = cv2.imread('candy-crush-saga-screenshot-01.jpg')
    
    blank_screenshot = screenshot.copy()

    if use_extended_candybook:
        candy_book.update(
            {name:('dynamic/{}'.format(pathlib.Path(name).stem), 70) for name in os.listdir('{}dynamic/'.format(CANDY_DIR))}
        )
    
    if catch_all:
        print('CATCH_ALL is ON')
        
        candies = catch_candies(screenshot, candy_book.values())

        grid = Grid(candies)
        print()
        print(grid)
        print()
        
        for y, row in enumerate(grid):
            for x, candy in enumerate(row):
                print(candy)

                # row = grid[y]
                # candy = grid[y][x]

                # Unknown type of candy
                if candy is None:
                    # zip(*grid) swaps cols and rows
                    col = list(zip(*grid))[x]

                    # Estimate the x,y of the unknown candy by taking the x from the col and y from the row
                    # As defined by any of its valid candy `list(filter(None, col))` (guaranteed to have at least 1)
                    unknown_x = list(filter(None, col))[0].x
                    unknown_y = list(filter(None, row))[0].y
                    
                    show_screenshot(
                        highlight_candies(
                            blank_screenshot,
                            [(unknown_x, unknown_y)],
                            RED
                        )
                    )

                    # If we, somehow, can keep track of what is where without needing to take another screenshot,
                    # then we don't actually need to capture this again once we know what it is and where it goes/went.

                    new_candy = None
                    while new_candy is None:
                        new_candy_type = input("Unknown thingy detected!! What is it? ")
                        
                        try:
                            new_candy = new_things[new_candy_type]
                        except KeyError:
                            if re.match(special_candy, new_candy_type):
                                new_candy = new_candy_type
                            else:
                                print('Invalid candy type... Try again!')

                    grid[y][x] = Candy(new_candy, unknown_x, unknown_y, col, row)

                    if new_candy != new_things['bg'] and new_candy != new_things['i']:
                        # Highlight the new candy if it isn't the background
                        highlight_candies(screenshot, [grid[y][x]], permanent_ink=True)

                        cv2.imwrite(
                            '{}/dynamic/{}.png'.format(CANDY_DIR, new_candy),

                            # Easier to crop with PIL than this. :)
                            blank_screenshot[unknown_y:unknown_y+h, unknown_x:unknown_x+w]
                        )
                else:
                    # show_screenshot(highlight_candies(screenshot.copy(), [candy]))
                    show_screenshot(highlight_candies(screenshot, [candy], permanent_ink=True))
        
        print()
        print(grid)

        # # screenshot = highlight_candies(screenshot, candies)  # Useless as screenshot has been modified in-place
        show_screenshot(screenshot)
        
        save('result.png', screenshot)
    else:
        print('(Hit Ctrl+C to stop)')
        print()

        try:
            while True:
                try:
                    candy = input('Candy name: ')
                except KeyboardInterrupt:
                    raise AllowedKeyboardInterrupt
                
                if candy:
                    candies = catch_candy(
                        screenshot,
                        candy
                        # candy_book.get(candy, (candy, .7))
                    )
                    
                    print(list(candies))
                    
                    show_screenshot(
                        highlight_candies(
                            screenshot,
                            candies
                        )
                    )
        except AllowedKeyboardInterrupt:
            print()
            print('Bye bye~')
        except KeyboardInterrupt:
            print()
            print('ERROR: Script interrupted abruptly in the middle of a process! '
                  '(Not in between two stages) Did a step take too long to execute?')

    print()
    input('Press Enter to close screenshot.')

if __name__ == '__main__':
    CANDY_DIR = 'candies/'

    main(
        candy_book,
        take_screenshot=False,
        catch_all=True,
        use_extended_candybook=True
    )