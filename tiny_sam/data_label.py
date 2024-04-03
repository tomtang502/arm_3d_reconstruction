import numpy as np

color_mask = np.array([
    [0.89476749, 0.33368159, 0.4864254, 0.35],
    [0.87402995, 0.87554917, 0.23845383, 0.35],
    [0.76180614, 0.87748535, 0.94896623, 0.35],
    [0.63621036, 0.85940844, 0.29094335, 0.35],
    [0.46183132, 0.30754496, 0.56040098, 0.35],
    [0.56082841, 0.44334474, 0.39530188, 0.35],
    [0.59126498, 0.06068482, 0.39018335, 0.35],
    [0.41890977, 0.39352916, 0.24572865, 0.35],
    [0.7446586, 0.85465747, 0.57271576, 0.35],
    [0.14969804, 0.51036791, 0.33433093, 0.35],
    [0.00587782, 0.70844692, 0.04809337, 0.35]
])



class ImageLabel():
    def __init__(self):
        self.obj7_bg = {
            0: (color_mask[0], 'other (table, background, etc.)'),
            1: (color_mask[1], 'controller'),
            2: (color_mask[2], 'controller_no_cover'),
            3: (color_mask[3], 'green_roll_solder'),
            4: (color_mask[4], 'toolbox'),
            5: (color_mask[5], 'cup'),
            6: (color_mask[6], 'tape_measure'),
            7: (color_mask[7], 'red_tape_roll')
        }

        self.obj8_bg = {
            0: (color_mask[0], 'other (table, background, etc.)'),
            1: (color_mask[1], 'straight_ruler'),
            2: (color_mask[2], 'mouse'),
            3: (color_mask[3], 'volt_meter'),
            4: (color_mask[4], 'toolbox'),
            5: (color_mask[5], 'grey_battery'),
            6: (color_mask[6], 'hammer'),
            7: (color_mask[7], 'mango_lassi'),
            8: (color_mask[8], 'blue_ruler')
        }

        self.obj2_bg = {
            0: (color_mask[0], 'other (table, background, etc.)'),
            1: (color_mask[1], 'can'),
            2: (color_mask[2], 'paper_bowl'),
        }

        self.shelf = { 
            0: (color_mask[0], 'other (table, background, etc.)'),
            1: (color_mask[1], 'opened_book'),
            2: (color_mask[2], 'blue_book'),
            3: (color_mask[3], 'LEFTTWO_book_on_shelf'),
            4: (color_mask[4], 'Right_book_on_shelf'),
            5: (color_mask[5], 'shelf'),
            6: (color_mask[6], 'paper_bowl')
            }

        self.class_names = {
            '7obj': self.obj7_bg,
            '8obj': self.obj8_bg,
            '2obj': self.obj2_bg,
            'shelf': self.shelf
        }


def mask_viz(mask):
    class_img = np.ones((mask.shape[0], mask.shape[1], 4))
    h, w = mask.shape
    for i in range(h):
        for j in range(w):  
            class_img[i, j] = color_mask[mask[i, j]]
    return class_img


if __name__ == "__main__":
    mask = np.array([[0, 1], [2, 3]])
    print(mask_viz(mask))