from typing import Tuple
import numpy as np
from loguru import logger


def is_in_space(point: np.ndarray, space: np.ndarray) -> None:
    """Function used to check if a given point belongs to the parameter
    space given as a parameter. Raises a ValueError if not.

    Args:
        point (np.ndarray): 1D array. The point we want to check.
        space (np.ndarray): 2D array. The space we want to check the
            given point belongs to.

    Raises:
        ValueError: Raised if the point does not belong the given space.
    """
    for dim in range(len(list(point))):
        a = min(space[dim])
        b = max(space[dim])
        if not (a <= point[dim] <= b):
            raise ValueError(f"Point not in space on dimension {dim}")


def find_cell(point: np.ndarray, space: np.ndarray, N: int) -> np.ndarray:
    """In a space with each dimension divided uniformly in N division, gives
    the indexes
    in which a given point is located.

    Args:
            point (np.ndarray): 1D array. The point to be located.
            space (np.ndarray): 2D Array. The space in which we want to
                locate the point. An array of arrays, each array in space
                is a list of the possible values for one dimension.
            N (int): Number of divisions for each dimension.

    Returns:
            np.ndarray: An array containing an index of a division for each
            dimension.
    """
    coords = []
    for dim in range(len(list(point))):
        axis = space[dim]
        x = point[dim]
        a = min(axis)
        b = max(axis)
        i = int(N*(x-a)/(b-a))
        if x == b:
            i -= 1
        coords.append(i)
    return np.array(coords)


def divide_axis(axis: np.ndarray, N: int) -> list:
    """Divides an axis in cells of equal size.

    Args:
            axis (np.ndarray): 1D array. The axis to be divided
            N (int): The numbe of cells wanted.

    Returns:
            list: 2D array. An array of arrays representing the cells.
    """
    sorted_axis = np.sort(axis)
    step = (sorted_axis[-1]-sorted_axis[0])/N
    list_of_cells = []

    tab = []
    borne_sup = sorted_axis[0] + step
    for k, val in enumerate(sorted_axis):
        if (val >= borne_sup) and (k != (len(list(sorted_axis)) - 1)):
            list_of_cells.append(list(tab))
            tab = []
            borne_sup += step
        tab.append(val)
    list_of_cells.append(tab)
    return list_of_cells


def divide_space(space: np.ndarray, N: int) -> list:
    """Divides each axis of a space and returns the divided space.

    Args:
            space (np.ndarray): 2D Array. An array of arrays representing the
                different dimension of the space. Each sub array contains all
                the values possible for the axis.
            N (int): the number of cells to divide each dimension into.

    Returns:
            list: 3D array. The divided space. Each dimension is represented
                by an array of arrays.
    """

    res = []
    for dim in range(len(list(space))):
        axis = space[dim]
        list_of_cells = divide_axis(axis, N)
        res.append(list_of_cells)
    return res


def pick_random_cell(divided_space: list) -> list:
    """Picks a random cell in a divided space and returns the index of the
        cell on each dimension.

    Args:
            divided_space (list): 3D Array. The divided space. Each dimension
                is represented by an array of arrays.

    Returns:
            list: 1D array. A list of the index of the chosen cell on each
                dimension.
    """
    chosen_cell = []
    for dim in divided_space:
        chosen_cell.append(np.random.choice(len(dim)))
    return chosen_cell


def remove_cell_from_divided_space(cell: list, divided_space: list) -> list:
    """Removes a cell descibed by its indexes from the divided space.

    Args:
            cell (list): 1D array. A list of the index of the chosen cell on
                each dimension.
            divided_space (list): 3D Array. The divided space. Each dimension
                is represented by an array of arrays.

    Returns:
            list: 3D array. The divided space without the cell.
    """
    space_left = []
    for dim, division in enumerate(cell):
        dimension = list(divided_space[dim])
        dimension.pop(division)
        space_left.append(dimension)
    return space_left


def generate_point_in_cell(chosen_cell: list, divided_space: list) -> list:
    """Takes a cell in a divided space and generate a random point in this cell.

    Args:
            chosen_cell (list): 1D array. A list of the index of the chosen
                cell on each dimension.
            divided_space (list): 3D Array. The divided space. Each dimension
                is represented by an array of arrays.

    Returns:
            list: 1D array. The generated point.
    """
    point = []
    for dim, divisions in enumerate(chosen_cell):
        dimension = list(divided_space[dim])
        possible_values = dimension.pop(divisions)
        x = np.random.choice(possible_values)
        point.append(x)
    return point


def generate_point_in_space(divided_space: list) -> Tuple[list, list]:
    """Generates a random point in a random cell in a divided space and
    removes this cell from the available space.

    Args:
        divided_space (list): 3D Array. The divided space. Each dimension
        is represented by an array of arrays.

    Returns:
            list: 1D array. The generated point.
            list: 3D array. The divided space after removing the used space.
    """
    cell = pick_random_cell(divided_space)
    space_left = remove_cell_from_divided_space(cell, divided_space)
    point = generate_point_in_cell(cell, divided_space)
    return point, space_left


def initialization(starting_point: list, space: list, N: int) -> list:
    """Generates an initialization plan taking a starting point
    into account.

    Args:
        starting_point (list): 1D array. A point we want to have in the
            initialization plan.
        space (list): 2D Array. An array of arrays representing the
            different dimensionof the space. Each sub array contains
            all the values possible for the axis.
        N (int): The number of point we want in the initialization
            plan.

    Returns:
        list: 2D array. The list of points for the initialization
            plan.
    """
    points = [list(starting_point)]
    divided_space = divide_space(space, N)
    first_cell = find_cell(starting_point, space, N)
    divided_space = remove_cell_from_divided_space(first_cell, divided_space)
    for i in range(N-1):
        point, divided_space = generate_point_in_space(divided_space)
        points.append(point)
    return points


def initialize_optimizer_from_model(
        number_of_parameters: int,
        parameter_space: np.ndarray,
        matcher,
        application_description: np.ndarray) -> np.ndarray:

    """Takes a model that predict one point for the initialization plan
    and generates the remaining points to create the initialization
    plan.

    Args:
        matcher (Model): The model used to predict the first point
            of the initialization plan. Can be any object with a predict
            method.
        application_description (np.ndarray): 1D array. The description of the
            application to optimize.
        number_of_parameters (int): The number of samples to draw.
        parameter_space (np.ndarray): 2D array. The parameter space,
            each array representing a dimension.

    Returns:
        np.ndarray: an array of size number_of_parameters *
            number_of_axis containing the parameters.
    """
    chosen_point = matcher.predict([application_description])
    logger.debug(f"Predicted point for fakeapp: {chosen_point}")
    is_in_space(chosen_point[0], parameter_space)
    points = initialization(
        chosen_point[0], parameter_space, number_of_parameters)
    logger.debug(f"Smart initialization plan: {points}")
    return np.array(points)
