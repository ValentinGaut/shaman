# Copyright 2020 BULL SAS All rights reserved
def str_to_int(x):
    if isinstance(x, str):
        try:
            if x == "0":
                value = 0
                exponent = 1
            else:
                value = int(x[:-1])
                exponent = x[-1]
            if exponent == "k":
                return value*1024
            elif exponent == "M":
                return value*1024**2
            elif exponent == "G":
                return value*1024**3
            else:
                return int(x)
        except ValueError:
            return x
    else:
        return x


def clean_input(input_data):
    """
    Clean the input dataframe.
    """
    for column in input_data.columns:
        input_data[column] = input_data[column].apply(str_to_int)
    return input_data
