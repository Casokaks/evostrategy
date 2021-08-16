"""
evostrategy
==================================
Collection of usefull functions

Author: 
Andrea Casati, andrea1.casati@gmail.com
Casokaks (https://github.com/Casokaks/)

Created on: Nov 26th 2018

"""


def dedup_list(my_list):
    """Remove duplicates from a list, while keeping the order. Especially useful for list of lists"""
    new_list = []
    for elem in my_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


