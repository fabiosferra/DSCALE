import numpy as np
import fnmatch
import re
from typing import Callable, Dict, List, Optional, Union, List, Dict
import difflib
from downscaler.utils_dictionary import fun_sort_dict_by_key_length


def fun_sort_list_order_based_on_element_name(mylist: list, element: str) -> list:
    """This functions sorts a list (`mylist`) by moving elements containing `element` (or parts of `element` string) to first.
    If `element` is not found in `mylist` it will return the initial list (`mylist`)
    NOTE: Element will be splitted by using delimiters: "-", "_", " ", "|", '/', '.'

    Example 1
    ---------
        fun_sort_list_order_based_on_element_name([ 'WITCH 5.0', 'MESSAGE 1.1', 'IMAGE 3.0'], "MEssAGE")
        returns -> ['MESSAGE 1.1', 'WITCH 5.0', 'IMAGE 3.0'], because 'MESSAGE 1.1' is the best match (for "MEssAGE")
    Example 2:
    ---------
        fun_sort_list_order_based_on_element_name([ 'WITCH 5.0', 'MESSAGE 1.1', 'IMAGE 3.0'], "MESSAGE IMA")
        returns -> ['MESSAGE 1.1', 'IMAGE 3.0', 'WITCH 5.0'],  because 'MESSAGE 1.1' is the best match, followed by 'IMAGE 3.0'
    Example 3:
        fun_sort_list_order_based_on_element_name([ 'WITCH 5.0', 'MESSAGE 1.1', 'IMAGE 3.0'], "MESSAGE IMAAAAAAAAAA")
        returns -> ['MESSAGE 1.1', 'IMAGE 3.0', 'WITCH 5.0'],  because 'MESSAGE 1.1' is still the best match, followed by 'IMAGE 3.0'

    Parameters
    ----------
    mylist : list
        List that you want to sort
    element : str
        Element that you are looking for (to be moved to first in the list)

    Returns
    -------
    list
        Updated list
    """
    if type(mylist) is not list:
        raise ValueError(f"mylist needs to be a list, you passed: {type(mylist)}")
    delimiters = "-", "_", " ", "|", "/", ".", ","
    regex_pattern = "|".join(map(re.escape, delimiters))
    targerlist_dict = {}
    for model_split in re.split(regex_pattern, element):
        targetlist = []
        # NOTE: below we use `len(model_split)//4` -> so that we have at maximum 5 loops
        for s in range(0, len(model_split), max(1, len(model_split) // 4)):
            # min_characters = min(len(model_split), 2)
            modeltarget = model_split[:-s] if s >= 2 else model_split
            targetlist = [x for x in mylist if modeltarget.upper() in str(x).upper()]
            if len(targetlist):
                targerlist_dict[modeltarget] = targetlist[0]
                break

    # Sort elements in `targerlist_dict` based on best match (lenght of the key)
    # Example: fun_sort_list_order_based_on_element_name([ 'WITCH 5.0', 'MESSAGE 1.1', 'IMAGE 3.0'], "MESSAGE WIT") -> ['MESSAGE 1.1', 'WITCH 5.0', 'IMAGE 3.0'], because 'MESSAGE 1.1' is the best match, followed by 'WITCH 5.0'

    targetlist_all = list(fun_sort_dict_by_key_length(targerlist_dict).values())
    if not targetlist_all:
        print(f"No match found for {element}, initial list is returned")
    return targetlist_all + [x for x in mylist if x not in targetlist_all]


def fun_find_previous_next_item(
    target_value: Union[int, str], mylist: List[Union[int, str]]
) -> tuple:
    """
    Finds the previous and next items relative to the target value in a list.

    Parameters:
    - target_value (Union[int, str]): The value to find in the list.
    - mylist (List[Union[int, str]]): The list to search.

    Returns:
    tuple[str, Optional[Union[int, str]]]: A tuple containing 'previous' and 'next' items.
    """
    result = {"previous": None, "next": None}

    for index, obj in enumerate(mylist):
        if obj == target_value:
            if index > 0:
                result["previous"] = mylist[index - 1]
            if index < len(mylist) - 1:
                result["next"] = mylist[index + 1]
            break  # Stop searching after finding the target value

    return tuple(result.values())


def fun_fuzzy_match(
    possibilities: list,
    word: str,
    case_sensitive: bool = False,
    n: int = 4,
    cutoff: float = 0.4,
) -> list:
    """
    Find close matches for a word in a list of possibilities.

    Parameters
    ----------
    possibilities : list
        A list of sequences against which to match `word` (typically a list of strings).
    word : str
        The string for which close matches are needed.
    case_sensitive : bool, optional
        If True, performs case-sensitive matching, by default False.
    n : int, optional
        The maximum number of matches to return, by default 4.
    cutoff : float, optional
        The minimum similarity score for a match to be considered (ranges from 0 to 1), by default 0.4.


    Example usage
    -------------
    possibilities = ["apple", "banana", "cherry", "grape"]
    word = "aple"
    matches = fun_fuzzy_match(possibilities, word, n=2, cutoff=0.3, case_sensitive=False)

    Returns
    -------
    list
        A list of close matches sorted by similarity score, or an empty list if no matches.
    """
    poss_dict = {x if case_sensitive else x.upper(): x for x in possibilities}
    word = word if case_sensitive else word.upper()
    res = difflib.get_close_matches(word, poss_dict, n, cutoff)
    return [poss_dict[x] for x in res] if res else []


def fun_wildcard(
    list_of_wildcard_patterns: List[str], original_list: List[str]
) -> List[str]:
    """Filters a list based on a list of wildcard patterns.
    Example: fun_wildcard(['*SSP1*'], ['SSP1-NDC', 'SSP1-NPi', 'SSP2-1p5C']) ->['SSP1-NDC', 'SSP1-NPi']

    Parameters
    ----------
    list_of_wildcard_patterns : List[str]
        list of wildcard patterns e.g. [*SSP1*,'*SSP2*']
    original_list : List[str]
        List to be filtered e.g. ['SSP1_low', 'SSP1_high', 'SSP3_low', 'SSP2_high']

    Returns
    -------
    List[str]
        Filtered list
    """
    return [
        path
        for path in original_list
        if any(fnmatch.fnmatch(path, pat) for pat in list_of_wildcard_patterns)
    ]

def fun_flatten_list_recursive(
    lst: List[Union[List, int, str]], unique: bool = False
) -> List[Union[int, str]]:
    
    """
    Flattens a nested list.
    
    Parameters
    ----------
    lst: List[Union[List, int, str]]
        The nested list to flatten.
    unique: bool
        If True, removes duplicate elements from the flattened list. Default is False.

    Returns
    -------
    List:[Union[int, str]]
        The flattened list.
    """
    flat_set, result = set(), [] # Initialize an empty set for uniqueness and an empty list for the result.

    def flatten_recursive(sublist):
        nonlocal flat_set, result # Declare nonlocal variables to modify the outer function's variables.
        [
            flatten_recursive(item)
            if isinstance(item, list) # If the item is a list, recursively call flatten_recursive.
            else (result.append(item), flat_set.add(item))
            if unique and item not in flat_set # If unique and not in set, append and add to set.
            else result.append(item) # If not unique or already in set, just append.
            for item in sublist # Iterate over each item in the sublist.
        ]

    flatten_recursive(lst) # Start the recursion with the initial list.
    return list(set(result)) if unique else result # Return the flattened list.

def group_elements(lst: list, x: int) -> list:
    """
    Group elements of a list into sublists of a specified size.

    This function takes a list and an integer `x`, and returns a new list where the elements 
    are grouped into sublists of size `x`. If the number of elements in the list is not 
    perfectly divisible by `x`, the last sublist will contain the remaining elements.

    Parameters
    ----------
    lst : list
        The list of elements to be grouped.
    x : int
        The size of the sublists.

    Returns
    -------
    list
        A new list where the elements are grouped into sublists of size `x`.

    Examples
    --------
    >>> group_elements(["a", "b", "c", "d"], 2)
    [['a', 'b'], ['c', 'd']]
    
    >>> group_elements(["a", "b", "c", "d", "e"], 3)
    [['a', 'b', 'c'], ['d', 'e']]
    """
    if x <= 0:
        raise ValueError("The group size must be a positive integer.")
    return [lst[i:i+x] for i in range(0, len(lst), x)]