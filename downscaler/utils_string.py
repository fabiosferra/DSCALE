import numpy as np
from typing import Callable, Dict, List, Optional, Union, List, Dict, Optional
from difflib import SequenceMatcher
import re

def fun_check_if_all_characters_are_numbers(text: str) -> bool:
    """Check whether a string can be interpreted as a number (checks if all characters in a string are numbers).
        Examples: '2010' -> True ; 'aaa'  -> False ; '2010a'-> False

    Parameters
    ----------
    text : str
        _description_

    Returns
    -------
    bool
        Whether string can be interpreted as a number
    """
    # NOTE: see alternative function here https://github.com/iiasa/downscaler_repo/issues/181
    if isinstance(text, (float, int)):
        return True
    # We replace dot otherwise "2015.0".isnumeric() => False
    return (str(text).replace(".","")).isnumeric()


def fun_sorts_list_of_strings(mylist: list, desired_order: Union[tuple, list]) -> list:
    """Sorts a list of strings (`mylist`) based on a desired list order (`desired_order`).
    It will exclude all elements that can be interpreted as numbers e.g. '2010' (by using `fun_check_if_all_characters_are_numbers`).
    Elements that are not present will be put at the end (right-end side) of `mylist`.

    Parameters
    ----------
    mylist : list
        Initial list
    desired_order : Union[tuple, list]
        Preferred elements order

    Returns
    -------
    list
        Sorted list
    """
    my_str_list = [x for x in mylist if not fun_check_if_all_characters_are_numbers(x)]
    common = [x for x in desired_order if x in my_str_list]
    not_common = [x for x in my_str_list if x not in common]
    return common + not_common


def fun_filter_list_of_strings(
    mylist: List[str], containing: List[str], keep: bool = True
) -> List[str]:
    """Filters a list of strings `mylist` to `keep` (or exclude if `keep=False`) those that contain any of the strings in a `containing` list.

    Parameters
    ----------
    mylist : List[str]
        Initial list of strings
    containing : List[str]
        Any of the strings that should be present in each element of `mylist`
    keep : bool, optional
        Whehter yuo want to keep them or exclude, by default True

    Returns
    -------
    List[str]
        Filtered list of strings
    """
    if keep:
        return [x for x in mylist if any(b in x for b in containing)]
    return [x for x in mylist if all(b not in x for b in containing)]

def fun_first_letter_in_each_word(s:str):
    """Return the first letter (upper case) of each word in a string."""
    return ''.join(word[0].upper() for word in s.split())

def find_longest_common_substring(string1: str, string2: str) -> str:
    """
    Finds the longest common substring between two input strings using SequenceMatcher.
    
    Parameters:
    -----------
    string1: str
        The first input string.
    string2: str
        The second input string.
    
    Returns:
    --------
    str
        The longest common substring found between the two strings.
    """
    # Initialize the SequenceMatcher and find the longest match
    matcher = SequenceMatcher(None, string1, string2)
    match = matcher.find_longest_match(0, len(string1), 0, len(string2))
    
    # Return the substring from string2
    return string2[match.b: match.b + match.size]


def extract_text_in_parentheses(text: str) -> Optional[str]:
    """
    Extracts the text within parentheses from a given string.

    Parameters:
    - text (str): The input string from which to extract text within parentheses.

    Returns:
    - Optional[str]: The extracted text within parentheses, or None if no parentheses are found.
    
    Example:
    >>> extract_text_in_parentheses("2050, over GDP per capita (linear)")
    'linear'
    """
    match = re.search(r'\((.*?)\)', text)
    return match.group(1) if match else None