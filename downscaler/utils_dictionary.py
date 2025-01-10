import numpy as np
from typing import Callable, Dict, List, Optional, Union, List, Dict


def fun_sort_dict_by_key_length(d: dict) -> dict:
    """Sort dictionary (`d`) by key length

    Parameters
    ----------
    d : dict
        Your dictionary

    Returns
    -------
    dict
        Sorted dictionary by key length
    """    
    return {k: d[k] for k in sorted(d, key=len, reverse=True)}


def fun_sort_dict(d: dict, by="keys", reverse=False) -> dict:
    """Returns a sorted dictionary (`d`), by keys or values.

    Parameters
    ----------
    d : dict
        Your dictionary
    by : str, optional
        Wheter you want to sort your dictionary by "keys" or "values", by default "keys"
    reverse : bool, optional
        Wheter you want to sort in reverse order, by default False

    Returns
    -------
    dict
        Sorted dictionary

    Raises
    ------
    ValueError
        If `by` is not in ['keys', 'values']
    """    
    i_dict = {"keys": 0, "values": 1}
    if by in i_dict:
        return dict(sorted(d.items(), key=lambda item: item[i_dict[by]], reverse=reverse))
    text="`by` needs to match one of the following:"
    raise ValueError(f"{text} {i_dict.keys()}. You have selected: {by}")

def sum_two_dictionaries(x:dict,y:dict, only_common_keys:bool=False)->dict:
    """Sum values of two dictionaries (`x` and `y`).
    If `only_common_keys=True` it sums up only keys in common.
    Inspired by: https://stackoverflow.com/questions/10461531/how-to-merge-and-sum-of-two-dictionaries-in-python

    Parameters
    ----------
    x : dict
        First dictionary
    y : dict
        Second dictionary
    only_common_keys : bool, optional
        Whether you want to sum only keys that they have in common, by default False

    Returns
    -------
    dict
        Dictionary with the sum of values in `x` and `y`
    """    
    if only_common_keys:
       return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) & set(y)}
    else:
        return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def sum_multiple_dictionaries(dicts: List[dict], only_common_keys: bool = False) -> dict:
    """Sum values of multiple dictionaries.

    This function sums up the values of multiple dictionaries.
    If `only_common_keys=True`, it sums up only keys that are common across all dictionaries.

    Parameters
    ----------
    dicts : List[dict]
        List of dictionaries to be summed.
    only_common_keys : bool, optional
        Whether to sum only keys that are common across all dictionaries, by default False.

    Returns
    -------
    dict
        Dictionary with the sum of values from all input dictionaries.
    """
    # Initialize a counter
    count=0
    # Iterate over each dictionary in the list
    for d in dicts:
        if count==0:
           # Initialize summed_dict as the first dictionary in your list
           summed_dict=d
        else:
            # Update the summed dictionary based on whether only_common_keys is True or False
            summed_dict = sum_two_dictionaries(summed_dict, d, only_common_keys=only_common_keys)
        count+=1
    return summed_dict

def fun_sort_dict_by_value_length(d: dict, reverse:bool=False) -> dict:
    """Sort dictionary `d` by the length of its values in ascending order (if reverse=False).

    Parameters
    ----------
    d : dict
        Dictionary to be sorted.
    reverse:bool
        Whether you want to sort in reversed order, by default False
        
    Returns
    -------
    dict
        Sorted dictionary by the length of its values.
    """
    return {k: v for k, v in sorted(d.items(), key=lambda item: len(item[1]), reverse=reverse)}


## Best match across keys in a dictionary (based on how many values the keys have in common). Find similar keys in a dictionary
# def find_most_common_keys(mydict):
# def find_key_pairs_with_most_common_values(mydict):
def find_best_key_match_by_common_values(mydict:dict)->dict:
    """Finds best match across keys in a dictionary (based on how many values they have in common).

    Parameters
    ----------
    mydict : dict
        Your dictionary

    Returns
    -------
    dict
        A dictionary with pairs of similar keys (based on how many values they have in common)
    """    
    # mydict={k:v if isinstance(v, list) else [v] for k,v in mydict.items()}
    # res={}
    # for k in mydict:
    #     best_match=list(fun_sort_dict({x:len(set(mydict[k])&set(mydict[x])) for x in mydict.keys() if x!=k}, by='values', reverse=True).keys())[:1]
    #     res[k]=best_match
    # return {k:v for k,v in res.items() if len(v)>0}
    mydict={k:v if isinstance(v, list) else [v] for k,v in mydict.items()}
    mydict={k:v+[k] for k,v in mydict.items()}
    res={}
    for k in mydict:
        best_match=fun_sort_dict({x:len(set(mydict[k])&set(mydict[x])) for x in mydict.keys() if x!=k}, by='values', reverse=True)
        best_match={k:v for k,v in best_match.items() if v!=0}
        res[k]=list(best_match.keys())[:1]
    return res
  

def fun_append_list_of_dicts(mydictlist: List[dict], values_as_list=False) -> dict:
    """
    Combine a list of dictionaries into a single dictionary.

    This function takes a list of dictionaries and combines them into a single dictionary. 
    If there are duplicate keys across the dictionaries, the value from the last dictionary 
    with that key will be used. Optionally, it can wrap each value in a list.

    Parameters
    ----------
    mydictlist : list of dict
        A list containing dictionaries to be combined.
    values_as_list : bool, optional
        If True, wraps each value in a list (if it is not already a list), by default False.

    Returns
    -------
    dict
        A single dictionary containing all key-value pairs from the input list of dictionaries.
    
    Examples
    --------
    >>> dict_list = [{'a': 1}, {'b': 2}, {'a': 3}]
    >>> fun_append_list_of_dicts(dict_list)
    {'a': 3, 'b': 2}

    >>> dict_list = [{'a': 1}, {'b': [2]}, {'a': 3}]
    >>> fun_append_list_of_dicts(dict_list, values_as_list=True)
    {'a': [3], 'b': [2]}

    Notes
    -----
    If there are duplicate keys across the dictionaries in the input list, 
    the value from the last dictionary with that key will be retained in the output.
    """
    if values_as_list:
        return {k: [v] if not isinstance(v, list) else v for d in mydictlist for k, v in d.items()}
    return {k: v for d in mydictlist for k, v in d.items()}
