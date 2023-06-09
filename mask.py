# 
# This file is part of the Gronan/Chester distribution (https://github.com/Gronan/Chester).
# Copyright (c) 2023 Ronan TREILLET.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np

class MaskGroupShouldHaveChildrenException(Exception):
    def __init__(self):
        super().__init__('MaskGroupShouldHaveChildrenException')

class MaskAbstract:
    name : str
    length : int
    has_children : bool
    np_distribution : np.array
    np_mask : np.array
    depth: int
    def __init__(self, mask_parameters, depth=0):
        pass

    def get_mask(self) -> np.array:
        if self.has_children:
            # parent's mask is the logical OR (element wise) of all the children
            return np.any([child.get_mask() for child in self.children], axis=0)
        else:
            # if mask has no children then it's a leaf of the tree then he gets it's own name in the tree
            return np.where(self.np_distribution == self.name, True, False)

    def get_probability(self, parent_probability = 1.0) -> dict:
        pass

    def set_distribution(self, cartesian_mask) -> dict:
        self.np_distribution = cartesian_mask
        np_masks_dict = {}
        if self.has_children:
            for child in self.children:
                    np_mask_dict = child.set_distribution(cartesian_mask)
                    np_masks_dict.update(np_mask_dict)
        #for each level of depth we compute a mask that is the logical OR of all the children
        self.np_mask = self.get_mask()
        if hasattr(self, 'name'):
            np_masks_dict.update({self.name: self.np_mask})
        return np_masks_dict


class MaskGroup(MaskAbstract):
    children : list[MaskAbstract]
    children_probability : dict

    def __init__(self, mask_parameters, depth=0):
        #calling parent class constructor
        super().__init__(mask_parameters, depth=depth)
        self.has_children = False
        self.children = []
        self.mask = None
        self.depth = depth

    def get_probability(self, parent_probability = 1.0) -> dict:
        if not self.has_children:
            raise MaskGroupShouldHaveChildrenException()
        return {child.name: parent_probability * child.get_probability() for child in self.children}
        

    
class MaskGroupExcludingCategories(MaskGroup):
    children : list[MaskAbstract]
    def __init__(self, mask_parameters, depth=0):
        #calling parent class constructor
        super().__init__(mask_parameters, depth=depth)
        self.has_children = True
        if 'children' not in mask_parameters:
            raise MaskGroupShouldHaveChildrenException()
        if isinstance(mask_parameters['children'], dict):
            self.children = [mask_factory_build(child, self.depth + 1) for _, child in mask_parameters['children'].items()]
        else:
            self.children = [mask_factory_build(child, self.depth + 1) for child in mask_parameters['children']]
        
    def get_probability(self, parent_probability = 1.0) -> dict:
        if not self.has_children:
            raise MaskGroupShouldHaveChildrenException()
        children_probability = [child.get_probability(parent_probability) for child in self.children]
        dict_probability = {}
        for p in children_probability:
            dict_probability.update(p)
        
        return dict_probability
        

class Mask(MaskAbstract):
    children : list[MaskAbstract]
    def __init__(self, mask_parameters, depth=0):
        #calling parent class constructor
        super().__init__(mask_parameters, depth=depth)
        self.name = mask_parameters['name']
        self.probability = mask_parameters['probability']
        self.has_children = False
        self.children = []
        self.depth = depth
        if 'children' in mask_parameters and len(mask_parameters['children']) > 0 :
            self.has_children = True
            if isinstance(mask_parameters['children'], dict):
                self.children = [mask_factory_build(child, self.depth + 1) for _, child in mask_parameters['children'].items()]
            else:
                self.children = [mask_factory_build(child, self.depth + 1) for child in mask_parameters['children']]
        
    def get_probability(self, parent_probability = 1.0) -> dict:
        
        if self.has_children:
            # parent's mask is the logical OR (element wise) of all the children
            children_probability = [child.get_probability(parent_probability * self.probability) for child in self.children]
            #return a flatten the array of dict into a dict
            dict_probability = {}
            for p in children_probability:
                dict_probability.update(p)
            
            return dict_probability
        else:
            return {self.name : parent_probability * self.probability}

    def get_mask(self) -> np.array:
        if self.has_children:
            # parent's mask is the logical OR (element wise) of all the children
            return np.any([child.get_mask() for child in self.children], axis=0)
        else:
            # if mask has no children then it's a leaf of the tree then he gets it's own name in the tree
            return np.where(self.np_distribution == self.name, True, False)

#we use simple function factory to generate a tree of masks
def mask_factory_build(mask_parameters, depth=0) -> MaskAbstract:
    if 'type' not in mask_parameters:
        return Mask(mask_parameters, depth=depth)
    if mask_parameters['type'] == 'excluding_categories':
        return MaskGroupExcludingCategories(mask_parameters, depth=depth)
    if mask_parameters['type'] == 'mask':   
        return Mask(mask_parameters, depth=depth)
