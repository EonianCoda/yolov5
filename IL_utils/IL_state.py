

import random
import copy
from collections import defaultdict
import pandas as pd
from pycocotools.coco import COCO
import numpy as np

# num_knowing_class = num_new_class + num_past_class
EMPTY_STATE = {'knowing_class':{'id':[],'name':[]},
                'new_class':{'id':[],'name':[]},
                'num_past_class':0, 
                'num_new_class':0, 
                'num_knowing_class':0}

class Enhance_COCO(COCO):
    def __init__(self, path):
        super().__init__(path)
        self.classes = defaultdict()
        self.reverse_classes = defaultdict()
        for category in self.loadCats(self.getCatIds()):
            self.classes[category['id']] = category['name']
            self.reverse_classes[category['name']] = category['id']

    def get_cats_by_imgs(self, imgIds, return_name=False):
        """ Given some image ids, and return the category in these imgs

            Args:
                imgIds: the image ids
                return_name: return the name of category or the index of category, when return_name = 'true', return names, else return indexs 
        """
        annotations = self.loadAnns(self.getAnnIds(imgIds=imgIds)) # get annotations
        #get categoryId appear in annotations
        catIds = [ann['category_id'] for ann in annotations]
        catIds = list(set(catIds))

        if not return_name:
            return catIds
        else:
            catNames = []
            #get category name from categord Id
            for catId in catIds:
                catNames.append(self.classes[catId])
            return catNames

    def get_imgs_by_cats(self, catIds):
        """ Given some indexs of category, and return all imgs having these category

            Args:
                catIds: the index of category
        """
        if type(catIds) == list:
            imgIds = set()
            for catId in catIds:
                imgIds.update(self.getImgIds(catIds=catId))
            return list(imgIds)
        else:
            return self.getImgIds(catIds=catIds)

    def catId_to_name(self, catIds):
        """Convert the indexs of categories to the name of them
            Args:
                catIds: a list of int or int, indicating the index of categories
            Return: a list contains the name of category which is given
        """
        if type(catIds) == int:
            return [self.classes[catIds]]
        else:
            names = [self.classes[catId] for catId in catIds]
            return names
                
    def catName_to_id(self, names, sort = True):
        """Convert the names of categories to the index of them
            Args:
                names: a list of str or str, indicating the nameof categories
                sort: whether the list which is returned is well-sorted, default = True
            Return: a list contains the indexs of category given
        """
        if isinstance(names, str):
            return [self.reverse_classes[names]]

        # names = list(set(names))
        
        ids = []
        for name in names:
            ids.append(self.reverse_classes[name])
        if sort:
            ids.sort()

        return ids
    
    def get_catNum_by_catId(self, catIds):
        result = {'image':[], 'object':[]}
        index = []
        catIds.sort()
        
        for catId in catIds:
            index.append(self.classes[catId])
            result['image'].append(len(self.getImgIds(catIds = catId)))
            result['object'].append(len(self.getAnnIds(catIds = catId)))

        index.append('Counts')
        result['image'].append(sum(result['image']))
        result['object'].append(sum(result['object']))
        result = pd.DataFrame(result, index=index)
        result.sort_values(by=['image'], ascending=False)
        return result

    def get_catNum_by_imgs(self, imgIds):
        #get annotations
        annotations = self.loadAnns(self.getAnnIds(imgIds=imgIds))
        #get categoryId appear in annotations
        catIds = [ann['category_id'] for ann in annotations]

        result = {'image':[], 'object':[]}

        index = self.catId_to_name(list(set(catIds)))
        #object counts
        result['object'] = np.unique(catIds, return_counts=True)[1].tolist()

        catIds = list(set(catIds))
        for catId in catIds:
            result['image'].append(len(set(self.getImgIds(catIds = catId)) & set(imgIds)))
        
        print('Counts meaning for  image is your input imgIds number')
        index.append('Counts')
        result['image'].append(len(imgIds))
        result['object'].append(sum(result['object']))
        result = pd.DataFrame(result, index=index)
        result.sort_values(by=['image'], ascending=False)
        return result

class IL_states(object):
    def __init__(self, coco_path: str, scenario_list:list):
        self.scenario = "+".join(scenario_list)
        self._init_states(Enhance_COCO(coco_path), scenario_list)

    def _init_states(self, coco_obj:Enhance_COCO, scenario_list:list, shuffle = False):
        """init incremeantal learning task
            Args:
                scenario: the incremental learning scenario
                shuffle: whether shuffle the order
        """

        self.states = dict([(idx, copy.deepcopy(EMPTY_STATE)) for idx, _ in enumerate(scenario_list)])

        classes = sorted(coco_obj.classes.values())
        if shuffle:
            random.shuffle(classes)
  
        total_num = 0
        for idx, target in enumerate(scenario_list):
            if isinstance(target, str):
                if target.isnumeric():
                    scenario_list[idx] = int(target)
                    total_num += int(target)
                else:
                    classes[total_num] = target
                    scenario_list[idx] = 1
                    total_num += 1
            elif isinstance(target, int):
                scenario_list[idx] = target
                total_num += target
                
        for idx, num in enumerate(scenario_list):
            self.states[idx]['num_new_class'] = num
            total_num += num
            # non-incremental initial state
            if idx == 0:
                self.states[idx]['new_class']['name'].extend(classes[:total_num])
                self.states[idx]['new_class']['id'] = coco_obj.catName_to_id(self.states[idx]['new_class']['name'], sort=False)
                self.states[idx]['knowing_class']['name'] = self.states[idx]['new_class']['name']
                self.states[idx]['knowing_class']['id'] = self.states[idx]['new_class']['id']
                self.states[idx]['num_knowing_class'] = num
                continue

            # incremental state
            self.states[idx]['num_past_class'] = self.states[idx - 1]['num_knowing_class']
            self.states[idx]['num_knowing_class'] = total_num

            self.states[idx]['new_class']['name'].extend(classes[total_num - num:total_num])
            self.states[idx]['new_class']['id'] = coco_obj.catName_to_id(self.states[idx]['new_class']['name'], sort=False)


            self.states[idx]['knowing_class']['name'].extend(self.states[idx - 1]['knowing_class']['name'])
            self.states[idx]['knowing_class']['name'].extend(self.states[idx]['new_class']['name'])
            self.states[idx]['knowing_class']['id'].extend(self.states[idx - 1]['knowing_class']['id'])
            self.states[idx]['knowing_class']['id'].extend(self.states[idx]['new_class']['id'])

        self.total_class_num = total_num

    def __getitem__(self, key):
        if isinstance(key, int) and key < 0:
            key = list(self.states.keys())[key]
        return self.states[key]

    def __len__(self):
        return self.total_class_num
