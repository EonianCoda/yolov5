# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
#IL utils
from IL_utils.IL_state import IL_states

class IL_cfg(object):
    def __init__(self, cfg):
        self._params = vars(cfg)
        
        self['scenario_list'] = self['scenario']
        self['scenario'] = "_".join([str(i) for i in self['scenario']])

        #TODO change path to dynamically
        path = "../dataset/voc2007/annotations/voc2007_trainval.json"
        self.states = IL_states(path, self['scenario_list'])

        if self['end_state'] == None or self['end_state'] < self['start_state']:
            self['end_state'] = self['start_state']


    def __setitem__(self, key, value):
        self._params[key] = value
    def __getitem__(self, key):
        if self._params.get(key) == None:
            return None
        else:
            return self._params[key]