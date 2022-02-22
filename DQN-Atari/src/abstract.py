from abc import ABC, abstractmethod
import numpy as np
import collections
from gym import spaces
from gym.core import Env

class Configurable(object):
    def __init__(self, config = None):
        self.config = self.default_config()
        if config:
            # override default config with variant
            Configurable.rec_update(self.config, config)
            # override incomplete variant with completed variant
            Configurable.rec_update(config, self.config)
        
    def update_config(self, config):
        Configurable.rec_update(self.config, config)
    
    @classmethod
    def default_config(cls):
        '''Override this function to provide the default configuration of the child class
        return : a configuration dictionary
        '''
        return {}

    @staticmethod
    def rec_update(d, u):
        '''
            Recursive update of a mapping
        :param d: a mapping
        :param u: a mapping
        :return: d updated recursively with u 
        '''

        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = Configurable.rec_update(d.get(k,{}), v)
            else:
                d[k] = v
            
        return d

class Dummy(object):
    pass

_ignored_keys = set(Dummy.__dict__.keys())

class Serializable(dict):
    """
        Automatically serialize all fields of an object to a dictionary.
    Keys correspond to field names, and values correspond to field values representation by default but are
    recursively expanded to sub-dictionaries for any Serializable field.
    """

    def to_dict(self):
        d = dict()

        for k,v in self.__dict__.items():
            if k not in _ignored_keys:
                if isinstance(v, Serializable):
                    d[k] = v.to_dict()
                else:
                    d[k] = repr(v)

        return d


def serialize(obj):
    """
        Serialize any object to a dictionary, so that it can be dumped easily to a JSON file.
     Four rules are applied:
        - To be able to recreate the object, specify its class, or its spec id if the object is an Env.
        - If the object has a config dictionary field, use it. It is assumed that this config suffices to recreate a
        similar object.
        - If the object is Serializable, use its recursive conversion to a dictionary.
        - Else, use its __dict__ by applying repr() on its values
    :param obj: an object
    :return: a dictionary describing the object
    """
    if hasattr(obj, "config"):
        d = obj.config
    elif isinstance(obj, Serializable):
        d = obj.to_dict()
    else:
        d = {key: repr(value) for (key, value) in obj.__dict__.items()}

    d['__class__'] = repr(obj.__class__)

    if isinstance(obj, Env):
        d['id'] = obj.spec.id
        d['import_module'] = getattr(obj, "import_module", None)

    return d


class AbstractAgent(Configurable, ABC):
    def __init__(self, config = None):
        super(AbstractAgent, self).__init__(config)
        self.writer = None
        self.directory = None
    
    @abstractmethod
    def record(self, state, action, reward, next_state, done, info):
        raise NotImplementedError()
    
    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    def plan(self, state):
        return [self.act(state)]

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self, filename):
        raise NotImplementedError()

    def eval(self):
        pass

    def set_writer(self, writer):
        self.writer = writer

    def set_directory(self, directory):
        self.directory = directory

    def set_time(selfm, time):
        pass

class AbstractStochasticAgent(AbstractAgent):
    def action_distribution(self, state):
        raise NotImplementedError()


class AbstractDQNAgent(AbstractStochasticAgent, ABC):
    def __init__(self, env, config = None):
        super(AbstractDQNAgent, self).__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), "Only compatible with Discrete action spaces"