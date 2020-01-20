import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    """
    Presumably, this class exists to keep track of the state
    """
    def __init__(self, bars_count: int, commission_perc: float, reset_on_close: bool, reward_on_close: bool = True,
                 volumes: bool = True):
        assert bars_count > 0
        assert commission_perc >= 0.0
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices: data.Prices, offset):
        assert offset >= self.bars_count-1
        # Do you currently hold the stock
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        """
        Obviously this returns the shape of the state,
        which depends on
        1. If volumes are included (otherwise just /open/, close, high, low) - Open not used
        2. How many "bars" are included in the observation
        The shape is (x, ) which indicates a ROW array
        :return:
        """
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3*self.bars_count + 1 + 1, )

    def encode(self) -> np.ndarray:
        """
        Convert current state into numpy array.
        We are not including the opening price in the observation
        Why? I have no idea!
        This method is called in the gym.env class that inherits this method
        :return: row array of shape self.shape
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        # it's a little strange, but as an example -9, 1 is the bar_idx range
        # and then we'll end up with observations by the offset
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        # rel_close is expressed as a movement (%/100) of opening price.
        return open * (1.0 + rel_close)

    def step(self, action: Actions):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            # TODO: Commission not properly computed
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            # TODO: Commission not properly computed
            reward -= self.commission_perc
            done |= self.reset_on_close
            # Remember: reward on close means that the agent only receives a reward when
            # the position is closed out
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        # This condition will handle when the reward is computed at every timestep
        # Hypothesis: this would converge faster
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):
    """

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, prices: dict, bars_count: int = DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close: bool = True, state_1d: bool = False,
                 random_ofs_on_reset: bool = True, reward_on_close: bool = False, volumes: bool = False):
        """
        :param prices: Appears to be a dictionary of length 1. To access data you have to go
        prices['YNDX'].open, .close, .high, .low, .volume. This might be different if state_1d is True.
        :param bars_count: Integer specifying the count of bars passed in as an observation
        :param commission: The % of the stock buy/sell price we have to pay to the broker
        :param reset_on_close: (If True) Every time the agent asks us to close the position, the episode is terminated.
        Else the episode continues until end of data
        :param state_1d: (If false) The prices are arranged as they are above. Apparently this is a convenient format
        for fully connected NN vs Conv1d.
        It doesn't appear to be used in this example.
        :param random_ofs_on_reset: (TRUE) Either we start on random data offsets (and continue to completion) for every
        episode or (FALSE) we start the episode at the beginning of the data.
        :param reward_on_close: If TRUE, agent receives reward only when closing their position.
        When FALSE, agent receives reward as a consequence of their current stock positions
        :param volumes: If True, include volume information in observation. If false, do not include
        volume information
        """
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        """
        Make selection of the instrument and it's offset. Then reset the state
        self._instrument is the exchange, and it bizarrely randomly picked
        In our example, the instrument is always YNDX (Yandex)

        prices goes 1 step down into the dictionaries hierarchy, which is a lib.data.Prices object
        bars is an int, ya' know, the number of bars being packed into a single observation

        For the most part it's just a thin wrapper around one of the two state class reset methods
        :return: Encoded reset state, np.ndarray / np.array
        """
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        """
        Once again, a very thin wrapper around State / State1D class
        returns observation, reward, done, info

        :param action_idx: int64 (64! for some reason) indicating 0: skip, 1: buy, 2: close
        :return: np.ndarray, float, bool, dict
        """
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        observation = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return observation, reward, done, info

    def render(self, mode='human', close=False):
        print('NOT IMPLEMENTED')
        pass

    def close(self):
        """
        This would normally clean up any objects
        :return:
        """
        print('NOT IMPLEMENTED')
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
