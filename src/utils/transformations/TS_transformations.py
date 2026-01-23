import numpy as np
from scipy import interpolate, signal
from scipy.fft import fft, ifft


class AugmentationMap:
    def __init__(self):
        self._registry = {}

    def register(self, name, transform_class):
        self._registry[name] = transform_class

    def get(self, name):
        return self._registry.get(name)


class Jittering:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        return x + np.random.normal(loc=0., scale=self.std, size=x.shape)
        # return x + torch.randn(x.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Scaling:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        # self.mean = mean

    def __call__(self, x):
        n_scale = np.random.normal(loc=1, scale=self.sigma, size=(x.shape[0], x.shape[1]))
        # n_scale = torch.randn(x.size()) * self.std + self.mean
        return x * n_scale


class Flipping:
    def __init__(self, axis=1):
        self.axis = axis  # on the direction of time

    def __call__(self, x):
        return np.flip(x, axis=self.axis).copy()


# will add more augmentations for time series in our benchmarking study

class Permutation:
    """This function is for univariate, equal segmentation, and random permuation.
    But it's easy to expand to more modes."""

    def __init__(self, n_segments=5):  # , pertub_mode="random", seg_mode ="equal"
        self.n_segments = n_segments

    def __call__(self, x):
        T = x.shape[1]
        sublength = int(T / self.n_segments)
        augmented = np.zeros_like(x)
        idx = np.random.permutation(self.n_segments)  # random shuffling the order
        for i in range(self.n_segments):
            j = idx[i]
            augmented[:, i * sublength: (i + 1) * sublength] = x[:, j * sublength: (j + 1) * sublength]
        return augmented


class Resizing:
    def __init__(self, cut_ratio=0.5):
        self.cut_ratio = cut_ratio

    def __call__(self, x):
        # orig_steps = np.arange(x.shape[1])
        T = x.shape[1]
        length = int(T * self.cut_ratio)
        start_step = np.random.choice(T - length)
        #         print(start_step)
        t_warped = np.zeros_like(x)

        for dim in range(x.shape[0]):
            # "nearest","zero","slinear","quadratic","cubic"
            interp_func = interpolate.interp1d(np.linspace(0, length - 1, length),
                                               x[:, start_step:length + start_step][dim], kind='slinear')
            time_wrap = interp_func(np.linspace(0, length - 1, T))
            t_warped[dim, :] = time_wrap
        return t_warped


class TimeMasking:
    def __init__(self, pertub_ratio=0.5):
        self.pertub_ratio = pertub_ratio

    def __call__(self, x):
        mask = np.random.choice([0, 1], size=(x.shape[1]), p=[self.pertub_ratio, (1 - self.pertub_ratio)])
        return x * mask


class Freq_RandomMasking:
    def __init__(self, pertub_ratio=0.5):
        self.pertub_ratio = pertub_ratio

    def __call__(self, x):
        mask = np.random.choice([0, 1], size=(x.shape[1]), p=[self.pertub_ratio, (1 - self.pertub_ratio)])
        freq_spectrum = fft(x)
        freq_random_masked_sample = ifft(freq_spectrum * mask)
        freq_random_masked_sample = freq_random_masked_sample.real
        return freq_random_masked_sample


class Filtering:
    def __init__(self, fs=125, low=0.1, high=5, order=4):
        self.fs = fs
        self.low = low
        self.high = int(np.random.choice(np.arange(high, int(fs / 2) - 1), size=1))
        self.order = order

    def __call__(self, x):
        b, a = signal.butter(self.order, [self.low * 2 / self.fs, self.high * 2 / self.fs], 'bandpass')
        freq_masked_sample = signal.filtfilt(b, a, x)
        return freq_masked_sample.copy()


class ChannelWise_neighboring:
    def __init__(self, channel_selection=None):
        self.selected_c = channel_selection

    def __call__(self, x):
        return x[self.selected_c]


class Timewise_neighboring:
    def __init__(self, seg_start=None, seg_end=None):
        self.seg_start = seg_start
        self.seg_end = seg_end

    def __call__(self, x):
        old_length = x.shape[1]
        timewise_cut = x[:, self.seg_start:self.seg_end]
        new_length = timewise_cut.shape[1]

        twise_resize = np.zeros_like(x)

        for dim in range(x.shape[0]):
            # "nearest","zero","slinear","quadratic","cubic"
            interp_func = interpolate.interp1d(np.linspace(0, new_length - 1, new_length),
                                               timewise_cut[dim], kind='slinear')
            time_wrap = interp_func(np.linspace(0, new_length - 1, old_length))
            twise_resize[dim, :] = time_wrap
        return twise_resize


augmap = AugmentationMap()
augmap.register('Jittering', Jittering)
augmap.register('TimeMasking', TimeMasking)
