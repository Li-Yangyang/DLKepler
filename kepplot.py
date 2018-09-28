import numpy as np
import collections

def median_filter(x, y, num_bins, bin_width=None, x_min=None, x_max=None):
  """Computes the median y-value in uniform intervals (bins) along the x-axis.
  The interval [x_min, x_max) is divided into num_bins uniformly spaced
  intervals of width bin_width. The value computed for each bin is the median
  of all y-values whose corresponding x-value is in the interval.
  NOTE: x must be sorted in ascending order or the results will be incorrect.
  Args:
    x: 1D array of x-coordinates sorted in ascending order. Must have at least 2
        elements, and all elements cannot be the same value.
    y: 1D array of y-coordinates with the same size as x.
    num_bins: The number of intervals to divide the x-axis into. Must be at
        least 2.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
        than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
        than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
        greater than x_min. Defaults to max(x).
  Returns:
    1D NumPy array of size num_bins containing the median y-values of uniformly
    spaced bins on the x-axis.
  Raises:
    ValueError: If an argument has an inappropriate value.
  """
  if num_bins < 2:
    raise ValueError("num_bins must be at least 2. Got: %d" % num_bins)

  # Validate the lengths of x and y.
  x_len = len(x)
  if x_len < 2:
    raise ValueError("len(x) must be at least 2. Got: %s" % x_len)
  if x_len != len(y):
    raise ValueError("len(x) (got: %d) must equal len(y) (got: %d)" % (x_len,
                                                                       len(y)))

  # Validate x_min and x_max.
  x_min = x_min if x_min is not None else x[0]
  x_max = x_max if x_max is not None else x[-1]
  if x_min >= x_max:
    raise ValueError("x_min (got: %d) must be less than x_max (got: %d)" %
                     (x_min, x_max))
  if x_min > x[-1]:
    raise ValueError(
        "x_min (got: %d) must be less than or equal to the largest value of x "
        "(got: %d)" % (x_min, x[-1]))

  # Validate bin_width.
  bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
  if bin_width <= 0:
    raise ValueError("bin_width must be positive. Got: %d" % bin_width)
  if bin_width >= x_max - x_min:
    raise ValueError(
        "bin_width (got: %d) must be less than x_max - x_min (got: %d)" %
        (bin_width, x_max - x_min))

  bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

  # Bins with no y-values will fall back to the global median.
  result = np.repeat(np.median(y), num_bins)

  # Find the first element of x >= x_min. This loop is guaranteed to produce
  # a valid index because we know that x_min <= x[-1].
  x_start = 0
  while x[x_start] < x_min:
    x_start += 1

  # The bin at index i is the median of all elements y[j] such that
  # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
  # bin i.
  bin_min = x_min  # Left endpoint of the current bin.
  bin_max = x_min + bin_width  # Right endpoint of the current bin.
  j_start = x_start  # Inclusive left index of the current bin.
  j_end = x_start  # Exclusive end index of the current bin.

  for i in range(num_bins):
    # Move j_start to the first index of x >= bin_min.
    while j_start < x_len and x[j_start] < bin_min:
      j_start += 1

    # Move j_end to the first index of x >= bin_max (exclusive end index).
    while j_end < x_len and x[j_end] < bin_max:
      j_end += 1

    if j_end > j_start:
      # Compute and insert the median bin value.
      result[i] = np.median(y[j_start:j_end])

    # Advance the bin.
    bin_min += bin_spacing
    bin_max += bin_spacing

  return result


def split(all_time, all_flux, gap_width=0.75):
  """Splits a light curve on discontinuities (gaps).
  This function accepts a light curve that is either a single segment, or is
  piecewise defined (e.g. split by quarter breaks or gaps in the in the data).
  Args:
    all_time: Numpy array or list of numpy arrays; each is a sequence of time
        values.
    all_flux: Numpy array or list of numpy arrays; each is a sequence of flux
        values of the corresponding time array.
    gap_width: Minimum gap size (in time units) for a split.
  Returns:
    out_time: List of numpy arrays; the split time arrays.
    out_flux: List of numpy arrays; the split flux arrays.
  """
  # Handle single-segment inputs.
  # We must use an explicit length test on all_time because implicit conversion
  # to bool fails if all_time is a numpy array, and all_time.size is not defined
  # if all_time is a list of numpy arrays.
  if len(all_time) > 0 and not isinstance(all_time[0], collections.Iterable):  # pylint:disable=g-explicit-length-test
    all_time = [all_time]
    all_flux = [all_flux]

  out_time = []
  out_flux = []
  for time, flux in zip(all_time, all_flux):
    start = 0
    for end in range(1, len(time) + 1):
      # Choose the largest endpoint such that time[start:end] has no gaps.
      if end == len(time) or time[end] - time[end - 1] > gap_width:
        out_time.append(time[start:end])
        out_flux.append(flux[start:end])
        start = end

  return out_time, out_flux

def phase_fold_time(time, period, t0):
  """Creates a phase-folded time vector.
  result[i] is the unique number in [-period / 2, period / 2)
  such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.
  Args:
    time: 1D numpy array of time values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.
  Returns:
    A 1D numpy array.
  """
  half_period = period / 2
  result = np.mod(time + (half_period - t0), period)
  result -= half_period
  return result


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  normalize=False): #True):
  """Generates a view of a phase-folded light curve using a median filter.
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  view = median_filter(time, flux, num_bins, bin_width, t_min,
                                     t_max)

  if normalize:
    view -= np.median(view)
    view /= np.abs(np.min(view))

  return view


def global_view(time, flux, period, num_bins=2000):
  """Generates a 'global view' of a phase folded light curve.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period /num_bins,
      t_min=-period / 2,
      t_max=period / 2)


def local_view(time,flux, period, duration,\
               num_bins=100):
  """Generates a 'local view' of a phase folded light curve.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(time,flux,\
      num_bins=num_bins,\
      bin_width=duration*3.0/24.0/num_bins,\
      t_min=-duration*1.5/24.0,\
      t_max=duration*1.5/24.0)
