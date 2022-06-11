from arviz import kde
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d, convolve, gaussian
import numpy as np

import matplotlib.pyplot as plt

import time


# this code was found elsewhere and I am makining it available for myself and anyone else
# for more https://github.com/tommyod/KDEpy


def kde2D(x, y, field, resolution=128, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    complex_number = complex(0, resolution)
    xbins = ybins = complex_number
    # x[x < 0] = 0
    # y[y < 0] = 0

    # create grid of sample locations (default: 128x128)
    xx, yy = np.mgrid[field[:, 0].min():field[:, 0].max():xbins,
                         field[:, 1].min():field[:, 1].max():ybins]

    xmin, xmax, ymin, ymax = field[:, 0].min(), field[:, 0].max(), field[:, 1].min(), field[:, 1].max()
    
    zz, _ = _fastkde(x, y, gridsize=(resolution, resolution), extents=(xmin, xmax, ymin, ymax), adjust=0.7)
    return xx, yy, zz, (xmin, xmax, ymin, ymax)

def _fastkde(x, y, gridsize=(200, 200), extents=None, nocorrelation=True, weights=None, adjust=1.):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    for computing the KDE on a regular grid

    Note that this is a different use case than scipy's original
    scipy.stats.kde.gaussian_kde

    IMPLEMENTATION
    --------------

    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    It computes the sparse bi-dimensional histogram of two data samples where
    *x*, and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).
    histogram of the data is a faster implementation than numpy.histogram as it
    avoids intermediate copies and excessive memory usage!


    This function is typically *several orders of magnitude faster* than
    scipy.stats.kde.gaussian_kde.  For large (>1e7) numbers of points, it
    produces an essentially identical result.

    Boundary conditions on the data is corrected by using a symmetric /
    reflection condition. Hence the limits of the dataset does not affect the
    pdf estimate.

    INPUTS
    ------

        x, y:  ndarray[ndim=1]
            The x-coords, y-coords of the input data points respectively

        gridsize: tuple
            A (nx,ny) tuple of the size of the output grid (default: 200x200)

        extents: (xmin, xmax, ymin, ymax) tuple
            tuple of the extents of output grid (default: extent of input data)

        nocorrelation: bool
            If True, the correlation between the x and y coords will be ignored
            when preforming the KDE. (default: False)

        weights: ndarray[ndim=1]
            An array of the same shape as x & y that weights each sample (x_i,
            y_i) by each value in weights (w_i).  Defaults to an array of ones
            the same size as x & y. (default: None)

        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.

    OUTPUTS
    -------
        g: ndarray[ndim=2]
            A gridded 2D kernel density estimate of the input points.

        e: (xmin, xmax, ymin, ymax) tuple
            Extents of g

    """
    # Variable check
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)

    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size as input x & y arrays!')

    # Optimize gridsize ------------------------------------------------------
    #Make grid and discretize the data and round it to the next power of 2
    # to optimize with the fft usage
    if gridsize is None:
        gridsize = np.asarray([np.max((len(x), 512.)), np.max((len(y), 512.))])
    gridsize = 2 ** np.ceil(np.log2(gridsize))  # round to next power of 2

    nx, ny = gridsize

    # Make the sparse 2d-histogram -------------------------------------------
    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Basically, this is just doing what np.digitize does with one less copy
    # xyi contains the bins of each point as a 2d array [(xi,yi)]
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y.
    # Exploit a sparse coo_matrix avoiding np.histogram2d due to excessive
    # memory usage with many points
    
    grid = coo_matrix((weights, xyi), shape=(int(nx), int(ny))).toarray()

    # Kernel Preliminary Calculations ---------------------------------------
    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = n ** (-1.0 / 6.) * adjust  # For 2D

    # Make the gaussian kernel ---------------------------------------------

    # First, determine the bandwidth using Scott's rule
    # (note that Silvermann's rule gives the # same value for 2d datasets)
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    print()
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the histogram with the gaussian kernel
    # use boundary=symm to correct for data boundaries in the kde
    grid = convolve2d(grid, kernel, mode='same', boundary='symm')
    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor ** 2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    return grid, (xmin, xmax, ymin, ymax)


if __name__ == '__main__':
    points = np.vstack([
        np.random.multivariate_normal((4, 4), [[2.5, 10], [10, 2.5]], 100_000) + 0.5,
        np.random.multivariate_normal((0, 0), [[4, 0], [0, 4]], 100_000) + 1
    ])

    st = time.time()
    xx, yy, zz, bounding_box = kde2D(points[:, 0], points[:, 1], points, resolution=512)
    
    print(f"It took {time.time() - st:.2f}s for {points.shape[0]:,} items.")

    fig = plt.figure()
    ax = fig.gca()
    cset = ax.contourf(xx, yy, zz)
    # ax.set_ylim((-10, 10))
    # ax.set_xlim((-10, 10))
    plt.show()