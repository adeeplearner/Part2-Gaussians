import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def Gaussian1D(x, mu, sigma):
    """Implements 1D Gaussian using the following equation
     P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\big[\frac{(x-\mu)^2}{\sigma^2}\big]},
    Args:
        x (np.array): x points to evaluate Gaussian on 
        mu (float): mean of the distribution
        sigma (float): standard deviation of the distribution

    Returns:
        np.array: distribution evaluated on x points
    """
    ATerm = 1/(sigma * np.sqrt(2 * np.pi))
    BTerm = np.exp(-0.5 * ((x-mu)/sigma) ** 2)
    return ATerm * BTerm


def discreteIntegral1D(p, dx):
    """Implements 1D discrete integral using the following equation
    integral = \sum_x P(x) dx

    It can also be thought of as area under the curve

    Args:
        p (np.array): points in distribution
        dx (float): delta x, spacing of discrete x points

    Returns:
        float: integral over the discrete space
    """
    return np.sum(p) * dx
    
def kullbackLeiblerDivergence(p, q, dx):
    """Implements Kullback-Leibler Divergence between two distribution P(x) and Q(x)
    Uses the following equation:
    D_{KL}(P \parallel Q) = \sum_x P(x) \log (\frac{P(x)}{Q(x)})

    Details about KL divergence:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Args:
        p (np.array): P(x) discrete distribution 
        q (np.array): Q(x) discrete distribution
        dx (float): delta x, spacing of discrete x points

    Returns:
        float: Kullback-Leibler Divergence
    """
    return discreteIntegral1D( (p * np.log(p/q)), dx )

def bhattacharyyaDistance(p, q, dx):
    """Implements Bhattacharrya Distance between two distribution P(x) and Q(x)
    Uses the following equation:
    D_{BC}(P, Q) = -\log \sum_x \sqrt {P(x) Q(x)}

    Details about Bhattacharyya distance:
    https://en.wikipedia.org/wiki/Bhattacharyya_distance

    Args:
        p (np.array): P(x) discrete distribution 
        q (np.array): Q(x) discrete distribution
        dx (float): delta x, spacing of discrete x points

    Returns:
        float: Bhattacharyya distance
    """
    return -np.log( discreteIntegral1D( np.sqrt(p*q), dx ) )


def saveVaryingMean_KL_BC_Gif(folder='figures'):
    save_folder = os.path.join(folder, 'distance_plots')
    os.makedirs(save_folder, exist_ok=True)
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    print('*' * 60)
    print('Generating gif for varying MU')
    print('*' * 60)
    x_grid = np.linspace(-6, 6, 100)
    deltax = x_grid[1]-x_grid[0]

    p = Gaussian1D(x_grid, 0, 1)

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)
    canvas = FigureCanvas(fig)
    sigma = 1.0
    MU = np.arange(-3, 3, 0.1)
    
    mux_list = []
    kldiv_list = []
    bcdis_list = []

    fig_list = []
    for idx, mu in enumerate(MU):
        print('%d/%d' % (idx, MU.shape[0]))
        q = Gaussian1D(x_grid, mu, sigma)
        kldiv = kullbackLeiblerDivergence(p, q, deltax)
        bcdis = bhattacharyyaDistance(p, q, deltax)

        mux_list.append(mu)
        kldiv_list.append(kldiv)
        bcdis_list.append(bcdis)

        plt.subplot(1, 2, 1)
        ax = plt.gca()
        plt.plot(x_grid, p, 'g')
        plt.plot(x_grid, q, 'r')
        plt.legend(['P(x)', 'Q(x)'])

        ax.fill_between(x_grid, p, 0, alpha=0.3, color='g')
        ax.fill_between(x_grid, q, 0, alpha=0.3, color='r')
        
        plt.ylim([-0.01, 0.5])
        # plt.title('Gaussian1D with $\mu=%0.1f$, $\sigma=%0.01f$' % (mu, sigma))
        canvas.draw()
        plt.grid()

        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(mux_list, kldiv_list)
        plt.plot(mux_list, bcdis_list)
        plt.xlim(np.min(MU), np.max(MU)+0.001)
        plt.ylim(0, 7)
        plt.legend(['KL', 'BC'])
        canvas.draw()
        

        # plt.savefig(os.path.join(save_folder, 'gaussian_1d_mu_%0.5d.png' % idx), dpi=100, bbox_inches='tight')

        # figure to image help from: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        fig_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_list.append(Image.fromarray(fig_image.reshape(canvas.get_width_height()[::-1] + (3,))))
        plt.clf()
    
    # gif help from: https://note.nkmk.me/en/python-pillow-gif/
    fig_list[0].save(os.path.join(save_folder, 'gaussian_1d_mu_klbc.gif'),
               save_all=True, append_images=fig_list[1:], optimize=False, duration=80, loop=0)
    print('*' * 60)
    print(' ')

def saveVaryingStd_KL_BC_Gif(folder='figures'):
    save_folder = os.path.join(folder, 'distance_plots')
    os.makedirs(save_folder, exist_ok=True)
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    print('*' * 60)
    print('Generating gif for varying SIGMA')
    print('*' * 60)
    x_grid = np.linspace(-6, 6, 100)
    deltax = x_grid[1]-x_grid[0]

    p = Gaussian1D(x_grid, 0, 1)

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)
    canvas = FigureCanvas(fig)
    mu = 0
    SIGMA = np.arange(0.01, 3, 0.1)

    sigx_list = []
    kldiv_list = []
    bcdis_list = []

    fig_list = []
    for idx, sigma in enumerate(SIGMA):
        print('%d/%d' % (idx, SIGMA.shape[0]))
        q = Gaussian1D(x_grid, mu, sigma)
        kldiv = kullbackLeiblerDivergence(p, q, deltax)
        bcdis = bhattacharyyaDistance(p, q, deltax)

        sigx_list.append(sigma)
        kldiv_list.append(kldiv)
        bcdis_list.append(bcdis)

        plt.subplot(1, 2, 1)
        plt.grid(True)
        ax = plt.gca()
        plt.plot(x_grid, p, 'g')
        plt.plot(x_grid, q, 'r')
        
        ax.fill_between(x_grid, p, 0, alpha=0.3, color='g')
        ax.fill_between(x_grid, q, 0, alpha=0.3, color='r')
        
        plt.ylim([-0.01, 0.5])
        # plt.title('Gaussian1D with $\mu=%0.1f$, $\sigma=%0.2f$' % (mu, sigma))
        canvas.draw()

        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(sigx_list, kldiv_list)
        plt.plot(sigx_list, bcdis_list)
        plt.xlim(np.min(SIGMA), np.max(SIGMA)+0.001)
        plt.ylim(0, 7)
        plt.legend(['KL', 'BC'])
        canvas.draw()

        # plt.savefig(os.path.join(save_folder, 'gaussian_1d_sig_%0.5d.png' % idx), dpi=100, bbox_inches='tight')


        # figure to image help from: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        fig_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_list.append(Image.fromarray(fig_image.reshape(canvas.get_width_height()[::-1] + (3,))))
        plt.clf()
    
    # gif help from: https://note.nkmk.me/en/python-pillow-gif/
    fig_list[0].save(os.path.join(save_folder, 'gaussian_1d_sig_klbc.gif'),
               save_all=True, append_images=fig_list[1:], optimize=False, duration=80, loop=0)
    print('*' * 60)
    print(' ')

def saveIntroIntegralFigure(folder='figures'):
    save_folder = os.path.join(folder, 'integral_intro')
    os.makedirs(save_folder, exist_ok=True)
    x_grid = np.linspace(-6, 6, 100)
    y = Gaussian1D(x_grid, 0, 1)

    plt.figure()
    ax = plt.gca()
    plt.plot(x_grid, y)
    ax.fill_between(x_grid,y,0, alpha=0.3, color='b')
    plt.ylim(-0.01, 0.5)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'gaussian_1d_integral_intro.png'), dpi=100, bbox_inches='tight')

def saveIntroGaussianFigure(folder='figures'):
    save_folder = os.path.join(folder, 'gaussian_intro')
    os.makedirs(save_folder, exist_ok=True)
    x_grid = np.linspace(-6, 6, 100)
    y = Gaussian1D(x_grid, 0, 1)

    plt.figure()
    ax = plt.gca()
    plt.plot(x_grid, y)
#    ax.fill_between(x_grid,y,0, alpha=0.3, color='b')
    plt.ylim(-0.01, 0.5)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'gaussian_1d_intro.png'), dpi=100, bbox_inches='tight')

def saveIntroIntegralIntervalFigure(folder='figures'):
    save_folder = os.path.join(folder, 'integral_intro')
    os.makedirs(save_folder, exist_ok=True)
    x_grid = np.linspace(-6, 6, 100)
    y = Gaussian1D(x_grid, 0, 1)

    plt.figure()
    ax = plt.gca()
    plt.plot(x_grid, y)
    x_grid_sel = x_grid[np.where(np.logical_and(x_grid>=-1, x_grid<=1))[0]]
    y_sel = Gaussian1D(x_grid_sel, 0, 1)
    ax.fill_between(x_grid_sel,y_sel, 0, alpha=0.3, color='b')

    prob = discreteIntegral1D(y_sel, x_grid_sel[1]-x_grid_sel[0])

    plt.title('Probability of x=0 with +-1 tol is %0.2f' % prob)
    
    # import matplotlib.ticker as ticker
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.ylim(-0.01, 0.5)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'gaussian_1d_integral_interval.png'), dpi=100, bbox_inches='tight')



def saveVaryingStdIntegralIntervalGif(folder='figures'):
    save_folder = os.path.join(folder, 'integral_intro', 'varystd')
    os.makedirs(save_folder, exist_ok=True)
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    print('*' * 60)
    print('Generating gif for varying SIGMA')
    print('*' * 60)
    x_grid = np.linspace(-6, 6, 100)

    fig = plt.figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)
    mu = 0
    SIGMA = np.arange(0.5, 3, 0.1)

    fig_list = []
    for idx, sigma in enumerate(SIGMA):
        print('%d/%d' % (idx, SIGMA.shape[0]))
        q = Gaussian1D(x_grid, mu, sigma)

        plt.grid(True)
        ax = plt.gca()
        plt.plot(x_grid, q, 'b')
        x_grid_sel = x_grid[np.where(np.logical_and(x_grid>=-1, x_grid<=1))[0]]
        y_sel = Gaussian1D(x_grid_sel, mu, sigma)
        ax.fill_between(x_grid_sel,y_sel, 0, alpha=0.3, color='b')

        prob = discreteIntegral1D(y_sel, x_grid_sel[1]-x_grid_sel[0])

        plt.title('Probability of x=0 with +-1 tol is %0.2f' % prob)
        plt.ylim([-0.01, 0.9])
        # plt.title('Gaussian1D with $\mu=%0.1f$, $\sigma=%0.2f$' % (mu, sigma))
        canvas.draw()

        plt.savefig(os.path.join(save_folder, 'gaussian_integral_int_sig_%0.5d.png' % idx), dpi=100, bbox_inches='tight')


        # figure to image help from: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        fig_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_list.append(Image.fromarray(fig_image.reshape(canvas.get_width_height()[::-1] + (3,))))
        plt.clf()
    
    # gif help from: https://note.nkmk.me/en/python-pillow-gif/
    fig_list[0].save(os.path.join(save_folder, 'gaussian_integral_int_sig.gif'),
               save_all=True, append_images=fig_list[1:], optimize=False, duration=100, loop=0)
    print('*' * 60)
    print(' ')

def integralExample():
    x_grid = np.linspace(-6, 6, 100)
    y = Gaussian1D(x_grid, mu=0, sigma=1)
    deltax = x_grid[1]-x_grid[0]
    integral = discreteIntegral1D(y, deltax)
    print('Integral is: %f' % integral)
    # Output:
    # Integral is: 0.9999
    
def divergenceExample():
    x_grid = np.linspace(-6, 6, 100)
    
    P_gt = Gaussian1D(x_grid, mu=0, sigma=1)
    P_m = Gaussian1D(x_grid, mu=3, sigma=1)

    deltax = x_grid[1]-x_grid[0]
    
    kldiv = kullbackLeiblerDivergence(P_gt, P_gt, deltax)
    bc = bhattacharyyaDistance(P_gt, P_gt, deltax)
    
    print('Comparing same distributions')
    print('KL div is: %f' % kldiv)
    print('BC is: %f' % bc)
    
    print('')
    kldiv = kullbackLeiblerDivergence(P_gt, P_m, deltax)
    bc = bhattacharyyaDistance(P_gt, P_m, deltax)
    print('Comparing different distributions')
    print('KL div is: %f' % kldiv)
    print('BC is: %f' % bc)
    
    # Output: 
    # Comparing same distributions
    # KL div is: 0.000000
    # BC is: 0.000000

    # Comparing different distributions
    # KL div is: 4.500000
    # BC is: 1.125003

    
    

if __name__ == '__main__':
    integralExample()
    divergenceExample()
    saveIntroGaussianFigure()
#    saveIntroIntegralFigure()
#    saveIntroIntegralIntervalFigure()
#    saveVaryingStdIntegralIntervalGif()
#    saveVaryingMean_KL_BC_Gif()
#    saveVaryingStd_KL_BC_Gif()
