# Helper file for the BI_WS01.ipynb notebook. 
#
# authors: rpkrijnen, mpeirlinck
# (c) Peirlinck Lab, Delft University of Technology, 2025  
# Correspondence: mplab-me@tudelft.nl  
#
# Changing parts of this file requires you to restart the kernel in the notebook!

import numpy as np

import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter

import scipy.stats as stats
from scipy.integrate import quad, dblquad
from scipy.special import gamma, gammaln, kl_div


COMPUTE_KLD = False

TIME_TEMPLATE = 'nr measurements = %d' # Text template for printing in the animation
TEXT_TEMPLATE = 'nr measurements = %d\nKLD = %f' # Text template for printing in the animation

def normal_pdf(x, loc:float=0.0, scale:float=1.0):
    """Implementation of the normal distribution's PDF. 
        x: draw from the domain of the distribution
        loc: mean of the distribution
        scale: the standard deviation of the distribution"""
    normalizer = (1/(2*np.pi*scale**2))**(1/2)
    return (normalizer*np.exp(-(1/(2*scale**2))*(x-loc)**2)).squeeze()

def inv_gamma_pdf(x, alpha=1, beta=1):
    """Implementation of the inverse-gamma PDF. 
        x: draw from the domain of the distribution
        alpha: shape parameter
        beta: rate parameter"""
    assert alpha > 0
    assert beta > 0
    return (beta**alpha / gamma(alpha)) * (x)**(-(alpha+1)) * np.exp(-beta/x)

## The animation update class
# Objects created from this class are repeatedly called upon by FuncAnimation objects. This allows us to animate the behavior of the posterior distribution when we continuously add more available measurements.
class UpdateDist:
    def __init__(self, 
                 ax, 
                 nr_measurements, 
                 true_pdf, 
                 true_pdf_sampler, 
                 marginalized_prior,
                 posterior_predictive,
                 show_freq:bool=False, 
                 nr_outliers:int=0
                 ):
        self.true_pdf = true_pdf
        self.true_pdf_sampler = true_pdf_sampler
        self.marginalized_prior = marginalized_prior
        self.posterior_predictive = posterior_predictive
        self.show_freq = show_freq

        self.divergence = np.inf

        self.nr_measurements = nr_measurements
        self.measurements = true_pdf_sampler(self.nr_measurements)

        #If we include outliers, add some random samples from a crazy distribution
        if nr_outliers>0:
            idcs = np.random.choice(np.arange(self.nr_measurements), nr_outliers)
            self.measurements[idcs] = np.random.normal(loc=1e1, scale=1e0, size=(nr_outliers,)).reshape(nr_outliers, 1)

        self.draws = np.linspace(-3, 3, 10_000).reshape(1, 10_000)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(0, 2)
        self.ax.grid(True)

        self.prior_line, = ax.plot([], [], 'g-', label="prior")
        self.posterior_line, = ax.plot([], [], 'r-', label="posterior")
        self.true_line, = ax.plot([], [], 'k-', label="true")
        self.freq_line, = ax.plot([], [], 'b-')
        if self.show_freq:
            self.freq_line, = ax.plot([], [], 'b-', label="frequentist")

        L=plt.legend(loc=1)
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def start(self):
        # Used for the *init_func* parameter of FuncAnimation; this is called when
        # initializing the animation, and also after resizing the figure.
        if COMPUTE_KLD:
            self.time_text.set_text(TEXT_TEMPLATE % (0,self.divergence))
        else:
            self.time_text.set_text(TIME_TEMPLATE % 0)
        return self.prior_line,self.posterior_line,self.true_line,self.time_text,self.freq_line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.prior_line.set_data([], [])
            self.posterior_line.set_data([], [])
            self.true_line.set_data([], [])
            self.freq_line.set_data([], [])

            return self.prior_line,self.posterior_line,self.true_line,self.time_text,

        # Generate draws from the true, prior and posterior distributions
        true_posterior = self.true_pdf(self.draws)
        prior_pdf_draws = self.marginalized_prior(self.draws)  #np.array([quad(prior_pdf, 0, np.inf, args=(draw)) for draw in self.draws])
        posterior_predictive_distribution_draws = self.posterior_predictive(self.draws, self.measurements[:i])

        # Compute KLD using entropy function by choosing qk!=None
        self.divergence = stats.entropy(pk=true_posterior.flatten(), qk=posterior_predictive_distribution_draws.flatten())

        self.prior_line.set_data(self.draws, prior_pdf_draws)
        self.posterior_line.set_data(self.draws, posterior_predictive_distribution_draws)
        self.true_line.set_data(self.draws, true_posterior)

        if i == 1 and self.show_freq:
            self.freq_line.set_data([np.mean(self.measurements[:i])]*100, np.linspace(0, 2, 100))
        elif self.show_freq:
            self.freq_line.set_data(self.draws, normal_pdf(self.draws, np.mean(self.measurements[:i]), np.std(self.measurements[:i])))
        if COMPUTE_KLD:
            self.time_text.set_text(TEXT_TEMPLATE % (i, self.divergence))
        else:
            self.time_text.set_text(TIME_TEMPLATE % i)
        return self.prior_line, self.posterior_line, self.true_line, self.time_text, self.freq_line,


def plot_distributions(i, true_pdf, true_pdf_sampler, marginalized_prior, posterior_predictive, nr_meassurements=100):
    stationary_fig, axs = plt.subplots()
    ud0 = UpdateDist(axs, nr_meassurements, true_pdf=true_pdf, true_pdf_sampler=true_pdf_sampler, marginalized_prior=marginalized_prior, posterior_predictive=posterior_predictive)
    ud0.start()
    lines = ud0.__call__(i)
    stationary_fig.lines=lines
    stationary_fig.show()