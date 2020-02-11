from typing import Optional, Any

import numpy as np
import multiprocessing as mp
import numdifftools as nd
from scipy.optimize import minimize
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
from LSTM.minimization import backproprnn, backpropgru
from mayavi import mlab
from LSTM.build_utils import build_rnn_ds, build_joint_rnn_ds, \
    build_gru_ds, build_lstm_ds, build_joint_gru_ds
from LSTM.minimization import adam_optimizer
from scipy.spatial.distance import pdist, squareform


class FixedPointFinder(object):

    _default_hps = {'q_threshold': 1e-12,
                    'tol_unique': 1e-03,
                    'use_input': False,
                    'verbose': True,
                    'random_seed': 0}


    def __init__(self, weights, rnn_type,
                 q_threshold=_default_hps['q_threshold'],
                 tol_unique=_default_hps['tol_unique'],
                 use_input=_default_hps['use_input'],
                 verbose=_default_hps['verbose'],
                 random_seed=_default_hps['random_seed']):
        """The class FixedPointFinder creates a fixedpoint dictionary. A fixedpoint dictionary contain 'fun',
        the function evaluation at the fixedpoint 'x' and the corresponding 'jacobian'.

        Args:
            hps: Dictionary of hyperparameters.

                unique_tol: Tolerance for when a fixedpoint will be considered unique, i.e. when two points are further
                away from each than the tolerance, they will be considered unique and discarded otherwise. Default: 1e-03.

                threshold: Minimization criteria. A fixedpoint must evaluate below the threshold in order to be considered
                a slow/fixed point. This value depends on the task of the RNN. Default for 3-Bit Flip-FLop: 1e-12.

                rnn_type: Specifying the architecture of the network. The network architecture defines the dynamical system.
                Must be one of ['vanilla', 'gru', 'lstm']. No default.

                n_hidden: Specifiying the number of hidden units in the recurrent layer. No default.

                algorithm: Algorithm that shall be employed for the minimization. Must be one of: scipy, adam. It is recommended
                to use any of the two for vanilla architectures but adam for gru and lstm architectures. No default.

                n_points: Number of points to use to plot the trajectories the network took. Recommended 200-5000, otherwise
                the plot will look too sparse or too crowded. Default: 3000.

                use_input: boolean parameter indicating if input to the recurrent layer shall be used during minimization or
                not. Default: False

            scipy_hps: Dictionary of hyperparameters specifically for minimize from scipy.

                method: Method to employ for minimization using the scipy package. Default: "Newton-CG".

                display: boolean array indication, if information about the minimization shall be printed to the console

            adam_hps: Dictionary of hyperparameters specifically for adam optimizer.

                max_iter: maximum number of iterations to run backpropagation for. Default: 5000.



            weights: list of weights as returned by tensorflow.keras for recurrent layer. The list must contain three objects:
            input weights, recurrent weights and biases."""

        self.weights = weights
        self.rnn_type = rnn_type

        self.q_threshold = q_threshold
        self.unique_tol = tol_unique

        self.verbose = verbose
        self.use_input = use_input

        self.rng = np.random.RandomState(random_seed)

        if self.rnn_type == 'vanilla':
            self.n_hidden = int(weights[1].shape[1])
        elif self.rnn_type == 'gru':
            self.n_hidden = int(weights[1].shape[1] / 3)
        elif self.rnn_type == 'lstm':
            self.n_hidden = int(weights[1].shape[1] / 4)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.rnn_type)

        if self.verbose:
            self._print_hps()

    def sample_states(self, activations, n_inits):
        """Draws [n_inits] random samples from recurrent layer activations."""

        if len(activations.shape) == 3:
            activations = np.vstack(activations)

        init_idx = self.rng.randint(activations.shape[0], size=n_inits)

        sampled_activations = activations[init_idx, :]

        return sampled_activations

    def _handle_bad_approximations(self, fps):
        """This functions identifies approximations where the minmization
        was
         a) the fixed point moved further away from IC than minimization distance threshold
         b) minimization did not return q(x) < q_threshold"""

        bad_fixed_points = []
        good_fixed_points = []
        for fp in fps:
            if fp['fun'] > self.q_threshold:
                bad_fixed_points.append(fp)
            else:
                good_fixed_points.append(fp)

        return good_fixed_points, bad_fixed_points

    def compute_velocities(self, activations, input):
        """Function to evaluate velocities at all recorded activations of the recurrent layer.

        Args:
             """

        if self.rnn_type == 'vanilla':
            func = build_rnn_ds(self.weights, input)
        elif self.rnn_type == 'gru':
            weights, n_hidden = self.weights, self.hps['n_hidden']
            func, _ = build_gru_ds(weights, n_hidden, input)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.hps['rnn_type'])

        activations = np.vstack(activations)
        # get velocity at point
        velocities = func(activations)
        return velocities


    def _find_unique_fixed_points(self, fps):
        """Identify fixed points that are unique within a distance threshold
        of """

        def extract_fixed_point_locations(fps):
            """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
            puts all coordinates in single array."""
            fixed_point_location = []
            for fp in fps:
                fixed_point_location.append(fp['x'])
            fixed_point_locations = np.vstack(fixed_point_location)
            return fixed_point_locations

        fps_locations = extract_fixed_point_locations(fps)
        d = int(np.round(np.max([0 - np.log10(self.unique_tol)])))
        ux, idx = np.unique(fps_locations.round(decimals=d), axis=0, return_index=True)

        unique_fps = []
        for id in idx:
            unique_fps.append(fps[id])
        # TODO: use pdist and also select based on lowest q(x)

        return unique_fps

    def _print_hps(self):
        COLORS = dict(
            HEADER='\033[95m',
            OKBLUE='\033[94m',
            OKGREEN='\033[92m',
            WARNING='\033[93m',
            FAIL='\033[91m',
            ENDC='\033[0m',
            BOLD='\033[1m',
            UNDERLINE='\033[4m'
        )
        bc, ec, wn = COLORS["HEADER"], COLORS["ENDC"], COLORS["WARNING"]
        print(f"-----------------------------------------\n"
              f"{wn}Architecture to analyse {ec}: {bc}{self.rnn_type}{ec}\n"
              f"The layer has {bc}{self.n_hidden}{ec} recurrent units. \n"
              # f"Using {bc}{self.algorithm}{ec} for minimization.\n"
              f"-----------------------------------------\n"
              f"{wn}HyperParameters{ec}: threshold - {self.q_threshold}\n unique_tolerance - {self.unique_tol}\n"
              f"-----------------------------------------\n")


class Adamfixedpointfinder(FixedPointFinder):
    adam_default_hps = {'alr_hps': {'decay_rate': 0.001},
                        'agnc_hps': {'norm_clip': 1.0,
                                     'decay_rate': 1e-03},
                        'adam_hps': {'epsilon': 1e-02,
                                     'max_iters': 5000,
                                     'method': 'joint',
                                     'print_every': 200}}

    def __init__(self, weights, rnn_type,
                 alr_hps=adam_default_hps['alr_hps'],
                 agnc_hps=adam_default_hps['agnc_hps'],
                 adam_hps=adam_default_hps['adam_hps'],
                 ):
        FixedPointFinder.__init__(self, weights, rnn_type)
        self.alr_hps = alr_hps
        self.agnc_hps = agnc_hps
        self.adam_hps = adam_hps



    def find_fixed_points(self, x0, inputs):
        if self.rnn_type == 'vanilla':
            fun = build_joint_rnn_ds(self.weights, inputs)
            # per_point_fun = build_rnn_ds(self.weights, inputs)
        elif self.rnn_type == 'gru':
            fun, dynamical_system = build_joint_gru_ds(self.weights, self.n_hidden, inputs)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of'
                             '[vanilla, gru, lstm] but was %s', self.rnn_type)

        x0 = adam_optimizer(fun, x0,
                            epsilon=self.adam_hps['epsilon'],
                            max_iter=self.adam_hps['max_iters'],
                            print_every=200,
                            agnc=self.agnc_hps['norm_clip'])
        n_hidden, weights = self.n_hidden, self.weights[1]
        jac_fun = lambda x: - np.eye(n_hidden, n_hidden) + weights * (1 - np.tanh(x) ** 2)
        # nd.Jacobian(dynamical_system)
        fixedpoints = []
        for i in range(len(x0)):
            jacobian = jac_fun(x0[i, :])
            q = fun(x0[i, :])
            fixedpoint = {'fun': q,
                          'x': x0[i, :],
                          'jac': jacobian}
            fixedpoints.append(fixedpoint)

        good_fps, bad_fps = self._handle_bad_approximations(fixedpoints)
        unique_fps = self._find_unique_fixed_points(good_fps)

        return unique_fps


class Scipyfixedpointfinder(FixedPointFinder):
    scipy_default_hps = {'method': 'Newton-CG',
                         'gtol': 1e-20,
                         'display': True}

    def __init__(self, weights, rnn_type,
                 scipy_hps=scipy_default_hps['scipy_hps']):
        FixedPointFinder.__init__(self, weights, rnn_type)
        self.scipy_hps = scipy_hps


    def find_fixed_points(self):
        pass