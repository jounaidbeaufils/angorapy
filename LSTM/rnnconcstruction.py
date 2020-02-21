import pickle

from LSTM.fixedpointfinder import Adamfixedpointfinder
from LSTM.three_bit_flip_flop import Flipflopper
from LSTM.plot_utils import plot_fixed_points
import autograd.numpy as np
from LSTM.minimization import adam_weights_optimizer
from LSTM.build_utils import build_rnn_ds
from LSTM.plot_utils import plot_fixed_points
import matplotlib.pyplot as plt

############################################################
# Create and train recurrent model on 3-Bit FlipFop task
############################################################
# specify architecture e.g. 'vanilla' and number of hidden units
rnn_type = 'vanilla'
n_hidden = 24
recurrentweights = np.random.randn(24, 24) * 1e-03
# initialize Flipflopper class
flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
# generate trials
stim = flopper.generate_flipflop_trials()
# train the model
# first_history = flopper.train(stim, 4000, save_model=True)
# visualize a single batch after training
# flopper.visualize_flipflop(stim)
# if a trained model has been saved, it may also be loaded
flopper.load_model()
############################################################
# Initialize fpf and find fixed points
############################################################
# get weights and activations of trained model

weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
activations = flopper.get_activations(stim)

# initialize adam fpf
fpf = Adamfixedpointfinder(weights, rnn_type,
                           q_threshold=1e-12,
                           epsilon=0.01,
                           alr_decayr=0.00001,
                           max_iters=7000)
# sample states, i.e. a number of ICs
states = fpf.sample_states(activations, 400)
# vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
# generate corresponding input as zeros for flip flop task
# please that the input does not need to be zero for all tasks
inputs = np.zeros((states.shape[0], 3))
# find fixed points
fps = fpf.find_fixed_points(states, inputs)
# get fps to have points to train for


class Rnnconstructor():
    adam_default_hps = {'alr_hps': {'decay_rate': 0.0001},
                        'agnc_hps': {'norm_clip': 1.0,
                                     'decay_rate': 1e-03},
                        'adam_hps': {'epsilon': 1e-03,
                                     'max_iters': 50000,
                                     'method': 'joint',
                                     'print_every': 200}}

    def __init__(self, fps,
                 alr_decayr=adam_default_hps['alr_hps']['decay_rate'],
                 agnc_normclip=adam_default_hps['agnc_hps']['norm_clip'],
                 agnc_decayr=adam_default_hps['agnc_hps']['decay_rate'],
                 epsilon=adam_default_hps['adam_hps']['epsilon'],
                 max_iters=adam_default_hps['adam_hps']['max_iters'],
                 method=adam_default_hps['adam_hps']['method'],
                 print_every=adam_default_hps['adam_hps']['print_every']):

        self.alr_decayr = alr_decayr
        self.agnc_normclip = agnc_normclip
        self.agnc_decayr = agnc_decayr
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.method = method
        self.print_every = print_every

        self.verbose = True

        self.fun_val = np.empty(len(fps))
        for i in range(len(fps)):
            self.fun_val[i] = fps[i]['fun']
        self.mean_velocity = np.mean(self.fun_val)

    def build_model(self, fps):

        def create_target(fps):
            target = np.empty((len(fps), 24))
            for i in range(len(fps)):
                target[i, :] = fps[i]['x']
            return target

        target = create_target(fps)
        def fun(x):
            return np.max(0.5 * np.sum(((- target + np.matmul(np.tanh(target), x)) ** 2), axis=1))

        return fun

    def train_recurrentweights(self, fps, weights):

        fun = self.build_model(fps)

        # weights = np.random.randn(24, 24) * 1e-03
        # weights, _ = np.linalg.qr(weights)
        #p = 24
        #A = np.random.rand(p, p)
        #P = (A + np.transpose(A)) / 2 + p * np.eye(p)

        # get the subset of its eigenvectors
        #vals, vecs = np.linalg.eig(P)
        #weights = vecs[:, 0:p] * 1e-03

        weights = adam_weights_optimizer(fun, weights, self.mean_velocity,
                                         epsilon=self.epsilon,
                                         alr_decayr=self.alr_decayr,
                                         max_iter=self.max_iters,
                                         print_every=self.print_every,
                                         init_agnc=self.agnc_normclip,
                                         agnc_decayr=self.agnc_decayr,
                                         verbose=self.verbose)

        return weights

plot_fixed_points(activations, fps, 2000, 2)
reco = Rnnconstructor(fps)

recurrentweights = reco.train_recurrentweights(fps, flopper.weights[1])
recurrentweights = np.multiply(recurrentweights, recurrentweights)
# weights[1] = recurrentweights
h = np.zeros(24)
inputs = np.vstack(stim['inputs'])
def fun(inputs, h):
    return np.matmul(np.tanh(h), recurrentweights) + np.matmul(inputs, weights[0]) + weights[2]


recorded_h = []
for i in range(len(inputs)):
    h = fun(inputs[i, :], h)
    recorded_h.append(h)

new_jacobians = np.empty((len(fps), n_hidden, n_hidden))
fph = fps
for i in range(len(fps)):
    fun, jac_fun = build_rnn_ds(weights, n_hidden, inputs[i, :], 'sequential')
    new_jacobians[i, :, :] = jac_fun(fps[i]['x'])
    # print(np.allclose(new_jacobians[i, :, :], fps[i]['jac'], 1e-02))
    fph[i]['jac'] = new_jacobians[i, :, :]
    fph[i]['fun'] = fun(fps[i]['x'])


plot_fixed_points(activations, fph, 2000, 4)


retrained_history = flopper.train_pretrained(stim, 1000, recurrentweights, True)
# retrained_fixed = flopper.train_pretrained(stim, 1000, recurrentweights, False)

history = pickle.load(open('firsttrainhistory', "rb"))
plt.plot(range(len(history['loss'])), history['loss'], 'r--')
plt.plot(range(len(retrained_history.epoch)), retrained_history.history['loss'], 'b-')
# plt.plot(range(len(retrained_fixed.epoch)), retrained_fixed.history['loss'], 'g-')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['naive training', 'after pretraining',
            'after pretraining with fixed recurrent weights'], loc='upper right')
plt.show()