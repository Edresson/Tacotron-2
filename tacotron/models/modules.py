import tensorflow as tf
import qrnn



import tensorflow as tf
import numbers
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def zoneout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """Computes zoneout (including dropout without scaling).
    With probability `keep_prob`.
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.
    Args:
      x: A tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      name: A name for this operation (optional).
    Returns:
      A Tensor of the same shape of `x`.
    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    with tf.name_scope(name or "dropout") as name:
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob,
                                        dtype=x.dtype,
                                        name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                 seed=seed,
                                                 dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        ret.set_shape(x.get_shape())
        return 1. - ret


class QRNN_pooling(tf.nn.rnn_cell.RNNCell):

    def __init__(self, out_fmaps, pool_type):
        self.__pool_type = pool_type
        self.__out_fmaps = out_fmaps

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        """
        inputs: 2-D tensor of shape [batch_size, Zfeats + [gates]]
        """
        pool_type = self.__pool_type
        # print('QRNN pooling inputs shape: ', inputs.get_shape())
        # print('QRNN pooling state shape: ', state.get_shape())
        with tf.variable_scope(scope or "QRNN-{}-pooling".format(pool_type)):
            if pool_type == 'f':
                # extract Z activations and F gate activations
                Z, F = tf.split(1, 2, inputs)
                # return the dynamic average pooling
                output = tf.mul(F, state) + tf.mul(tf.sub(1., F), Z)
                return output, output
            elif pool_type == 'fo':
                # extract Z, F gate and O gate
                Z, F, O = tf.split(1, 3, inputs)
                new_state = tf.mul(F, state) + tf.mul(tf.sub(1., F), Z)
                output = tf.mul(O, new_state)
                return output, new_state
            elif pool_type == 'ifo':
                # extract Z, I gate, F gate, and O gate
                Z, I, F, O = tf.split(1, 4, inputs)
                new_state = tf.mul(F, state) + tf.mul(I, Z)
                output = tf.mul(O, new_state)
                return output, new_state
            else:
                raise ValueError('Pool type must be either f, fo or ifo')



class QRNN_layer(object):
    """ Quasi-Recurrent Neural Network Layer
        (cf. https://arxiv.org/abs/1611.01576)
    """
    def __init__(self, out_fmaps, fwidth=2,
                 activation=tf.tanh, pool_type='fo', zoneout=0.1, infer=False,
                 bias_init_val=None,
                 name='QRNN'):
        """
        pool_type: can be f, fo, or ifo
        zoneout: > 0 means apply zoneout with p = 1 - zoneout
        bias_init_val: by default there is no bias.
        """
        self.out_fmaps = out_fmaps
        self.activation = activation
        self.name = name
        self.infer = infer
        self.pool_type = pool_type
        self.fwidth = fwidth
        self.out_fmaps = out_fmaps
        self.zoneout = zoneout
        self.bias_init_val = bias_init_val

    def __call__(self, input_):
        input_shape = input_.get_shape().as_list()
        batch_size = input_shape[0]
        fwidth = self.fwidth
        out_fmaps = self.out_fmaps
        pool_type = self.pool_type
        zoneout = self.zoneout
        with tf.variable_scope(self.name):
            # gates: list containing gate activations (num of gates depending
            # on pool_type)
            Z, gates = self.convolution(input_, fwidth, out_fmaps, pool_type,
                                        zoneout)
            # join all features (Z and gates) into Tensor at dim 2 merged
            T = tf.concat(2, [Z] + gates)
            # create the pooling layer
            pooling = QRNN_pooling(out_fmaps, pool_type)
            self.initial_state = pooling.zero_state(batch_size=batch_size,
                                                    dtype=tf.float32)
            # encapsulate the pooling in the iterative dynamic_rnn
            H, last_C = tf.nn.dynamic_rnn(pooling, T,
                                          initial_state=self.initial_state)
            self.Z = Z
            return H, last_C

    def convolution(self, input_, filter_width, out_fmaps, pool_type, zoneout_):
        """ Applies 1D convolution along time-dimension (T) assuming input
            tensor of dim (batch_size, T, n) and returns
            (batch_size, T, out_fmaps)
            zoneout: regularization (dropout) of F gate
        """
        in_shape = input_.get_shape()
        in_fmaps = in_shape[-1]
        num_gates = len(pool_type)
        gates = []
        # pad on the left to mask the convolution (make it causal)
        pinput = tf.pad(input_, [[0, 0], [filter_width - 1, 0], [0, 0]])
        with tf.variable_scope('convolutions'):
            Wz = tf.get_variable('Wz', [filter_width, in_fmaps, out_fmaps],
                                 initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
            z_a = tf.nn.conv1d(pinput, Wz, stride=1, padding='VALID')
            if self.bias_init_val is not None:
                bz = tf.get_variable('bz', [out_fmaps],
                                     initializer=tf.constant_initializer(0.))
                z_a += bz

            z = self.activation(z_a)
            # compute gates convolutions
            for gate_name in pool_type:
                Wg = tf.get_variable('W{}'.format(gate_name),
                                     [filter_width, in_fmaps, out_fmaps],
                                     initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
                g_a = tf.nn.conv1d(pinput, Wg, stride=1, padding='VALID')
                if self.bias_init_val is not None:
                    bg = tf.get_variable('b{}'.format(gate_name), [out_fmaps],
                                         initializer=tf.constant_initializer(0.))
                    g_a += bg
                g = tf.sigmoid(g_a)
                if not self.infer and zoneout_ > 0 and gate_name == 'f':
                    print('Applying zoneout {} to gate F'.format(zoneout_))
                    # appy zoneout to F
                    g = zoneout((1. - g), 1. - zoneout_)
                    # g = 1. - tf.nn.dropout((1. - g), 1. - zoneout)
                gates.append(g)
        return z, gates


class HighwayNet:
	def __init__(self, units, name=None):
		self.units = units
		self.scope = 'HighwayNet' if name is None else name

		self.H_layer = tf.layers.Dense(units=self.units, activation=tf.nn.relu, name='H')
		self.T_layer = tf.layers.Dense(units=self.units, activation=tf.nn.sigmoid, name='T', bias_initializer=tf.constant_initializer(-1.))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			H = self.H_layer(inputs)
			T = self.T_layer(inputs)
			return H * T + inputs * (1. - T)


class CBHG:
	def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size, n_highwaynet_layers, highway_units, rnn_units, bnorm, is_training, name=None):
		self.K = K
		self.conv_channels = conv_channels
		self.pool_size = pool_size

		self.projections = projections
		self.projection_kernel_size = projection_kernel_size
		self.bnorm = bnorm

		self.is_training = is_training
		self.scope = 'CBHG' if name is None else name

		self.highway_units = highway_units
		self.highwaynet_layers = [HighwayNet(highway_units, name='{}_highwaynet_{}'.format(self.scope, i+1)) for i in range(n_highwaynet_layers)]
		#self._fw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_forward_RNN'.format(self.scope))
		#self._bw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_backward_RNN'.format(self.scope))
                self._fw_cell = QRNN_layer(rnn_units, name='{}_forward_RNN'.format(self.scope), pool_type='fo')
                self._bw_cell = QRNN_layer(rnn_units, name='{}_backward_RNN'.format(self.scope), pool_type='fo')
                
	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('conv_bank'):
				#Convolution bank: concatenate on the last axis to stack channels from all convolutions
				#The convolution bank uses multiple different kernel sizes to have many insights of the input sequence
				#This makes one of the strengths of the CBHG block on sequences.
				conv_outputs = tf.concat(
					[conv1d(inputs, k, self.conv_channels, tf.nn.relu, self.is_training, 0., self.bnorm, 'conv1d_{}'.format(k)) for k in range(1, self.K+1)],
					axis=-1
					)

			#Maxpooling (dimension reduction, Using max instead of average helps finding "Edges" in mels)
			maxpool_output = tf.layers.max_pooling1d(
				conv_outputs,
				pool_size=self.pool_size,
				strides=1,
				padding='same')

			#Two projection layers
			proj1_output = conv1d(maxpool_output, self.projection_kernel_size, self.projections[0], tf.nn.relu, self.is_training, 0., self.bnorm, 'proj1')
			proj2_output = conv1d(proj1_output, self.projection_kernel_size, self.projections[1], lambda _: _, self.is_training, 0., self.bnorm, 'proj2')

			#Residual connection
			highway_input = proj2_output + inputs

			#Additional projection in case of dimension mismatch (for HighwayNet "residual" connection)
			if highway_input.shape[2] != self.highway_units:
				highway_input = tf.layers.dense(highway_input, self.highway_units)

			#4-layer HighwayNet
			for highwaynet in self.highwaynet_layers:
				highway_input = highwaynet(highway_input)
			rnn_input = highway_input

			#Bidirectional RNN
			outputs, states = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				rnn_input,
				sequence_length=input_lengths,
				dtype=tf.float32)
			return tf.concat(outputs, axis=2) #Concat forward and backward outputs


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
	'''Wrapper for tf LSTM to create Zoneout LSTM Cell

	inspired by:
	https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py

	Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.

	Many thanks to @Ondal90 for pointing this out. You sir are a hero!
	'''
	def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
		'''Initializer with possibility to set different zoneout values for cell/hidden states.
		'''
		zm = min(zoneout_factor_output, zoneout_factor_cell)
		zs = max(zoneout_factor_output, zoneout_factor_cell)

		if zm < 0. or zs > 1.:
			raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

		self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
		self._zoneout_cell = zoneout_factor_cell
		self._zoneout_outputs = zoneout_factor_output
		self.is_training = is_training
		self.state_is_tuple = state_is_tuple

	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._cell.output_size

	def __call__(self, inputs, state, scope=None):
		'''Runs vanilla LSTM Cell and applies zoneout.
		'''
		#Apply vanilla LSTM
		output, new_state = self._cell(inputs, state, scope)

		if self.state_is_tuple:
			(prev_c, prev_h) = state
			(new_c, new_h) = new_state
		else:
			num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
			prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
			prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
			new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
			new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

		#Apply zoneout
		if self.is_training:
			#nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
			c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
			h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

		else:
			c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
			h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

		new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

		return output, new_state


class EncoderConvolutions:
	"""Encoder convolutional layers used to find local dependencies in inputs characters.
	"""
	def __init__(self, is_training, hparams, activation=tf.nn.relu, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(EncoderConvolutions, self).__init__()
		self.is_training = is_training

		self.kernel_size = hparams.enc_conv_kernel_size
		self.channels = hparams.enc_conv_channels
		self.activation = activation
		self.scope = 'enc_conv_layers' if scope is None else scope
		self.drop_rate = hparams.tacotron_dropout_rate
		self.enc_conv_num_layers = hparams.enc_conv_num_layers
		self.bnorm = hparams.batch_norm_position

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(self.enc_conv_num_layers):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, self.drop_rate, self.bnorm, 'conv_layer_{}_'.format(i + 1)+self.scope)
		return x


class EncoderRNN:
	"""Encoder bidirectional one layer LSTM
	"""
	def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control zoneout
			size: integer, the number of LSTM units for each direction
			zoneout: the zoneout factor
			scope: EncoderRNN scope.
		"""
		super(EncoderRNN, self).__init__()
		self.is_training = is_training

		self.size = size
		self.zoneout = zoneout
		self.scope = 'encoder_LSTM' if scope is None else scope

		#Create forward LSTM Cell
		self._fw_cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_fw_LSTM')

		#Create backward LSTM Cell
		self._bw_cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_bw_LSTM')

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				inputs,
				sequence_length=input_lengths,
				dtype=tf.float32,
				swap_memory=True)

			return tf.concat(outputs, axis=2) # Concat and return forward + backward outputs


class Prenet:
	"""Two fully connected layers used as an information bottleneck for the attention.
	"""
	def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, scope=None):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Prenet, self).__init__()
		self.drop_rate = drop_rate

		self.layers_sizes = layers_sizes
		self.activation = activation
		self.is_training = is_training

		self.scope = 'prenet' if scope is None else scope

	def __call__(self, inputs):
		x = inputs

		with tf.variable_scope(self.scope):
			for i, size in enumerate(self.layers_sizes):
				dense = tf.layers.dense(x, units=size, activation=self.activation,
					name='dense_{}'.format(i + 1))
				#The paper discussed introducing diversity in generation at inference time
				#by using a dropout of 0.5 only in prenet layers (in both training and inference).
				x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
					name='dropout_{}'.format(i + 1) + self.scope)
		return x


class DecoderRNN:
	"""Decoder two uni directional LSTM Cells
	"""
	def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is in training or inference to control zoneout
			layers: integer, the number of LSTM layers in the decoder
			size: integer, the number of LSTM units in each layer
			zoneout: the zoneout factor
		"""
		super(DecoderRNN, self).__init__()
		self.is_training = is_training

		self.layers = layers
		self.size = size
		self.zoneout = zoneout
		self.scope = 'decoder_rnn' if scope is None else scope

		#Create a set of LSTM layers
		self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='decoder_LSTM_{}'.format(i+1)) for i in range(layers)]

		self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

	def __call__(self, inputs, states):
		with tf.variable_scope(self.scope):
			return self._cell(inputs, states)


class FrameProjection:
	"""Projection layer to r * num_mels dimensions or num_mels dimensions
	"""
	def __init__(self, shape=80, activation=None, scope=None):
		"""
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
		super(FrameProjection, self).__init__()

		self.shape = shape
		self.activation = activation

		self.scope = 'Linear_projection' if scope is None else scope
		self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			#If activation==None, this returns a simple Linear projection
			#else the projection will be passed through an activation function
			# output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
			# 	name='projection_{}'.format(self.scope))
			output = self.dense(inputs)

			return output


class StopProjection:
	"""Projection to a scalar and through a sigmoid activation
	"""
	def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
		"""
		Args:
			is_training: Boolean, to control the use of sigmoid function as it is useless to use it
				during training since it is integrate inside the sigmoid_crossentropy loss
			shape: integer, dimensionality of output space. Defaults to 1 (scalar)
			activation: callable, activation function. only used during inference
			scope: StopProjection scope.
		"""
		super(StopProjection, self).__init__()
		self.is_training = is_training

		self.shape = shape
		self.activation = activation
		self.scope = 'stop_token_projection' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			output = tf.layers.dense(inputs, units=self.shape,
				activation=None, name='projection_{}'.format(self.scope))

			#During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
			if self.is_training:
				return output
			return self.activation(output)


class Postnet:
	"""Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
	"""
	def __init__(self, is_training, hparams, activation=tf.nn.tanh, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(Postnet, self).__init__()
		self.is_training = is_training

		self.kernel_size = hparams.postnet_kernel_size
		self.channels = hparams.postnet_channels
		self.activation = activation
		self.scope = 'postnet_convolutions' if scope is None else scope
		self.postnet_num_layers = hparams.postnet_num_layers
		self.drop_rate = hparams.tacotron_dropout_rate
		self.bnorm = hparams.batch_norm_position

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(self.postnet_num_layers - 1):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, self.drop_rate, self.bnorm, 'conv_layer_{}_'.format(i + 1)+self.scope)
			x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, self.drop_rate, self.bnorm,
				'conv_layer_{}_'.format(5)+self.scope)
		return x


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, bnorm, scope):
	assert bnorm in ('before', 'after')
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=activation if bnorm == 'after' else None,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched) if bnorm == 'before' else batched
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								name='dropout_{}'.format(scope))

def _round_up_tf(x, multiple):
	# Tf version of remainder = x % multiple
	remainder = tf.mod(x, multiple)
	# Tf version of return x if remainder == 0 else x + multiple - remainder
	x_round =  tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
		lambda: x,
		lambda: x + multiple - remainder)

	return x_round

def sequence_mask(lengths, r, expand=True):
	'''Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
	'''
	max_len = tf.reduce_max(lengths)
	max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
	if expand:
		return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
	return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked Mean Squared Error
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(mels)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		return tf.losses.mean_squared_error(labels=targets, predictions=outputs, weights=mask_)

def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked SigmoidCrossEntropy with logits
	'''

	#[batch_size, time_dimension]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[1., 0., 0., 0., 0.],
	#							    [1., 1., 1., 0., 0.],
	#							    [1., 1., 0., 0., 0.]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
		#Use a weighted sigmoid cross entropy to measure the <stop_token> loss. Set hparams.cross_entropy_pos_weight to 1
		#will have the same effect as  vanilla tf.nn.sigmoid_cross_entropy_with_logits.
		losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs, pos_weight=hparams.cross_entropy_pos_weight)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask

	return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

def MaskedLinearLoss(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked MAE loss with priority to low frequencies
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(freq)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	l1 = tf.abs(targets - outputs)
	n_priority_freq = int(2000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		masked_l1 = l1 * mask_
		masked_l1_low = masked_l1[:,:,0:n_priority_freq]

	mean_l1 = tf.reduce_sum(masked_l1) / tf.reduce_sum(mask_)
	mean_l1_low = tf.reduce_sum(masked_l1_low) / tf.reduce_sum(mask_)

	return 0.5 * mean_l1 + 0.5 * mean_l1_low
