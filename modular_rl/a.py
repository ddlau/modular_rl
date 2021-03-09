import numpy as np
import tensorflow as tf

from scipy.signal import lfilter
from tensorflow_probability.python.optimizer import lbfgs_minimize as lbfgs

tuple( map( lambda x: tf.config.experimental.set_memory_growth( x, True ), tf.config.list_physical_devices( 'GPU' ) ) )

lv = print


def discount( x, decay, check=None ):
	y = lfilter( [ 1 ], [ 1, -decay ], x[ ::-1 ], axis=0 )[ ::-1 ]

	if check:
		v = 0
		for t in reversed( range( len( x ) ) ):
			assert y[ t ] == (v := v * decay + x[ t ])

	return y


def flatten( seq ):
	return tf.concat( list( tf.reshape( x, [ -1 ] ) for x in seq ), axis=0 )


def replace( seq, vec ):
	b = 0
	for x in seq:
		s = tf.shape( x )
		l = tf.reduce_prod( s )
		x.assign( tf.reshape( tf.slice( vec, [ b ], [ l ] ), s ) )
		b = b + l


def reshape( seq, vec, replace=True ):
	r = list()
	b = 0
	for x in seq:
		s = tf.shape( x )
		l = tf.reduce_prod( s )
		z = tf.reshape( tf.slice( vec, [ b ], [ l ] ), s )

		if replace:
			x.assign( z )
		else:
			r.append( z )

		b = b + l

	return r


from scipy.optimize import fmin_l_bfgs_b


def build_Axp_function( objective, parameters, damping=0.001 ):
	@tf.function
	def Axp( x ):
		J = tf.gradients( objective, parameters )
		H = flatten( tf.gradients( tf.reduce_sum( flatten( J ) * x ), parameters ) )
		return H + x * damping

	return Axp


# def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
#
# 	p = tf.stop_gradient(b)
# 	r = tf.stop_gradient(b)
# 	x = tf.zeros( tf.shape(b))
#
# 	rtr = tf.reduce_sum(tf.square(r))
#
# 	for i in range(cg_iters):
# 		z = f_Ax(p)
# 		v = rtr / tf.reduce_sum( p*z )
#
# 		x = x+v*p
# 		r = r-v*z
#
# 		new_rtr = tf.reduce_sum(tf.square(r))
# 		mu = new_rtr/rtr
# 		p = r + mu*p
#
# 		rtr = new_rtr
#
# 		if rtr < residual_tol:
# 			print( 'cg, rtr' , rtr)
# 			break
#
# 	#tf.linalg.experimental.conjugate_gradient
#
# 	return x
#
#
# 	p = b.copy()
# 	r = b.copy()
# 	x = np.zeros_like(b)
# 	rdotr = r.dot(r)
#
#
# 	for i in range(cg_iters):
#
#
# 		z = f_Ax(p)
# 		v = rdotr / p.dot(z)
# 		x += v*p
# 		r -= v*z
# 		newrdotr = r.dot(r)
# 		mu = newrdotr/rdotr
# 		p = r + mu*p
#
# 		rdotr = newrdotr
# 		if rdotr < residual_tol:
# 			break
#
# 	return x


class GAE:

	def __init__( self, m, γ, λ, c, τ ):
		self.m = m
		self.γ = γ
		self.λ = λ

		self.c = c
		self.τ = τ

	def A( self, S, R, M ):
		assert len( S ) == len( R ) + 1
		assert len( S ) == len( M ) + 1

		V = self.m( S )
		δ = R + self.γ * M * V[ +1: ] - V[ :-1 ]
		A = discount( δ, self.γ * self.λ )

		return A

	@staticmethod
	def normalize( seq_of_A ):
		A = np.concatenate( seq_of_A, axis=0 )

		return (A - np.mean( A, axis=0 )) / np.std( A, axis=0 )

	def fit1st( self, X, Y ):
		Y = self.τ * Y + (1 - self.τ) * self.m( X )

		@tf.function
		def LnG( θ ):
			replace( self.m.trainable_variables, θ )

			mse = tf.reduce_mean( tf.square( Y - self.m( X ) ) )
			reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( x ) ) for x in self.m.trainable_variables ) )

			los = mse + self.c * reg

			grd = flatten( tf.gradients( los, self.m.trainable_variables ) )

			return los, grd

		los1st = LnG( flatten( self.m.trainable_variables ) )[ 0 ]

		res = lbfgs( LnG, flatten( self.m.trainable_variables ), max_iterations=25 )

		return los1st, res.objective_value


# @tf.function
# def fit2nd( self, X, Y ):
# 	X = tf.cast( X, tf.float32)
# 	Y = tf.cast( Y, tf.float32)
# 	Y = self.τ * Y + (1 - self.τ) * self.m( X )
#
# 	print( X.dtype, Y.dtype)
#
# 	with tf.GradientTape() as t:
# 		mse = tf.reduce_mean( tf.square( Y - self.m( X ) ) )
# 		reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( x ) ) for x in self.m.trainable_variables ) )
# 		los = mse + self.c * reg
#
# 	los1st = los
#
# 	JJ = t.gradient(los, self.m.trainable_variables )
#
#
# 	def Axp(x):
# 		with tf.GradientTape(persistent=True) as tape:
# 			mse = tf.reduce_mean( tf.square( Y - self.m( X ) ) )
# 			reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( x ) ) for x in self.m.trainable_variables ) )
# 			los = mse + self.c * reg
#
#
# 			J = tape.gradient(los, self.m.trainable_variables )
# 			res = tf.reduce_sum( flatten(J)*x)
#
# 		H = tape.gradient(res, self.m.trainable_variables)
#
# 		H = flatten(H)
# 		return H + x*0.001
#
#
# 	#d  = tf.linalg.experimental.conjugate_gradient(Axp, - flatten(tf.gradients(los, self.m.trainable_variables) ) )
# 	#d  = cg(Axp, - flatten(tf.gradients(los, self.m.trainable_variables) ) )
# 	d = cg(Axp, -flatten(JJ))
#
#
#
#
#
# 	constraint_approx = 0.5*    tf.reduce_sum( d * Axp(d))              #step_direction.dot(get_hessian_vector_product(step_direction))
# 	maximal_step_length = np.sqrt(0.01 / constraint_approx)
# 	full_step = maximal_step_length*d
#
#
# 	old = flatten(self.m.trainable_variables)
# 	new = old + d
#
# 	reshape(self.m.trainable_variables, new,True)
# 	mse = tf.reduce_mean( tf.square( Y - self.m( X ) ) )
# 	reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( x ) ) for x in self.m.trainable_variables ) )
# 	los2nd = mse + self.c * reg
# 	return los1st, los2nd
# 	print( d )
#
# 	return 1,2


# y -> Hy
# @tf.function
# def mvp( o, θ, y, damping=0.001 ):
# 	"""v to mv."""
#
# 	J = tf.gradients( o, θ )
# 	H = flatten(tf.gradients( tf.reduce_sum( flatten(J)*y ), θ) )
#
# 	return H + y * damping


def model( shape_of_input ):
	dtype = tf.float32

	s = tf.keras.layers.Input( shape_of_input, dtype=dtype )
	x = tf.keras.layers.Dense( 64, activation='tanh', dtype=dtype )( s )
	x = tf.keras.layers.Dense( 64, activation='tanh', dtype=dtype )( x )
	v = tf.keras.layers.Dense( 1, dtype=dtype )( x )

	m = tf.keras.Model( inputs=s, outputs=v )

	return m


def tst():
	shape_of_input = 5
	size_of_batch = 10

	gae = GAE( model( shape_of_input ), 0.99, 1.0, 0.001, 0.1 )

	x = np.random.randn( 100, 5 )
	y = np.random.randn( 100, 1 )
	gae.fit1st( x, y )


class Categorical:
	def __init__( self ):
		pass

	@staticmethod
	def sample( prob_nk ):
		prob_nk = np.asarray( prob_nk )
		assert prob_nk.ndim == 2
		N = prob_nk.shape[ 0 ]
		csprob_nk = np.cumsum( prob_nk, axis=1 )
		return np.argmax( csprob_nk > np.random.rand( N, 1 ), axis=1 )


#
# class Categorical( ProbType ):
#
# 	def sampled_variable( self ):
# 		return T.ivector( 'a' )
#
# 	def prob_variable( self ):
# 		return T.matrix( 'prob' )


class Agent:

	@staticmethod
	def build_π_model( shape_of_input, shape_of_output ):
		x = tf.keras.layers.Input( shape_of_input )
		y = tf.keras.layers.Dense( 64, activation='tanh' )( x )
		y = tf.keras.layers.Dense( 64, activation='tanh' )( y )
		z = tf.keras.layers.Dense( shape_of_output, activation='softmax' )( y )

		m = tf.keras.Model( inputs=x, outputs=z )

		return m

	@staticmethod
	def build_A_model( shape_of_input ):
		x = tf.keras.layers.Input( shape_of_input )
		y = tf.keras.layers.Dense( 64, activation='tanh' )( x )
		y = tf.keras.layers.Dense( 64, activation='tanh' )( y )
		z = tf.keras.layers.Dense( 1 )( y )

		m = tf.keras.Model( inputs=x, outputs=z )

		return m

	@staticmethod
	def likelihood( a, p ):
		return tf.gather_nd( p, tf.stack( (tf.range( len( p ) ), a), axis=-1 ) )

	# return p[ :, a ]

	@staticmethod
	def loglikelihood( a, p ):

		return tf.math.log(
			tf.gather_nd( p, tf.stack( (tf.range( len( p ) ), a), axis=-1 ) )
		)

	# return tf.math.log( p[ :, a ] )

	@staticmethod
	def kl( p0, p1 ):
		return tf.reduce_sum( p0 * tf.math.log( p0 / p1 ), axis=1 )

	@staticmethod
	def entropy( p ):
		return tf.reduce_sum( p * tf.math.log( 1 / p ), axis=-1 )

	@staticmethod
	def sample_categorical_probability( p ):
		p = np.cumsum( p, axis=1 )
		p = np.argmax( p > np.random.uniform( size=(len( p ), 1) ), axis=1 )
		return p

	@staticmethod
	def maxprobability( p ):
		return tf.argmax( p, axis=1 )

	def act( self, observation, stochastic=True ):
		observation = observation[ None ]
		probability = self.p( observation )

		if stochastic:
			return self.sample_categorical_probability( probability )[ 0 ], dict( probability=probability[ 0 ] )
		else:
			return self.maxprobability( probability )[ 0 ], dict( probability=probability[ 0 ] )

	def __init__( self ):
		self.p = self.build_π_model( 4, 2 )
		self.b = GAE( self.build_A_model( 5 ), 0.99, 1.0, 0.001, 0.1 )

	def calculate( self, obs, acs, ads, ops, damping=0.001 ):

		def G():
			with tf.GradientTape() as tape:
				nps = self.p( obs )

				nps_in_log = self.loglikelihood( acs, nps )
				ops_in_log = self.loglikelihood( acs, ops )

				x = - tf.reduce_mean( tf.exp( nps_in_log - ops_in_log ) * ads )
				g = flatten( tape.gradient( x, self.p.trainable_variables ) )

				return g

		def L():
			nps = self.p( obs )

			nps_in_log = self.loglikelihood( acs, nps )
			ops_in_log = self.loglikelihood( acs, ops )

			x = - tf.reduce_mean( tf.exp( nps_in_log - ops_in_log ) * ads )

			d = tf.reduce_mean( self.kl( ops, nps ) )
			e = tf.reduce_mean( self.entropy( nps ) )

			return x + d + e

		def HVP( v ):
			s = damping * v

			with tf.GradientTape() as outer:
				with tf.GradientTape() as inner:
					p = self.p( obs )
					d = tf.reduce_mean( self.kl( tf.stop_gradient( p ), p ) )
					g = inner.gradient( d, self.p.trainable_variables )

				v = reshape( g, v, None )
				x = tf.reduce_sum( list( tf.reduce_sum( a * b ) for (a, b) in zip( g, v ) ) )
				g = flatten( outer.gradient( x, self.p.trainable_variables ) )


				return g + s

		return G, L, HVP


def xCG( fAx, b, its=10, tol=1e-10 ):
	p = b
	r = b
	x = tf.zeros( tf.shape( b ) )
	m = tf.reduce_sum( r * r )

	for i in range( its ):
		z = fAx( p )
		v = m / tf.reduce_sum( p * z )
		x = x + v * p
		r = r - v * z
		n = tf.reduce_sum( r * r )
		p = r + n / m * p
		m = n

		if m < tol:
			break

	return x

def xxCG( fAx, b, its=10, tol=1e-10 ):
	p = b
	r = b
	x = np.zeros(len(b))
	m = np.sum( r * r )

	for i in range( its ):
		z = fAx( p )#.numpy()
		v = m / np.sum( p * z )
		x = x + v * p
		r = r - v * z
		n = np.sum( r * r )
		p = r + n / m * p
		m = n

		if m < tol:
			break

	return x


def CG(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):

	p = b.copy()
	r = b.copy()
	x = np.zeros_like(b)
	rdotr = np.sum( r * r ) #r.dot(r)



	for i in range(cg_iters):


		z = f_Ax(p)
		v = rdotr / np.transpose(p)@z #np.sum( p * z )# p.dot(z)
		x += v*p
		r -= v*z
		newrdotr = r.dot(r)
		#mu = newrdotr/rdotr
		p = r + newrdotr/rdotr*p

		rdotr = newrdotr
		if rdotr < residual_tol:
			break


	return x





#
# def make_mlps(ob_space, ac_space, cfg):
# 	assert isinstance(ob_space, Box)
# 	hid_sizes = cfg["hid_sizes"]
# 	if isinstance(ac_space, Box):
# 		outdim = ac_space.shape[0]
# 		probtype = DiagGauss(outdim)
# 	elif isinstance(ac_space, Discrete):
# 		outdim = ac_space.n
# 		probtype = Categorical(outdim)
# 	net = Sequential()
# 	for (i, layeroutsize) in enumerate(hid_sizes):
# 		inshp = dict(input_shape=ob_space.shape) if i==0 else {}
# 		net.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
# 	if isinstance(ac_space, Box):
# 		net.add(Dense(outdim))
# 		Wlast = net.layers[-1].kernel
# 		Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
# 		net.add(ConcatFixedStd())
# 	else:
# 		net.add(Dense(outdim, activation="softmax"))
# 		Wlast = net.layers[-1].kernel
#
#
# 		Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
# 	policy = StochPolicyKeras(net, probtype)
#
#
#
#
#
# 	vfnet = Sequential()
# 	for (i, layeroutsize) in enumerate(hid_sizes):
# 		inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i==0 else {} # add one extra feature for timestep
# 		vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
# 	vfnet.add(Dense(1))
# 	baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
# 	return policy, baseline


if __name__ == '__main__':
	tst()
