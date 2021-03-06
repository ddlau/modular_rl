import numpy as np

import tensorflow as tf

from scipy.signal import lfilter

tuple( map( lambda x: tf.config.experimental.set_memory_growth( x, True ), tf.config.list_physical_devices( 'GPU' ) ) )


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


import numpy as np

from tensorflow_probability.python.optimizer import lbfgs_minimize as lbfgs

from scipy.optimize import fmin_l_bfgs_b

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

	def fit1st( self, X, Y ):





		def LnG( θ ):
			replace( self.m.trainable_variables, θ )

			P = self.m( X )

			with tf.GradientTape() as tape:
				mse = tf.reduce_mean( tf.square( Y - P ) )
				reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( x ) ) for x in self.m.trainable_variables ) )

				loss = mse + self.c * reg
			gradient = flatten( tape.gradient( loss, self.m.trainable_variables ) )

			return loss, gradient


		def f(theta):
			loss, gradient = LnG(theta.astype(np.float32))
			print( 'me los, gr', loss, np.sum(gradient))
			return loss, gradient.numpy().astype(np.float64)



		x = flatten( self.m.trainable_variables )
		#print( type( x ), x.shape )



		res = fmin_l_bfgs_b(f, x, maxiter=3)
		#res = lbfgs( LnG, x, max_iterations=25 )
		print( 'res', res )


def model( shape_of_input ):
	s = tf.keras.layers.Input( shape_of_input, dtype=tf.float32 )
	x = tf.keras.layers.Dense( 64, activation='tanh', dtype=tf.float32 )( s )
	x = tf.keras.layers.Dense( 64, activation='tanh', dtype=tf.float32 )( x )
	v = tf.keras.layers.Dense( 1, dtype=tf.float32 )( x )

	m = tf.keras.Model( inputs=s, outputs=v )

	return m


def tst():
	shape_of_input = 5
	size_of_batch = 10

	gae = GAE( model( shape_of_input ), 0.99, 1.0, 0.001, 0.1 )

	x = np.random.randn( 100, 5 )
	y = np.random.randn( 100, 1 )
	gae.fit1st( x, y )


if __name__ == '__main__':
	tst()
