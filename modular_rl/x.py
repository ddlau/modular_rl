import numpy as np

import tensorflow as tf

from scipy.signal import lfilter


def flatten( seq ):
	return tf.concat( list( tf.reshape( x, [ -1 ] ) for x in seq ), axis=0 )


def replace( seq, vec ):
	b = 0
	for x in seq:
		s = tf.shape( x )
		l = tf.reduce_prod( s )
		x.assign( tf.reshape( tf.slice( vec, [ b ], [ l ] ), s ) )
		b = b + l


def discount( x, decay, check=None ):
	y = lfilter( [ 1 ], [ 1, -decay ], x[ ::-1 ], axis=0 )[ ::-1 ]

	if check:
		v = 0
		for t in reversed( range( len( x ) ) ):
			assert y[ t ] == (v := v * decay + x[ t ])

	return y


class GAE:

	def __init__( self, net, γ, λ ):
		self.net = net
		self.γ = γ
		self.λ = λ

	def A( self, S, R, M ):
		assert len( R ) + 1 == len( S )
		assert len( M ) + 1 == len( S )

		V = self.net( S )
		δ = R + M * self.γ * V[ +1: ] - V[ :-1 ]
		A = discount( δ, self.γ * self.λ )

		return A


import tensorflow as tf


class GAE:

	def get_θ( self ):
		return tf.concat( list( tf.reshape( v, -1 ) for v in self.net.trainable_variables ), axis=0 )

	def set_θ( self, θ ):
		r = list()
		s = 0
		for v in self.net.trainable_variables:
			u = s + tf.reduce_prod( tf.shape( v ) )
			r.append( tf.reshape( θ[ s:u ], tf.shape( v ) ) )
			s = u

		self.net.set_weights( r )

	def __init__( self, net, γ, λ, c ):
		self.net = net
		self.γ = γ
		self.λ = λ

		self.c = c  # coefficient of regularization

	def A( self, S, R, M ):
		assert len( R ) + 1 == len( S )
		assert len( M ) + 1 == len( S )

		V = self.net( S )
		δ = R + M * self.γ * V[ :-1 ] - V[ +1: ]
		A = discount( δ, self.γ * self.λ, True )
		return A

	def fit1st( self, S, Y ):
		@tf.function
		def vng( θ ):
			replace( self.net.trainable_variables, θ )

			with tf.GradientTape() as tape:
				mse = tf.reduce_mean( tf.square( self.net( S ) - Y ) )
				reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( v ) ) for v in self.net.trainable_variables ) )
				los = mse + self.c * reg

			g = tape.gradient( los, self.net.trainable_variables )

			return los, flatten( g )

		import scipy.optimize

		scipy.optimize.fmin_l_bfgs_b( vng, flatten( self.net.trainable_variables ) )


class FNV:

	@staticmethod
	def build( shape_of_input ):
		I = tf.keras.layers.Input( shape_of_input )
		X = tf.keras.layers.Dense( 64, activation='tanh' )( I )
		X = tf.keras.layers.Dense( 64, activation='tanh' )( X )
		V = tf.keras.layers.Dense( 1 )( X )

		return tf.keras.Model( inputs=I, outputs=V )

	def __init__( self, shape_of_input, c=0.001, γ=0.99, λ=0.98 ):
		self.c = c
		self.γ = γ
		self.λ = λ
		self.net = self.build( shape_of_input )

	def set_from_flat( self ):
		...

	def infer( self, X ):
		return self.net( X )

	def train( self, X, Y ):
		with tf.GradientTape() as tape:
			L2reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( v ) ) for v in self.net.trainable_variables ) )
			mse = tf.reduce_mean( tf.pow( self.net( X ) - Y, 2 ) )
			loss = mse + self.c * L2reg

		prev_theta = flatten( self.net.trainable_variables )

		def loss_and_gradients( theta ):
			self.set_from_flat( theta )
			l, g = loss_and_grad( X, Y )
			return l, g

	####

	def update( self, *args ):
		def lossandgrad( th ):
			self.set_params_flat( th )
			l, g = self.f_lossgrad( *args )
			g = g.astype( 'float64' )
			return (l, g)

		losses_before = self.f_losses( *args )
		theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b( lossandgrad, thprev, maxiter=self.maxiter )
		del opt_info[ 'grad' ]
		print( opt_info )
		self.set_params_flat( theta )
		losses_after = self.f_losses( *args )
		info = OrderedDict()
		for (name, lossbefore, lossafter) in zip( self.all_losses.keys(), losses_before, losses_after ):
			info[ name + "_before" ] = lossbefore
			info[ name + "_after" ] = lossafter
		return info

		g = tape.gradient( L, self.net.trainable_variables )
		g = flatten( g )

		print( g.shape )


# def flatten( seq ):
# 	return np.concatenate( list( np.reshape( x, -1 ) for x in seq ) )
#
#
# def flatten( seq ):
# 	return tf.concat( list( tf.reshape( x, [ -1 ] ) for x in seq ), axis=0 )
#
#
# def restore( seq, vec ):
# 	res = list()
#
# 	s = 0
# 	for x in seq:
# 		u = s + tf.cast( tf.reduce_prod( tf.shape( x ) ), tf.int32 )
# 		# print( 'u', type(u), u,u.shape)
#
# 		print( f's={s}' )
# 		print( f'u={u}' )
# 		vvv = tf.slice( vec, [ s ], [ u - s ] )
#
# 		res.append( tf.reshape( vvv, tf.shape( x ) ) )
# 		s = u
#
# 	return res
#
#
# def restore( seq, vec ):
# 	since = 0
# 	for x in seq:
# 		s = tf.cast( tf.reduce_prod( tf.shape( x ) ), tf.int32 )
# 		x.assign( tf.reshape( tf.slice( vec, [ since ], [ s ] ), tf.shape( x ) ) )
# 		since += s


def test():
	x = tf.keras.layers.Input( 3 )
	y = tf.keras.layers.Dense( 4 )( x )
	y = tf.keras.layers.Dense( 5 )( y )
	y = tf.keras.layers.Dense( 6 )( y )
	z = tf.keras.layers.Dense( 7 )( y )

	m = tf.keras.Model( inputs=x, outputs=z )
	m( np.random.rand( 100, 3 ) )
	m.summary()

	def g( value ):
		a = flatten( m.trainable_variables )
		b = np.arange( len( a ) ).astype( np.float32 ) * value

		replace( m.trainable_variables, b )
		return flatten( m.trainable_variables )

	@tf.function
	def f( value ):
		a = flatten( m.trainable_variables )
		b = np.arange( len( a ) ).astype( np.float32 ) * value

		replace( m.trainable_variables, b )
		return flatten( m.trainable_variables )

	print( f'g(3): {g( 3 )}' )
	print( f'g(4): {g( 4 )}' )

	print( f'f(3): {f( 3 )}' )
	print( f'f(4): {f( 4 )}' )
	return

	def g():
		a = flatten( m.trainable_variables )
		print( len( a ), type( a ), tf.shape( a ) )

		b = np.arange( len( a ) ).astype( np.float32 )
		print( 'b.shape', b.shape )
		restore( m.trainable_variables, b )  ############################m.set_weights(restore(m.trainable_variables, b))
		#
		print( flatten( m.trainable_variables ).numpy() )
		return tf.shape( a )

	@tf.function
	def f():
		a = flatten( m.trainable_variables )
		print( len( a ), type( a ), tf.shape( a ) )

		b = np.arange( len( a ) ).astype( np.float32 )
		print( 'b.shape', b.shape )

		restore( m.trainable_variables, b )  ################################aaa = restore(m.trainable_variables, b)
		#
		# for xxx,yyy in zip( m.trainable_variables,aaa):
		# 	xxx.assign(yyy)
		# print( f'aaa: {type(aaa),aaa}')
		# m.set_weights(aaa)
		#
		print( flatten( m.trainable_variables ) )
		return tf.shape( a )

	print( f'shape-of-a: {g()}' )
	print( f'shape-of-a: {g()}' )
	print( f'shape-of-a: {f()}' )
	print( f'shape-of-a: {f()}' )
	return

	fnV = FNV( 10 )

	for x in fnV.net.trainable_variables:
		print( tf.shape( x ), x.dtype )

	x = np.random.normal( size=(100, 10) )
	y = np.random.normal( size=(100, 1) )

	fnV.train( x, y )


def explainedVar( p, y ):
	return 1 - np.var( y - p, axis=0 ) / np.var( y, axis=0 )


if __name__ == '__main__':
	test()
	exit()

	discount( np.array( [ 1, 10, 100, 1000, 10000, 100000, 1000000 ] ), 0.99, True )

	exit()
