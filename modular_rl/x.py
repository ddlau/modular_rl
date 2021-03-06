import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b
import time
from tensorflow_probability.python.optimizer import lbfgs_minimize as lbfgs
from tensorflow_probability.python.math import value_and_gradient
import scipy.optimize
np.set_printoptions(linewidth=3000, formatter={
	'float': '{:+.6f}'.format,

})
def flatten( seq ):
	return tf.concat( list( tf.reshape( x, [ -1 ] ) for x in seq ), axis=0 )


def replace( seq, vec ):
	b = 0
	for x in seq:
		s = tf.shape( x )
		l = tf.reduce_prod( s )
		x.assign( tf.reshape( tf.slice( vec, [ b ], [ l ] ), s ) )
		b = b + l


#
# theta = np.linspace(0,100,10000)
# plt.plot( list(np.sum( np.square( x*t-z)) for t in theta ) )
# #plt.show()
#
# ###
# def lng(theta):
# 	return np.sum( np.square(x*theta[0]- z)), np.sum( 2*( x*theta[0]-z )*x )
# print( 'fmin_l_bfgs_b:')
# print( fmin_l_bfgs_b(lng, np.array([1.23456]),pgtol=1e-9,epsilon=1e-9,))
#
#
# print( 'minimize:')
# print( scipy.optimize.minimize(lambda theta:np.sum( np.square(x*theta[0]- z)), np.array([12345.6])) )
#
# return
# ###




import tensorflow



# yyy = tf.Variable([0.001234], dtype=tf.float32, trainable=True)
# z = tf.keras.layers.Dense(1,use_bias=False)(x)

def tst():
	shape_of_input = 3
	size_of_batch = 100

	x = tf.keras.layers.Input( shape_of_input )
	y = tf.keras.layers.Dense( 64, activation='tanh')(x)
	y = tf.keras.layers.Dense( 32, activation='tanh')(y)
	y = tf.keras.layers.Dense( 16, activation='tanh')(y)
	z = tf.keras.layers.Dense( 1 )(y)

	m = tf.keras.Model( inputs=x,outputs=z )
	#m.summary()

	theTheta = flatten(m.trainable_variables)

	#x = np.random.normal( size=(size_of_batch,shape_of_input) ).astype(np.float32)
	x = np.linspace(0,1,size_of_batch*shape_of_input).reshape( size_of_batch,shape_of_input)#.astype(np.float32)
	#z = np.sum( x[:,0]*3 +x[:,1]*30+x[:,2]*300, axis=1,keepdims=True )


	z = np.sum( x*np.array([[3,30,300]]), axis=1,keepdims=True)

	print( f'x.shape={x.shape}, z.shape={z.shape}')

	opt = tf.keras.optimizers.Adam(0.01)
	since = time.time()
	for i in range(100):
		with tf.GradientTape() as tape:
			loss = tf.keras.losses.mse(z,  m(x))


		opt.minimize( loss, m.trainable_variables, tape=tape )
		if i % 50==0:
			print(tf.reduce_mean(loss).numpy())

	print( f'cost: time', time.time()-since)
	print( 'm', m(x).numpy().reshape(-1))

	# print( (  x*15-z ))
	print( 'z', z.reshape(-1))

	print('-'*32)


	def LnG(theta):

		def f(theta):


			replace( m.trainable_variables, theta)
			tf.reduce_mean( tf.square( m(x)-z))

		return value_and_gradient(f)

		#
		# return value_and_gradient( lambda )
		#
		# with tf.GradientTape() as tape:
		# 	loss = tf.reduce_mean( tf.square( m(x)-z)) #tf.keras.losses.mse(z,  m(x))
		# gradient = tape.gradient(loss, m.trainable_variables)
		# gradient = flatten(gradient)
		#
		# print('.',end='')
		#
		# #print( f'LnG: ', type(theta), theta.shape, type(loss), loss, type(gradient), gradient.shape)
		# return loss, gradient

	since = time.time()
	res= lbfgs( LnG, theTheta, max_iterations=10000)
	print(res)
	print(res.converged.numpy())
	print(res.num_objective_evaluations.numpy())
	print(res.objective_value.numpy())
	print( f'cost: time', time.time()-since)

	#replace(m.trainable_variables, res.position)
	# print( 'xxx', tf.reduce_mean(tf.square(x*yyy-z)))
	# print('xxx',tf.reduce_mean(tf.square(x*15-z)))
	#
	#
	print( 'm', m(x).numpy().reshape(-1))

	# print( (  x*15-z ))
	print( 'z', z.reshape(-1))
	return


	losses = list()

	for i in range(500):
		with tf.GradientTape() as tape:
			loss = tf.keras.losses.mse(z,m(x)) #tf.reduce_mean( tf.square( m(x) - z ) )

		gradient = tape.gradient(loss, m.trainable_variables )


		opt.apply_gradients(zip(gradient, m.trainable_variables ))

		losses.append(loss.numpy())

		if i % 50 == 0:
			print( loss.numpy())
			print( m.trainable_variables)

	print( x.reshape(-1))
	print( m(x).numpy().reshape(-1))
	print( z.reshape(-1))
	print( np.square(m(x).numpy().reshape(-1)   -z.reshape(-1)))
	#plt.plot(losses)
	#plt.show()




tst()
exit()



from scipy.signal import lfilter
from scipy.optimize import fmin_l_bfgs_b





def discount( x, decay, check=None ):
	y = lfilter( [ 1 ], [ 1, -decay ], x[ ::-1 ], axis=0 )[ ::-1 ]

	if check:
		v = 0
		for t in reversed( range( len( x ) ) ):
			assert y[ t ] == (v := v * decay + x[ t ])

	return y


class GAE:

	def __init__( self, net, γ, λ, c ):
		self.net = net
		self.γ = γ
		self.λ = λ

		self.c = c

	def A( self, S, R, M ):
		assert len( R ) + 1 == len( S )
		assert len( M ) + 1 == len( S )

		V = self.net( S )
		δ = R + M * self.γ * V[ +1: ] - V[ :-1 ]
		A = discount( δ, self.γ * self.λ )

		return A

	def los( self, S, Y ):

		mse = tf.reduce_mean( tf.square( self.net( S ) - Y ) )
		reg = tf.reduce_sum( list( tf.reduce_sum( tf.square( v ) ) for v in self.net.trainable_variables ) )

		loss = mse + self.c * reg

		return loss, mse, reg



	def fit1st( self, S, Y ):

		print('los 111111111111111', self.los(S,Y))

		a1 = self.net(S).numpy().reshape(-1)
		print(np.stack( (a1,Y), axis=1))

		opt = tf.keras.optimizers.SGD()#0.1)

		@tf.function
		def train(S,Y):
			with tf.GradientTape() as tape:
				los = tf.reduce_mean( tf.square( self.net(S) -Y))
				#los = self.los(S,Y)
			grd = tape.gradient(los, self.net.trainable_variables)

			opt.apply_gradients(zip(grd, self.net.trainable_variables))

		for i in range(10000):
			train(S,Y)


		print('los 222222222222222', self.los(S,Y))
		a1 = self.net(S).numpy().reshape(-1)
		print(np.stack( (a1,Y), axis=1))

		#print( self.net(S).numpy())
		print('iterations', opt.iterations)
		return






		# @tf.function
		# def JnG( θ ):
		# 	print( 'JnG:', type(θ))
		# 	replace( self.net.trainable_variables, tf.cast(θ,tf.float32) )
		#
		# 	los = self.los(S, Y )[0]
		# 	grd = flatten(tf.gradients( los, self.net.trainable_variables ))
		#
		# 	#print( 'JnG:', type(los), los, type(grd))
		#
		# 	return los, grd
		#
		# 	# a,b = JnG(θ)
		# 	# return a.numpy().astype(np.float64), b.numpy().astype(np.float64)
		# 	#
		#
		# print('los 111111111111111', self.los(S,Y))
		#
		# x = flatten(self.net.trainable_variables)
		#
		#
		x, f, d = fmin_l_bfgs_b( lambda xxx: list( x.numpy().astype(np.float64) for x in JnG(xxx)), x, maxiter=250 )
		#
		# #print(f, d)
		#
		# replace(self.net.trainable_variables, x.astype(np.float32) )
		#
		# print('los 222222222222222', self.los(S,Y))


def tst():

	shape_of_input = 1

	x = tf.keras.layers.Input(shape_of_input)
	y = tf.keras.layers.Dense(100,activation='tanh')(x)
	y = tf.keras.layers.Dense(32,activation='tanh')(y)
	# y = tf.keras.layers.Dense(16,activation='tanh')(y)
	z = tf.keras.layers.Dense(1)(y)

	m = tf.keras.Model(inputs=x,outputs=z )

	#x = np.random.normal(size=(3,shape_of_input)).astype(np.float32)
	x = np.linspace( 0.1,0.9, 3)[:,None]
	y = np.arange(3,shape_of_input+3,1)[:,None]
	z = np.sum( x * y, axis=1).astype(np.float32)

	print(x)
	print(y)
	print( z)
	#print( x )
	#print(z)

	gae = GAE(m, 0.99, 0.98, 0.00001 )
	gae.fit1st(x,z)


tst()
exit()







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
	# m( np.random.rand( 100, 3 ) )
	# m.summary()

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
