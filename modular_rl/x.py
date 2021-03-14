class lop( tf.linalg.LinearOperator):
	def __init__(self,):
		super().__init__(tf.float32, is_self_adjoint=True, is_positive_definite=True)

	def _shape(self):
		return tf.TensorShape( (len(g), len(g)) )

	def _matmul(self, x, adjoint=False, adjoint_arg=False):

		res = list()
		for c in tf.range( tf.shape(x)[1]):



		return None
	def matvec(self, x, adjoint=False, name="matvec"):
		return HVP(x)

dd = tf.linalg.experimental.conjugate_gradient( lop(), -g )


# tf.linalg.LinearOperator( tf.float32, matvec=HVP)
# tf.linalg.experimental.conjugate_gradient()


















def cpuCG( fAx, b, its=256, tol=1e-10 ):
	p = b
	r = b
	x = np.zeros( len( b ) )
	m = np.dot( r, r )

	for i in range( its ):
		z = fAx( p )
		v = m / np.dot( p, z )
		x = x + v * p
		r = r - v * z
		n = np.dot( r, r )
		p = r + n / m * p
		m = n

		#lv( f'---------------------------------------------------------------------------CG: err={m}, its={i}', np.linalg.norm(fAx(x)-b))
		if m < tol:
			lv( f'---------------------------------------------------------------------------CG: err={m}, its={i}', np.linalg.norm(fAx(x)-b))
			#lv( f'CG: err={m}, its={i}' )
			break
	else:
		lv( f'---------------------------------------------------------------------------CG: run out of iterations, err={m}, its={i}', np.linalg.norm(fAx(x)-b))

	return x
def tsttt():
	#from scipy.optimize import fmin_cg
	from scipy.sparse.linalg import cg, LinearOperator

	import scipy.sparse.linalg


	A = np.random.rand(100,100)
	A = A + A.T
	x = 100+ np.random.rand(100)
	b = A@x
	#print( b )

	def fAx(v):
		return A@ v


	res = scipy.sparse.linalg.cg(A,b)
	print( 'result', res[1], 'error', np.linalg.norm( res[0]-x ) )

	func = LinearOperator( dtype=np.float64, shape=(100,100),matvec=fAx)
	res = scipy.sparse.linalg.cg(func,b)
	print( 'result', res[1], 'error', np.linalg.norm( res[0]-x ) )


	res = cpuCG( fAx, b, )
	print( np.linalg.norm( res - x))

tsttt()
exit()



































def xCG( fAx, b, its=10, tol=1e-10 ):
	"""exact match, workable,"""
	p = b
	r = b
	x = tf.zeros( tf.shape( b ) )
	m = np.dot( r, r )

	for i in range( its ):
		z = fAx( p )
		v = m / np.dot( p, z )
		x = x + v * p
		r = r - v * z
		n = np.dot( r, r )
		p = r + n / m * p
		m = n

		if m < tol:
			break

	print( '!!!############CG', m )
	return x










def ok1CG( fAx, b, its=10, tol=1e-10 ):
	"""exactly match, sdd = CG( fisher_vector_product, -g )"""
	p = b
	r = b
	x = tf.zeros( tf.shape( b ) )
	m = (r[ None, : ] @ r[ :, None ])[ 0, 0 ]

	for i in range( its ):
		z = fAx( p )
		v = m / (p[ None, : ] @ z[ :, None ])[ 0, 0 ]
		x = x + v * p
		r = r - v * z
		n = (r[ None, : ] @ r[ :, None ])[ 0, 0 ]
		p = r + n / m * p
		m = n

		if m < tol:
			break

	print( '#######################CG', m )
	return x


def ok2CG( fAx, bbb, its=10, tol=1e-10 ):
	"""exactly match, sdd = CG( fisher_vector_product, -g )"""
	b = tf.Variable( bbb )

	p = b
	r = b
	x = tf.zeros( tf.shape( b ) )

	m = tf.Variable( np.dot( bbb, bbb ) )
	# m = ( r[None,:]@ r[:,None])[0,0]
	# mm = tf.experimental.numpy.dot(r,r)
	# mmm = np.dot(bbb,bbb)
	# print(f'############## m', m.numpy())
	# print(f'############# mm', mm)
	# print(f'############ mmm', mmm)

	for i in range( its ):

		xxxx = p.numpy()
		xxx = fAx( xxxx )
		z = tf.Variable( xxx )
		# z = tf.Variable(fAx(p))

		ppp = np.asarray( p )
		zzz = np.asarray( z )
		ggg = np.dot( ppp, zzz )
		v = m / tf.Variable( ggg )  # (p[None,:]@z[:,None])[0,0]
		x = x + v * p
		r = r - v * z
		n = tf.Variable( np.dot( r, r ) )  # ( r[None,:]@ r[:,None])[0,0]
		p = r + n / m * p
		m = n

		if m < tol:
			break

	print( '#######################CG', m )
	return x.numpy()


def tfCG( fAx, b, its=10, tol=1e-10 ):
	"""tf."""

	p = b
	r = b
	x = tf.zeros( tf.shape( b ) )
	m = tf.experimental.numpy.dot( b, b )

	for i in range( its ):
		z = fAx( p )
		v = m / tf.experimental.numpy.dot( p, z )
		x = x + v * p
		r = r - v * z
		n = tf.experimental.numpy.dot( r, r )
		p = r + n / m * p
		m = n

		if m < tol:
			break

	return x


def moreorlessCG( fAx, b, its=10, tol=1e-10 ):
	p = b
	r = b
	x = np.zeros( len( b ) )
	m = np.dot( b, b )

	for i in range( its ):
		z = np.asarray( fAx( p ) )
		v = m / np.dot( p, z )
		x = x + v * p
		r = r - v * z
		n = np.dot( r, r )
		p = r + n / m * p
		m = n

		if m < tol:
			break


	print( '#######################CG', m )
	return x





# def CG( fAx, bbb, its=10, tol=1e-10):
# 	"""exactly match, sdd = CG( fisher_vector_product, -g )"""
# 	b = tf.Variable(bbb)
#
#
# 	p = b
# 	r = b
# 	x = tf.zeros( tf.shape(b))
#
# 	m = tf.experimental.numpy.dot(b,b)        #tf.Variable( np.dot(bbb,bbb) )
# 	# m = ( r[None,:]@ r[:,None])[0,0]
# 	# mm = tf.experimental.numpy.dot(r,r)
# 	# mmm = np.dot(bbb,bbb)
# 	# print(f'############## m', m.numpy())
# 	# print(f'############# mm', mm)
# 	# print(f'############ mmm', mmm)
#
# 	for i in range(its):
#
# 		xxxx = p.numpy()
# 		xxx = fAx(xxxx)
# 		z = tf.Variable(xxx)
# 		#z = tf.Variable(fAx(p))
#
# 		#ppp = np.asarray(p)
# 		#zzz = np.asarray(z)
# 		#ggg = np.dot(ppp,zzz)
# 		#v = m / tf.Variable(ggg ) #(p[None,:]@z[:,None])[0,0]
#
#
# 		v = m / tf.experimental.numpy.dot( p,z)
#
#
#
#
# 		x = x + v * p
# 		r = r - v * z
# 		n =  tf.experimental.numpy.dot(r,r)  #                   tf.Variable( np.dot(r,r) )# ( r[None,:]@ r[:,None])[0,0]
# 		p = r + n / m * p
# 		m = n
#
# 		if m < tol:
# 			break
#
# 	print( '#######################CG', m)
# 	return x.numpy()


def qqCG( fAx, b, its=10, tol=1e-10 ):
	p = b
	r = b
	x = tf.zeros( tf.shape( b ) )

	# mm = tf.experimental.numpy.dot(r,r) #
	m = (r[ None, : ] @ r[ :, None ])[ 0, 0 ]  # tf.reduce_sum( r * r )
	# print( f'mm-m', mm-m)

	for i in range( its ):
		z = fAx( p )

		x1 = (p[ None, : ] @ z[ :, None ])[ 0, 0 ]
		x2 = tf.experimental.numpy.dot( p, z )

		x3 = np.dot( p, z )
		print( f'x1,x2,x3', x1, x2.numpy(), x3 )
		# print( f'x1-x2', x1-x2)

		v = m / x2  # (p[None,:]@z[:,None])[0,0]  #tf.reduce_sum( p * z )
		x = x + v * p
		r = r - v * z

		x3 = (r[ None, : ] @ r[ :, None ])[ 0, 0 ]
		# x4 =tf.experimental.numpy.dot(r,r)
		# print( f'x3-x4', x3-x4)
		n = x3  # (r[None,:]@r[:,None])[0,0] #tf.reduce_sum( r * r )
		p = r + n / m * p
		m = n

		if m < tol:
			break

	print( '#######################CG', m )
	return x


def xxCG( fAx, b, its=10, tol=1e-10 ):
	p = b
	r = b
	x = np.zeros( len( b ) )
	m = np.sum( r * r )

	for i in range( its ):
		z = fAx( p )  # .numpy()
		v = m / np.sum( p * z )
		x = x + v * p
		r = r - v * z
		n = np.sum( r * r )
		p = r + n / m * p
		m = n

		if m < tol:
			break

	return x


def xxxCG( f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10 ):
	p = b.copy()
	r = b.copy()
	x = np.zeros_like( b )
	rdotr = (r[ None, : ] @ r[ :, None ])[ 0, 0 ]  # np.sum( r * r ) #r.dot(r)

	for i in range( cg_iters ):

		z = f_Ax( p )

		xxx = (p[ None, : ] @ z[ :, None ])[ 0, 0 ]

		v = rdotr / xxx  # np.sum( p * z )# p.dot(z)
		x += v * p
		r -= v * z
		newrdotr = r.dot( r )
		# mu = newrdotr/rdotr
		p = r + newrdotr / rdotr * p

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

#
# def bls( η, θ, s, expected, qualified=1 / 10, backtracks=1 * 10 ):
# 	r_at_first = η( θ )
# 	for f in np.power( 1 / 2, np.arange( backtracks ) ):
# 		x = θ + s * f
# 		r = η( x )
#
# 		a = r_at_first - r
# 		e = expected * f
# 		r = a / e
# 		if r > qualified and a > 0:
# 			return True, x
# 	else:
# 		return False, x

#
# def bls( l, θ, s, expected, qualified=1 / 10, backtracks=1 * 10 ):
# 	lv( f'calling bls:' )
#
# 	u = l( θ )
# 	lv( '\t', f'first: {u}' )
#
# 	for f in np.power( 1 / 2, np.arange( backtracks ) ):
# 		x = θ + s * f
# 		v = l( x )
# 		a = u - v
# 		e = f * expected
# 		r = a / e
# 		if r > qualified and a > 0:
# 			lv( '\t', f'final: {v}' )
# 			return True, x
#
# 	lv( '\t', f'failed' )
# 	return False, θ

#
# def bls( l, θ, s, expected, qualified=1 / 10, backtracks=1 * 10 ):
# 	u = l( θ )
# 	lv( f'first: {u}' )
#
# 	for f in np.power( 1 / 2, np.arange( backtracks ) ):
# 		x = θ + s * f
# 		v = l( x )
# 		a = u - v
# 		e = f * expected
# 		r = a / e
# 		if r > qualified and a > 0:
# 			lv( f'final: {v}' )
# 			return x
#
# 	lv( f'failed' )
# 	return θ

#
#
#
# def bls( η, θ, s, initial, expected, qualified=1/10, backtracks=1*10, descending_not_ascending=True ):
# 	if initial is None:
# 		initial = η(θ)
#
# 	for f in np.power( 1/2, np.arange( backtracks)):
# 		x = θ + s * f
# 		v = η(x)
# 		a = ( initial - v ) if descending_not_ascending else ( v - initial)
# 		e = expected * f
# 		r = a / e
# 		if r > qualified and a > 0:
# 			return x
#
# 	return θ
#
#
#
#
# def xxbls( η, θ, s, expected, qualified=1/10, backtracks=1*10):
#
# 	r_at_first = η(θ)
# 	print( f'before ', r_at_first )
# 	for f in np.power( 1/2, np.arange( backtracks)):
# 		x = θ + s * f
# 		r = η(x)
#
# 		a = r_at_first-r
# 		e = expected * f
# 		r = a / e
# 		if r > qualified and a > 0 :
# 			return True, x
# 	else:
# 		return False, x


# class Categorical:
# 	def __init__( self ):
# 		pass
#
# 	@staticmethod
# 	def sample( prob_nk ):
# 		prob_nk = np.asarray( prob_nk )
# 		assert prob_nk.ndim == 2
# 		N = prob_nk.shape[ 0 ]
# 		csprob_nk = np.cumsum( prob_nk, axis=1 )
# 		return np.argmax( csprob_nk > np.random.rand( N, 1 ), axis=1 )
#
# class Categorical( ProbType ):
#
# 	def sampled_variable( self ):
# 		return T.ivector( 'a' )
#
# 	def prob_variable( self ):
# 		return T.matrix( 'prob' )
