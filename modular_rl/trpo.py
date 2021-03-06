from modular_rl import *

# ================================================================
# Trust Region Policy Optimization
# ================================================================

from .a import Agent, flatten, reshape, CG, bls

ppp = Agent()


class TrpoUpdater( EzFlat, EzPickle ):
	options = [
		("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
		("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
	]

	def __init__( self, stochpol, usercfg ):
		EzPickle.__init__( self, stochpol, usercfg )
		cfg = update_default_config( self.options, usercfg )  # damping0.001, maxkl0.01

		self.stochpol = stochpol
		self.cfg = cfg

		probtype = stochpol.probtype
		params = stochpol.trainable_variables
		EzFlat.__init__( self, params )

		ob_no = stochpol.input
		act_na = probtype.sampled_variable()  ##### 符号， 轨迹中的动作
		adv_n = T.vector( "adv_n" )  #### 符号，轨迹中的优势

		# Probability distribution:
		prob_np = stochpol.get_output()  # 网络输出，符号
		oldprob_np = probtype.prob_variable()  # prob 符号， 轨迹中的prob，采样概率，q

		logp_n = probtype.loglikelihood( act_na, prob_np )
		oldlogp_n = probtype.loglikelihood( act_na, oldprob_np )
		N = ob_no.shape[ 0 ]

		# Policy gradient:
		surr = (-1.0 / N) * T.exp( logp_n - oldlogp_n ).dot( adv_n )
		pg = flatgrad( surr, params )

		prob_np_fixed = theano.gradient.disconnected_grad( prob_np )  ##############################################不计算对prob_np的梯度
		kl_firstfixed = probtype.kl( prob_np_fixed, prob_np ).sum() / N
		grads = T.grad( kl_firstfixed, params )
		flat_tangent = T.fvector( name="flat_tan" )
		shapes = [ var.get_value( borrow=True ).shape for var in params ]
		start = 0
		tangents = [ ]
		for shape in shapes:
			size = np.prod( shape )
			tangents.append( T.reshape( flat_tangent[ start:start + size ], shape ) )
			start += size
		gvp = T.add( *[ T.sum( g * tangent ) for (g, tangent) in zipsame( grads, tangents ) ] )  # pylint: disable=E1111
		# Fisher-vector product
		fvp = flatgrad( gvp, params )

		ent = probtype.entropy( prob_np ).mean()
		kl = probtype.kl( oldprob_np, prob_np ).mean()

		losses = [ surr, kl, ent ]
		self.loss_names = [ "surr", "kl", "ent" ]

		args = [ ob_no, act_na, adv_n, oldprob_np ]

		self.compute_policy_gradient = theano.function( args, pg, **FNOPTS )
		self.compute_losses = theano.function( args, losses, **FNOPTS )
		self.compute_fisher_vector_product = theano.function( [ flat_tangent ] + args, fvp, **FNOPTS )

	def __call__( self, paths ):
		cfg = self.cfg
		prob_np = concat( [ path[ "prob" ] for path in paths ] )
		ob_no = concat( [ path[ "observation" ] for path in paths ] )
		action_na = concat( [ path[ "action" ] for path in paths ] )
		advantage_n = concat( [ path[ "advantage" ] for path in paths ] )
		args = (ob_no, action_na, advantage_n, prob_np)

		thprev = self.get_params_flat()

		reshape( ppp.p.trainable_variables, thprev, True )

		G, L, HVP = ppp.calculate( ob_no, action_na, advantage_n, prob_np )

		def fisher_vector_product( p ):
			r1 = self.compute_fisher_vector_product( p, *args )
			r2 = cfg[ "cg_damping" ] * p  # pylint: disable=E1101,W0640
			r3 = r1 + r2
			# r4 = HVP(p).numpy()
			# print( '################## dif', np.max( np.abs( r1-r4)))
			return r3

		g = self.compute_policy_gradient( *args )
		losses_before = self.compute_losses( *args )

		#   ggg= G()
		LLL = L()
		#  print( '################## g dif', np.max( np.abs( g-ggg.numpy())))
		print( '################## l dif', np.sum( losses_before ) - LLL.numpy(), losses_before, LLL )

		if np.allclose( g, 0 ):
			print( "got zero gradient. not updating" )
		else:
			stepdir = cg( fisher_vector_product, -g )
			###OK1 sdd = CG( fisher_vector_product, -g )#.numpy()  ################# CG(HVP,-g).numpy()

			# sdd = CG( HVP, -ggg).numpy()
			#########################################################

			beta = np.sqrt( 2 * cfg[ 'max_kl' ] / stepdir.dot( fisher_vector_product( stepdir ) ) )

			#########################################################

			# sdd = CG( fisher_vector_product, -g )

			# print( f'################## d dif', np.max( np.abs( stepdir-sdd )))

			shs = .5 * stepdir.dot( fisher_vector_product( stepdir ) )
			lm = np.sqrt( shs / cfg[ "max_kl" ] )

			print( "lagrange multiplier:", 1 / lm, "gnorm:", np.linalg.norm( g ), 'beta', beta )
			fullstep = stepdir / lm
			neggdotstepdir = -g.dot( stepdir )

			def loss( th ):
				self.set_params_flat( th )
				return self.compute_losses( *args )[ 0 ]  # pylint: disable=W0640

			success, theta = linesearch( loss, thprev, fullstep, neggdotstepdir / lm )
			x, ttt = bls( loss, thprev, fullstep, neggdotstepdir / lm )
			print( "success", success, 'bls vvvvvvvvvvvvvvvvvvvvvvvvvvs ls', np.max( np.abs( theta - ttt ) ) )
			self.set_params_flat( theta )
		losses_after = self.compute_losses( *args )

		out = OrderedDict()
		for (lname, lbefore, lafter) in zipsame( self.loss_names, losses_before, losses_after ):
			out[ lname + "_before" ] = lbefore
			out[ lname + "_after" ] = lafter
		return out


def linesearch( f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1 ):
	"""
	Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
	"""
	fval = f( x )
	print( "fval before", fval )
	for (_n_backtracks, stepfrac) in enumerate( .5 ** np.arange( max_backtracks ) ):
		xnew = x + stepfrac * fullstep
		newfval = f( xnew )
		actual_improve = fval - newfval
		expected_improve = expected_improve_rate * stepfrac
		ratio = actual_improve / expected_improve
		print( "a/e/r", actual_improve, expected_improve, ratio )
		if ratio > accept_ratio and actual_improve > 0:
			print( "fval after", newfval )
			return True, xnew
	return False, x





def cg( f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10 ):
	"""
	Demmel p 312
	"""
	p = b.copy()
	r = b.copy()
	x = np.zeros_like( b )
	rdotr = r.dot( r )

	fmtstr = "%10i %10.3g %10.3g"
	titlestr = "%10s %10s %10s"
	if verbose:
		print( titlestr % ("iter", "residual norm", "soln norm") )

	for i in range( cg_iters ):
		if callback is not None:
			callback( x )
		if verbose: print( fmtstr % (i, rdotr, np.linalg.norm( x )) )
		z = f_Ax( p )
		v = rdotr / p.dot( z )
		x += v * p
		r -= v * z
		newrdotr = r.dot( r )
		mu = newrdotr / rdotr
		p = r + mu * p

		rdotr = newrdotr
		if rdotr < residual_tol:
			break

	if callback is not None:
		callback( x )
	if verbose: print( fmtstr % (i + 1, rdotr, np.linalg.norm( x )) )  # pylint: disable=W0631

	print( '#######################cg', rdotr )
	return x
