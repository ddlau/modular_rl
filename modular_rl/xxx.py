import numpy as np

import tensorflow as tf

from scipy.signal import lfilter

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


def reshape( vec, seq, replace=True ):
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


###


# its_at_all
# its_at_min

# break_when_fit
# return_final / return minimum
# its_refers_all / its_refers_min


def cg_via_np(
		f,
		b,
		its=10,
		tol=1e-10,
		verbose=None,
		callback=None,
		all_not_met=True,
		exit_when_fit=True,
		final_not_prime=True,
):
	n_at_prime=float('inf')
	x_at_prime=None



	p = np.copy(b)
	r = np.copy( b)
	x = np.zeros( np.shape( b ))
	n = np.dot( r, r )

	i = 0
	while i < its:
		z = f(p)
		v = n / np.dot( p, z )
		x = x+v*p
		r = r-v*z
		m = np.dot(r,r)
		u = m / n
		p = r + u * p
		n = m



		if n < tol:
			if exit_when_fit:
				break

		if new_prime_met := n < n_at_prime:
			n_at_prime = n
			x_at_prime = x

		if all_not_met or new_prime_met:
			i += 1



	if verbose:
		lv( f'x-at-final: {np.linalg.norm(x):>16.8f}')
		lv( f'x-at-prime: {np.linalg.norm(x_at_prime)}')


	return x if final_not_prime else x_at_prime





def conjugate_gradient(f_ax, b_vec, cg_iters=10, callback=None, verbose=False, r_tol=1e-10):

	p = b_vec.copy()  # the first basis vector
	r = b_vec.copy()  # the r
	x = np.zeros_like(b_vec)  # vector x, where Ax = b
	n = r.dot(r)  # L2 norm of the r

	fmt_str = "%10i %10.3g %10.3g"
	title_str = "%10s %10s %10s"
	if verbose:
		print(title_str % ("iter", "r norm", "soln norm"))

	for i in range(cg_iters):
		if callback is not None:
			callback(x)
		if verbose:
			print(fmt_str % (i, n, np.linalg.norm(x)))
		z_var = f_ax(p)
		v_var = n / p.dot(z_var)
		x += v_var * p
		r -= v_var * z_var
		m = r.dot(r)
		mu_val = m / n
		p = r + mu_val * p

		n = m
		if n < r_tol:
			break

	if callback is not None:
		callback(x)
	if verbose:
		print(fmt_str % (i + 1, n, np.linalg.norm(x)))
	return x






def conjugate_gradient_ok(f_ax, b_vec, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
	"""
	conjugate gradient calculation (Ax = b), bases on
	https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel p 312

	:param f_ax: (function) The function describing the Matrix A dot the vector x
				 (x being the input parameter of the function)
	:param b_vec: (numpy float) vector b, where Ax = b
	:param cg_iters: (int) the maximum number of iterations for converging
	:param callback: (function) callback the values of x while converging
	:param verbose: (bool) print extra information
	:param residual_tol: (float) the break point if the residual is below this value
	:return: (numpy float) vector x, where Ax = b
	"""
	first_basis_vect = b_vec.copy()  # the first basis vector
	residual = b_vec.copy()  # the residual
	x_var = np.zeros_like(b_vec)  # vector x, where Ax = b
	residual_dot_residual = residual.dot(residual)  # L2 norm of the residual

	fmt_str = "%10i %10.3g %10.3g"
	title_str = "%10s %10s %10s"
	if verbose:
		print(title_str % ("iter", "residual norm", "soln norm"))

	for i in range(cg_iters):
		if callback is not None:
			callback(x_var)
		if verbose:
			print(fmt_str % (i, residual_dot_residual, np.linalg.norm(x_var)))
		z_var = f_ax(first_basis_vect)
		v_var = residual_dot_residual / first_basis_vect.dot(z_var)
		x_var += v_var * first_basis_vect
		residual -= v_var * z_var
		new_residual_dot_residual = residual.dot(residual)
		mu_val = new_residual_dot_residual / residual_dot_residual
		first_basis_vect = residual + mu_val * first_basis_vect

		residual_dot_residual = new_residual_dot_residual
		if residual_dot_residual < residual_tol:
			break

	if callback is not None:
		callback(x_var)
	if verbose:
		print(fmt_str % (i + 1, residual_dot_residual, np.linalg.norm(x_var)))
	return x_var









def test_cg():
	dim=100
	its=100

	for _ in range(its):
		A = np.random.rand(dim)
		A = 1 / 2 * (A+A.T)
		x = np.random.rand(dim)
		b = np.dot(A,x)

		x1 = conjugate_gradient_ok( lambda x: np.dot(A,x), b )
		x2 = cg_via_np(lambda x: np.dot(A,x),b,)
		print( f'dif: {np.sum( np.abs(x1-x2))}')

test_cg()
if __name__ == '__main__':
	exit()
