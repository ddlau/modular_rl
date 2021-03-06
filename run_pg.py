#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""

import os
os.environ['KERAS_BACKEND']='theano'
from keras import backend as K
K.set_image_dim_ordering('th')

from gym.envs import make
from modular_rl import *
import argparse, sys, _pickle as cPickle
from tabulate import tabulate
import shutil, os, logging
import gym

if __name__ == "__main__":

    # import tensorflow as tf
    #
    # p = tf.range(5)
    # q = tf.range(5)
    # #print( tf.matmul(p,q)) # error
    # # print( tf.multiply( p,q ))
    # #print( tf.experimental.numpy.dot(p,q) )
    # print( tf.tensordot(p,q,0))
    # exit()

    #
    # p = tf.random.uniform( (10,10) )
    # q = tf.random.uniform( (10,10))
    #
    # x1 = p/q
    # x2 = tf.exp( tf.math.log( p) - tf.math.log(q))
    # print( tf.reduce_max( tf.abs( x1-x2)))
    #
    # a = tf.range(5)
    # b = tf.range(5)*10
    # print( a*b)
    #
    # exit()
    #
    #
    # x = tf.reshape( tf.clip_by_value(tf.range(100)/100, 0.001,0.999),(10,10) )
    # print(x)
    #
    # a = x * tf.math.log( 1/x)
    # b = - x * tf.math.log(x)
    # print( 'aaaa', a)
    # print( 'b',b )
    #
    # print( tf.reduce_max( tf.abs( a-b)) )
    # exit()

    #
    # x = tf.Variable(np.arange(5))
    #
    # print( tf.multiply(x,x))
    # exit()


    ###
    # x = DiagGauss(3)
    #
    # a = x.sampled_variable()
    # print( a, type(a), vars(a) )

    # a,b,c=test_probtypes()
    # exit()
    ###







    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",default="CartPole-v0")#required=True)
    parser.add_argument("--agent",default="modular_rl.agentzoo.TrpoAgent")#required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = make(args.env)
    env_spec = env.spec
    #print( f'ddlau: env.spec: {env_spec}')
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)



    #env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    #env = gym.wrappers.Monitor(env, mondir, video_callable=lambda x:env.render())# if args.video else VIDEO_NEVER)



    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    print( f'ddlau: args.agent: {args.agent}, agent_ctor.options: {agent_ctor.options}')
    args = parser.parse_args()
    if args.timestep_limit == 0:
        #ddlau print( vars(env_spec))
        args.timestep_limit = env_spec.max_episode_steps#.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)













    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print ("*********** Iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%i ****************" % COUNTER)
        #print (tabulate(filter(lambda k,v : np.asarray(v).size==1, stats.items()))) #pylint: disable=W0110
        print (tabulate(filter(lambda x : np.asarray(x[1]).size==1, stats.items()))) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat,val) in stats.items():
                if np.asarray(val).ndim==0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)):
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception:
            print ("failed to pickle env" )#pylint: disable=W0703
    env.close()
