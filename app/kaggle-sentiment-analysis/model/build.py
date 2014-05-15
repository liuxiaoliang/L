#! /usr/bin/env python
# -*- encoding: utf-8 -*-

"""build model

using machine learning method to build model.
"""

__author__ = "xiaoliang liu"

import sys
sys.path.append("../../../common")
sys.path.append("../../../math")
sys.path.append("../../../mllib")
sys.path.append("../../../thirdparty")

import gflags
from classify import bayes

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('algo', 0, 'choose ML algorithm (default 0).\n0: navie bayes')
gflags.DEFINE_bool('train', False, 'training')
gflags.DEFINE_bool('predict', False, 'predicting')
gflags.DEFINE_bool('eval', False, 'evaluating')
gflags.DEFINE_string('feature', '', 'feature file path')
gflags.DEFINE_string('sample', '', 'training or predicting file path')
gflags.DEFINE_string('output', '', 'model file path if trianing, or predciting result file if predicting')
gflags.DEFINE_string('model', '', 'model file for predicting')

if __name__ == '__main__':
    try:
        argv = FLAGS(sys.argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)
    
    m = None
    if FLAGS.train:
        if (not FLAGS.feature or
            not FLAGS.sample or
            not FLAGS.output):
            print "missing feature file or sample file or output file"
            sys.exit(1)

        if FLAGS.algo == 0:
            m = bayes.NavieBayes(FLAGS.feature, FLAGS.sample, FLAGS.output)
        # trian
        m.train()
    elif FLAGS.predict:
        if (not FLAGS.feature or
            not FLAGS.sample or
            not FLAGS.output or
            not FLAGS.model):
            print "missing feature file or sample file or output file or model file"
            sys.exit(1)
        if FLAGS.algo == 0:
            m = bayes.NavieBayes(FLAGS.feature, FLAGS.sample, FLAGS.output, FLAGS.model)
        # predict
        m.predict()
    
