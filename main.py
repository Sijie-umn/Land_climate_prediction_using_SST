from lcp_nonlinear_models import nonlinear_function
from lcp_linear_models import linear_function
import numpy as np
import pickle



def main():
    #Run gradient boosted tree
    g = nonlinear_function.nonlinear_gbt()
    g.run_gbt()
    #Run deep nets
    m = nonlinear_function.nonlinear_mlp()
    m.run_mlp()
    #Run PCR
    p = linear_function.linear_pcr()
    p.run_PCR()
    #Run adaptive Lasso
    a = linear_function.linear_alasso()
    a.run_alasso()
    #Run Lasso
    l = linear_function.linear_lasso()
    l.run_lasso()


if __name__ == '__main__':
    main()
