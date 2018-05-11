# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

class geometric_brownian_motion:
    def __init__(self,x0,mu,sigma,T,trials,N=100):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.trials = trials
        self.N = int(T*N)
        self.Delta = 1.0/N
        self.T = T
        self.GBM = False
        
    def exact_solution(self, GBM_Plot=False):
        from math import sqrt
        
        __dW = np.empty((self.trials, self.N + 1),dtype=float)
        __dW[:,0] = 0.0
        
        __t = np.empty((self.trials,self.N+1),dtype = float)
        __t_for_theory = np.linspace(0.0,self.T,self.N + 1)
        self.t = __t_for_theory
        
        for __k in range(self.trials):
            __t[__k]= __t_for_theory
        self.t_matrix = __t
        # __delta determines the "speed" of the Brownian motion.  The random variable
        # of the position at time t, X(t), has a normal distribution whose mean is
        # the position at time t=0 and whose variance is delta**2*t.
        __delta = 1.0
        np.random.seed(100)
        __dB = sqrt(self.Delta) * np.random.standard_normal((self.trials,self.N))
        
        np.cumsum(__dB, axis = 1, out = __dW[:,1:self.N+1])
        
        __GBM = self.x0 * np.exp((self.mu-1./2.*self.sigma**2)*__t+self.sigma*__dW)
               
        __meanGBM = np.mean(__GBM,axis = 0)
        
        self.GBM = __GBM
        
        
        __theory_expected_value = np.exp(self.mu*__t_for_theory)
        
        __averr = np.linalg.norm((__meanGBM - __theory_expected_value),ord=np.inf)
        
        #plot
        if GBM_Plot:
            plt.plot(__t_for_theory,__meanGBM,'b-')
            for __k in range(5):
                plt.plot(__t_for_theory,__GBM[__k],'r--')

            plt.xlabel('t',fontsize = 16)
            plt.ylabel('$X_{t}$',fontsize = 16,rotation = 0,horizontalalignment='right')

            __lg1 = u'mean of {trials} paths'.format(trials = self.trials)
            __lg2 = u'{n} individual path'.format(n = 5)

            plt.legend([__lg1,__lg2])

            plt.title(u'Geometric Brownian motion(e=%.4g)\n %d paths,$\mu$ = %.4g, $\sigma$ = %.4g, $\delta$ = %.4g'
            % ( __averr, self.trials , self.mu, self.sigma, self.Delta)
            )        
            plt.show()
        

    def value_function(self,**arg):
        if isinstance(self.GBM, bool):
            print('You need to run exact_solution first')
            return
        
        # calculate the theorical value of the value function

        __temp_t = np.linspace(0.00, self.T, num=self.N+1)
        __t_matrix = np.ones((self.trials, self.N+1), dtype = np.float)
        for __i in range(self.trials):
            __t_matrix[__i,:] = __temp_t
        # calculate the theorical value of v1 in respect to x0
        __v1_theory = (self.x0-arg['cs'])*(self.x0>=arg['xswitch'])+(arg['psi'](self.x0))*(self.x0<arg['xswitch'])
        __discounted_matrix = np.exp(-arg['discounted'] * __t_matrix)
        __temp_J2_upper = (__discounted_matrix * self.GBM) >= arg['xswitch']
        __temp_J2_lower = (__discounted_matrix * self.GBM) < arg['xswitch']
        __temp_J2_switch = __temp_J2_upper * __discounted_matrix * (self.GBM-arg['cs'])+__temp_J2_lower * __discounted_matrix * arg['psi'](self.GBM)
        __temp_J2 = __discounted_matrix * (-self.GBM-arg['cb']) + __temp_J2_switch
        __temp_J1 = np.exp(-arg['discounted'] * __t_matrix) * (self.GBM-arg['cs'])

        __temp_mean_J2 = np.mean(__temp_J2, axis=0)
        __temp_mean_J1 = np.mean(__temp_J1, axis=0)

        __temp_max_J2 = __temp_J2[0,0]-10
        __tau2 = 0.0
        __temp_max_J1 = __temp_J1[0,0]-10
        __tau1 = 0.0

        for __i in range(self.N+1):
            if __temp_mean_J2[__i] >= __temp_max_J2:
                __temp_max_J2 = __temp_mean_J2[__i]
                __tau2 = __temp_t[__i]
            
            if __temp_mean_J1[__i] >= __temp_max_J1:
                __temp_max_J1 = __temp_mean_J1[__i]
                __tau1 = __temp_t[__i]

        if 'text' in arg.keys() and arg['text']:
            self.__text_html(text="""
            $J_1$最大值: %.4f, 执行点$\\tau_1 $: %.2f<br>
            $J_2$最大值: %.4f, 执行点$\\tau_2 $: %.2f<br>
            $\\textit{v} _{1}^{\\textit{Theory}} $ = %.4f
            """ % (__temp_max_J1, __tau1,__temp_max_J2, __tau2, __v1_theory),
                             HTML=True
                            )
        
        self.GBM = False
        
        if 'output' in arg.keys() and arg['output']:
            return {'j1':__temp_max_J1, 'tau1':__tau1,'j2':__temp_max_J2, 'tau2':__tau2, 'v1':__v1_theory}
    
    def __text_html(self, **arg):
        from IPython.core.display import display
        from IPython.core.display import HTML
        
        if 'HTML' in arg.keys() and arg['HTML']:
            display(HTML(arg['text']))
        else:
            print(arg['text'])
        return