# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

class CIR_model:
    def __init__(self,x0,kappa,sigma,T,theta,trials,N=100):
        self.x0 = x0
        self.kappa = kappa
        self.sigma = sigma
        self.theta = theta
        self.trials = trials
        self.N = int(T*N)
        self.Delta = 1.0/N
        self.T = T
        self.CIR = False
        self.t = np.linspace(0.00, self.T, num=self.N+1)
        self.t_matrix = np.ones((self.trials, self.N+1), dtype = np.float)
        for __i in range(trials):
            self.t_matrix[__i,:] = self.t
        
    def EM_chain_rule(self, EM_Plot=False):
        from math import sqrt
        
        np.random.seed(100)

        # __delta determines the "speed" of the Brownian motion.  The random variable
        # of the position at time t, X(t), has a normal distribution whose mean is
        # the position at time t=0 and whose variance is delta**2*t.

        __dB = sqrt(self.Delta) * np.random.standard_normal((self.trials,self.N)) # Brownian increments
        
        self.CIR = {}
        
        self.CIR['em'] = np.zeros((self.trials, self.N + 1), dtype = np.float)
        self.CIR['chain'] = np.zeros((self.trials, self.N + 1), dtype = np.float)
        
        for __i in range(self.trials):
            self.CIR['em'][__i, 0] = self.x0
            self.CIR['chain'][__i,0] = sqrt(self.x0)

            __x_temp_em = self.x0
            __x_temp_chain = sqrt(self.x0)

            for __j in range(self.N):
                __f1 = self.kappa * (self.theta - __x_temp_em)
                __g1 = self.sigma *sqrt(abs(__x_temp_em))
                __x_temp_em = __x_temp_em + __f1 * self.Delta + __g1 * __dB[__i,__j]
                self.CIR['em'][__i, __j+1] = __x_temp_em

                __f2 = (4.0*self.kappa*self.theta - self.sigma**2)/(8.0*__x_temp_chain)-(self.kappa*__x_temp_chain)/2.0
                __g2 = self.sigma/2.0
                __x_temp_chain = __x_temp_chain + __f2 * self.Delta + __g2 * __dB[__i,__j]
                self.CIR['chain'][__i,__j+1] = __x_temp_chain

        __diff1 = np.sqrt(np.abs(self.CIR['em'][:,self.N])) - self.CIR['chain'][:,self.N]
        # calculation of the error between square root of EM and Transforming EM scheme
        __xdiff1 = np.linalg.norm(__diff1, ord=np.inf)
        
        __diff2 = self.CIR['em'][:,self.N] - (self.CIR['chain'][:,self.N]**2)
        # calculation of the error between EM and Transforming back EM scheme
        __xdiff2 = np.linalg.norm(__diff2, ord=np.inf)

        
        #plot
        if EM_Plot:
            # square root of EM and Transforming EM 
            fig,ax = plt.subplots()
            __p1 = plt.plot(self.t ,np.sqrt(self.CIR['em'][0,:]), 'b-')
            __p2 = plt.plot(self.t ,self.CIR['chain'][0,:], 'r--')

            plt.xlabel('t',fontsize = 16)
            plt.ylabel('$V_{t}$',fontsize = 16,rotation = 0,horizontalalignment='right')

            __lg1 = 'Square root of EM'
            __lg2 = 'Transforming EM'

            plt.legend([__lg1,__lg2])

            plt.title(u'Square root of EM and Transforming EM (e=%.4g)\n %d paths,$\\kappa$ = %.4g, $\\theta$ = %.4g, $\\sigma$ = %.4g, $\\delta$ = %.4g'
            % ( __xdiff1, self.trials , self.kappa, self.theta, self.sigma, self.Delta)
            )
            for __i in range(1,4):
                plt.plot(self.t ,np.sqrt(self.CIR['em'][__i,:]), 'b-')
                plt.plot(self.t ,self.CIR['chain'][__i,:], 'r--')

            plt.close(0)

            # EM and Transforming back EM
            fig,ax = plt.subplots()
            __p1 = plt.plot(self.t,self.CIR['em'][0,:], 'b-')
            __p2 = plt.plot(self.t,self.CIR['chain'][0,:]**2, 'r--')

            plt.xlabel('t',fontsize = 16)
            plt.ylabel('$X_{t}$',fontsize = 16,rotation = 0,horizontalalignment='right')

            __lg1 = 'Euler Method scheme'
            __lg2 = 'Transforming back EM'

            plt.legend([__lg1,__lg2])

            plt.title(u'EM and Transforming back EM (e=%.4g)\n %d paths,$\\kappa$ = %.4g, $\\theta$ = %.4g, $\\sigma$ = %.4g, $\\delta$ = %.4g'
            % ( __xdiff2, self.trials , self.kappa, self.theta, self.sigma, self.Delta)
            )
            
            for __i in range(1,4):
                plt.plot(self.t,self.CIR['em'][__i,:], 'b-')
                plt.plot(self.t,self.CIR['chain'][__i,:]**2, 'r--')
            
            plt.show()
        

    def value_function(self,**arg):
        if isinstance(self.CIR, bool):
            print('You need to run exact_solution first')
            return
       
        try:
            __A = arg['A']
            __B = arg['B']
            __x_1 = arg['x1'] # the selling point
            __x_2 = arg['x2'] # the buying point
            __discounted_factor = arg['discounted'] # the discounted factor
            __selling_cost = arg['cs'] # the selling cost
            __buying_cost = arg['cb'] # the buying cost
        except:
            print("check out the input parametres have the A, B, x1, x2, discounted")
        # # set the mpmath precision
        # if 'dps' in arg.keys() and arg['dps']:
        #     mmm.mp.dps = arg['dps']
        #     mmm.mp.pretty = True

        # construct the Apsi and Bphi function
        Apsi = lambda x: __A*self.__hyp1f1(__discounted_factor/self.kappa,
                (2.0*self.kappa*self.theta)/(self.sigma)**2.0,
                2.0*self.kappa*x/self.sigma**2.0)
        Bphi = lambda x: __B*self.__hyperU(__discounted_factor/self.kappa,
            (2.0*self.kappa*self.theta)/(self.sigma)**2.0,
            2.0*self.kappa*x/self.sigma**2.0
            )


        # calculate the theorical value of v1 in respect to x0
        __v1_theory = (Bphi(self.x0)+self.x0-__selling_cost)*(self.x0>=__x_1)+(Apsi(self.x0))*(self.x0<__x_1) # calculate the theorical value of v1 in respect to x0
        __v2_theory = Bphi(self.x0)*(self.x0>__x_2)+(Apsi(self.x0)-self.x0-__buying_cost)*(self.x0<=__x_2) # calculate the theorical value of v2 in respect to x0

        # calculation of the EM scheme of the value function
        __Apsi_Xem = np.zeros((self.trials, self.N+1),dtype = float)
        __Bphi_Xem = np.zeros((self.trials, self.N+1),dtype = float)
        
        for __i in range(self.trials):
            for __j in range(self.N+1):
                __Apsi_Xem[__i,__j] = Apsi(self.CIR['em'][__i,__j])
                __Bphi_Xem[__i,__j] = Bphi(self.CIR['em'][__i,__j])

        __temp_discounted_factor = np.exp(-__discounted_factor * self.t_matrix)

        __temp_J1_upper = __temp_discounted_factor * self.CIR['em'] <= __x_2
        __temp_J1_lower = __temp_discounted_factor * self.CIR['em'] > __x_2
        __temp_J1_switch = __temp_J1_upper * __temp_discounted_factor * (__Apsi_Xem-self.CIR['em']-__buying_cost)+__temp_J1_lower * __temp_discounted_factor * __Bphi_Xem
        __temp_J1 = __temp_discounted_factor * (self.CIR['em']-__selling_cost) + __temp_J1_switch

        
        __temp_J2_upper = __temp_discounted_factor * self.CIR['em'] >= __x_1
        __temp_J2_lower = __temp_discounted_factor * self.CIR['em'] < __x_1
        __temp_J2_switch = __temp_J2_upper * __temp_discounted_factor * (__Bphi_Xem+self.CIR['em']-__selling_cost)+__temp_J2_lower * __temp_discounted_factor * __Apsi_Xem
        __temp_J2 = __temp_discounted_factor * (-self.CIR['em']-__buying_cost) + __temp_J2_switch

        __temp_mean_J2 = np.mean(__temp_J2, axis=0)
        __temp_mean_J1 = np.mean(__temp_J1, axis=0)

        __temp_max_J2 = __temp_J2[0,0]-10
        __tau2 = 0.0
        __temp_max_J1 = __temp_J1[0,0]-10
        __tau1 = 0.0

        for __i in range(self.N+1):
            if __temp_mean_J2[__i] >= __temp_max_J2:
                __temp_max_J2 = __temp_mean_J2[__i]
                __tau2 = self.t[__i]
            
            if __temp_mean_J1[__i] >= __temp_max_J1:
                __temp_max_J1 = __temp_mean_J1[__i]
                __tau1 = self.t[__i]

        if 'text' in arg.keys() and arg['text']:
            self.__text_html(text="""
            $J_1$最大值: %.4f, 执行点$\\tau_1 $: %.2f<br>
            $J_2$最大值: %.4f, 执行点$\\tau_2 $: %.2f<br>
            $\\textit{v} _{1}^{\\textit{Theory}} $ = %.4f
            $\\textit{v} _{1}^{\\textit{Theory}} $ = %.4f
            """ % (__temp_max_J1, __tau1,__temp_max_J2, __tau2, __v1_theory,__v2_theory),
                             HTML=True
                            )
        
        self.CIR = False
        
        if 'output' in arg.keys() and arg['output']:
            return {'j1':__temp_max_J1, 'tau1':__tau1,'j2':__temp_max_J2, 'tau2':__tau2, 'v1':__v1_theory, 'v2':__v2_theory}
    
    def __hyp1f1(self,a,b,z,**arg):
        if 'tol' in arg.keys() and arg['tol']:
            __tol = arg['tol']
        else:
            __tol = 1e-5

        if 'nmin' in arg.keys() and arg['nmin']:
            __nmin = arg['nmin']
        else:
            __nmin = 10

        __term = z*a/b
        __output = 1 + __term
        __n = 1
        __an = a
        __bn = b
        
        while(__n < __nmin) or (abs(__term) > __tol):
            __n = __n + 1
            __an = __an + 1
            __bn = __bn + 1
            __term = z * __term * __an / __bn / __n
            __output = __output + __term

        return __output

    def __hyperU(self,a,b,z,**arg):
        from math import gamma
        return (gamma(1-b)/gamma(a-b+1)) * self.__hyp1f1(a,b,z,**arg)+(gamma(b-1)/gamma(a)) * z**(1-b) * self.__hyp1f1(a-b+1,2-b,z,**arg)


    def __text_html(self, **arg):
        from IPython.core.display import display
        from IPython.core.display import HTML
        
        if 'HTML' in arg.keys() and arg['HTML']:
            display(HTML(arg['text']))
        else:
            print(arg['text'])
        return