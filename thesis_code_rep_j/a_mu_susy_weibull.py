# We need to rewrite everything using functions so we can separate the data we'll use in the neural network. 
from scipy.stats import truncnorm 
import matplotlib.pyplot as plt   #libreries
import numpy as np
from time import perf_counter  #start counting compilation time  ------------------
# first part 
#Defining fixed parameters 
start = perf_counter() #start counting here ----------------------

theta = np.pi 
g_1 = 2 
g_c = ((np.tan(theta)**2)*g_1**2)/16
w, y = -1, 1 
m_tau = 1.777 #en GeV/c^2
## m_l will be defined below 
m_mu= 105.66 * 10**-3 #muon mass on GeV/c^2 
c= (4*np.pi)**2 #this factor is for the denominator of a_mu 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--g_c**2--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Weinberg angle Sw=theta_w
alpha = 1/137.036
sw = np.sqrt(0.23116)
e = np.sqrt(4*np.pi*alpha) 
g_1 = e/np.sin(sw)
g_csqr = ((np.tan(sw)**2)*(g_1**2))/16 

#%%%%%%%%%%%%%%%%%%%--for defining v--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m_z = 91.2 #GeV
v = (np.sin(2*sw)*m_z)/(2*np.sqrt(alpha*np.pi))
##########################--------------S_bmul y P_bmul----------#################################

########################----- let's define the values separately
#####################-----we need to know phi value
###################------we need this values also A_y, X_m y x_t
#################------ Similarly for A_y, we need the following parameters
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
####################### For this case we use UNIFORM distribution ######################################## 
import dist_trunc as dtr 
mu_SUSY, A_0 = dtr.weibull_trunc(1.14,-15000,15000,1600000,loc=-20000), dtr.weibull_trunc(1.14,50,5000,1600000)
m_s, M1divms = dtr.weibull_trunc(1.14,50,5000,1600000), dtr.weibull_trunc(1.14,0.2,5,1600000)
beta = dtr.weibull_trunc(1.14,np.arctan(1),np.arctan(60),1600000) #change the number of n samples implies you must change to the same values of mu1sqr1,etc

#################################################################################################
def a_mutot(mu_SUSY,A_0,m_s,M1divms,beta):
    # A_y
    def A_y(A_0,beta):
        A_y = (1/np.sqrt(2))*y*A_0*v*np.cos(beta)
        return A_y
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--For X_m--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def X_m(A_0,beta,mu_SUSY):
        X_m = (1/np.sqrt(2))*w*A_0*v*np.cos(beta) - mu_SUSY*m_mu*np.tan(beta)
        return X_m
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--For X_t--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #We still have X_t left to do 
    def X_t(A_0,beta,mu_SUSY):
        X_t = (1/np.sqrt(2))*A_0*v*np.cos(beta)-mu_SUSY*m_tau*np.tan(beta)
        return X_t

    phi= np.arctan(2*A_y(A_0,beta)/(X_m(A_0,beta,mu_SUSY)-X_t(A_0,beta,mu_SUSY)))
    #for mu_1, mu_2, tau_1, tau_2 en el S_bmul
    mu_1s = 3*np.cos(phi/2)
    mu_2s = np.cos(phi/2)
    tau_1s = -np.sin(phi/2)
    tau_2s = -3*np.sin(phi/2)

    S_bmul1=np.array([mu_1s])
    S_buml2=np.array([mu_2s])
    S_buml3=np.array([tau_1s])
    S_buml4=np.array([tau_2s])

    #for mu_1, mu_2, tau_1, tau_2 en el P_bmul
    mu_1p = np.cos(phi/2)
    mu_2p=3*np.cos(phi/2)
    tau_1p = -3*np.sin(phi/2)
    tau_2p = -np.sin(phi/2)


    P_bmul1=np.array([mu_1p])
    P_bmul2=np.array([mu_2p])
    P_bmul3=np.array([tau_1p])
    P_bmul4=np.array([tau_2p])
    #print(P_bmul1.shape)  ###-array (1,N)--###
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--Slpeton masses (m_l)--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #We choose a value because m_l= m_mui o m_taui ; i=1,,2 for instance mu_mu1**2, 
    # so with table 2 and defined values

    R=np.sqrt(4*(A_y(A_0,beta)**2)+(X_t(A_0,beta,mu_SUSY)-X_m(A_0,beta,mu_SUSY))**2)
    def m_mu1sqr1(m_s): #process used for the next 3 functions of m_l
        m_sadj1 = m_s.copy()
        negative_mask1 = m_sadj1**2 <= (-R-X_m(A_0,beta,mu_SUSY)-X_t(A_0,beta,mu_SUSY))/2
        k=0
        while negative_mask1.any(): #Use any for the True outcomes 'cause if you use .all() it means all must be True, with just one False
            m_srandtn1= dtr.weibull_trunc(1.14,50,5000,1600000) #loop won't execute, try lots of sampling here so you won't need too much loops
            rand_val = np.random.choice(m_srandtn1, size=negative_mask1.sum(), replace=False)
            # negative_mask.sum() ensures you'll have same shape values of rand_val as mask and doesn't repeat values with replace False
            m_sadj1[negative_mask1] = rand_val # updates values where mask is TRUE, both in the same shape
            negative_mask1 = m_sadj1**2 <= (-R-X_m(A_0,beta,mu_SUSY)-X_t(A_0,beta,mu_SUSY))/2 # They coincide with the positions of m_srandtn.                                                        #la posición de ms_randtn
            # Reevaluate the mask with the new values to confirm it’s resolved; if not, repeat the loop
            k = k + 1
        if k > 0:
            print("\nNumber of times the loop ran to change the value:", k)
        else:
            print("\nNo value change occurred, so the loop didn’t need to run")
        # print("\nUpdated m_s values:", m_sadj1)
        # m_mu1sqr1noadj = (1/2)*(2*m_s**2 + X_m(A_0,beta,mu_SUSY) + X_t(A_0,beta,mu_SUSY) + R)
        # print("\nm_mu1sqr2 Values not fixed:", m_mu1sqr1noadj)  
        m_mu1sqr1 = (1/2)*(2*m_sadj1**2 + X_m(A_0,beta,mu_SUSY) + X_t(A_0,beta,mu_SUSY) + R)
        return m_mu1sqr1
    def m_mu1sqr2(m_s): # Here we resolve the issue with the negative values that are being returned. similarly for the others
        m_sadj = m_s.copy()
        negative_mask = m_sadj**2 <= (R-X_m(A_0,beta,mu_SUSY)-X_t(A_0,beta,mu_SUSY))/2
        # print(negative_mask)
        # print("Number of True entries where m_s <= the inequality:", np.count_nonzero(negative_mask))
        # print("Number of False entries (positive values where the m_s equality holds):", negative_mask.size - np.count_nonzero(negative_mask))
        # i=0
        #max_iters = 200   # loop limit just if you want
        while negative_mask.any(): 
            m_srandtn= dtr.weibull_trunc(1.14,50,5000,1600000) 
            rand_val2=np.random.choice(m_srandtn,size=negative_mask.sum(), replace=False)
            m_sadj[negative_mask] = rand_val2
            negative_mask = m_sadj**2 <= (R-X_m(A_0,beta,mu_SUSY)-X_t(A_0,beta,mu_SUSY))/2                                                                 #la posición de ms_randtn
            # We re-evaluate the mask with the new values to confirm it’s resolved; if not, we repeat the loop
        #     i = i+1
        # if i > 0:
        #     print("\nNumber of times the loop ran to change the value:", i)
        # if negative_mask.any():
        #     print("\nWARNING: unresolved entries:")
        # print("\nUpdated m_s values:", m_sadj)
        m_mu1sqr2 = (1/2)*(2*m_sadj**2 + X_m(A_0,beta,mu_SUSY) + X_t(A_0,beta,mu_SUSY) - R)
        m_mu1sqr2noadj = (1/2)*(2*m_s**2 + X_m(A_0,beta,mu_SUSY) + X_t(A_0,beta,mu_SUSY) - R)
        # print("\nm_mu1sqr2 values not fixed:", m_mu1sqr2noadj)
        return m_mu1sqr2


    # For mu_tau1**2 y mu_tau2**2 ----------------------------------------------------------------
    def m_tau1sqr1(m_s):
        m_sadj3 = m_s.copy()
        negative_mask3 = m_sadj3**2 <= (X_m(A_0,beta,mu_SUSY)+X_t(A_0,beta,mu_SUSY)-R)/2
        l=0
        while negative_mask3.any(): 
            m_srandtn3= dtr.weibull_trunc(1.14,50,5000,1600000) 
            #m_sadj3[negative_mask3] = m_srandtn3[negative_mask3] 
            rand_val3 = np.random.choice(m_srandtn3, size=negative_mask3.sum(), replace=False)
            m_sadj3[negative_mask3] = rand_val3 
            negative_mask3 = m_sadj3**2 <= (X_m(A_0,beta,mu_SUSY)+X_t(A_0,beta,mu_SUSY)-R)/2    
            l = l+1
        if l > 0:
            print("\nNumber of times the loop ran to change the value:", l)
        else:
            print("\nNo value change occurred, so running the loop wasn’t required")
        # # print("\nUpdated m_s values:", m_sadj)
        m_tau1sqr1nadj = (1/2)*(2*m_s**2 - X_m(A_0,beta,mu_SUSY) - X_t(A_0,beta,mu_SUSY) + R)
        #print("\nValores de m_tau1sqr2 sin ajuste:", m_tau1sqr1nadj)
        m_tau1sqr1 = (1/2)*(2*m_sadj3**2 - X_m(A_0,beta,mu_SUSY) - X_t(A_0,beta,mu_SUSY) + R)
        return m_tau1sqr1 

    def m_tau1sqr2(m_s): # Here we resolve the issue with the negative values that are being returned. 
        m_sadj2 = m_s.copy()
        negative_mask2 = m_sadj2**2 <= (R+X_m(A_0,beta,mu_SUSY)+X_t(A_0,beta,mu_SUSY))/2

        # j=0
        while negative_mask2.any(): 
            m_srandtn2= dtr.weibull_trunc(1.14,50,5000,1600000)
            rand_val2 = np.random.choice(m_srandtn2, size=negative_mask2.sum(), replace=False)
            m_sadj2[negative_mask2] = rand_val2 
            negative_mask2 = m_sadj2**2 <= (R+X_m(A_0,beta,mu_SUSY)+X_t(A_0,beta,mu_SUSY))/2                                                               #la posición de ms_randtn
        #     j = j+1
        # if j > 0:
        #     print("\nNumber of times the loop ran to change the value:", j)
        # else:
        #     print("\nNo value change occurred, so running the loop wasn’t required")
        # print("\nUpdated m_s values:", m_sadj)

        m_tau1sqr2nadj = (1/2)*(2*m_s**2 - X_m(A_0,beta,mu_SUSY) - X_t(A_0,beta,mu_SUSY) - R)
        #print("\nm_tau1sqr2 values not fixed:", m_tau1sqr2nadj)
        m_tau1sqr2 = (1/2)*(2*m_sadj2**2 - X_m(A_0,beta,mu_SUSY) - X_t(A_0,beta,mu_SUSY) - R)
        return m_tau1sqr2

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--bino mass (m_B=m_N1)--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # E_ms=np.mean(m_s)
    # M1 = trunc_norm_sample(E_ms*0.2,E_ms*5,1600)
    M1 = np.array([m_s*0.2,m_s*5]) #For simplicity we shall use this way of M1
    def m_B(mu_SUSY,beta):
        m_B = M1 - (((m_z**2)*(sw**2)*(M1+mu_SUSY*np.sin(2*beta)))/(mu_SUSY**2 - M1**2))
        #print(m_B.shape)
        #print(m_mu1sqr1(m_s).shape)
        return m_B

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--F_1^N & F_2^N--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##########___Define the x functions_____##################
    x1 = m_B(mu_SUSY,beta)**2/m_mu1sqr1(m_s) #for m_mu1 
    nmask = x1 <= 0
    print("x1 values that produce a negative result", sum(nmask))

    x2 = m_B(mu_SUSY,beta)**2/m_mu1sqr2(m_s) #for m_mu2
    nmask2 = x2 <= 0
    print("x2 values — with the adjusted m_s that still produce a negative result", sum(nmask2))
    x3 = m_B(mu_SUSY,beta)**2/m_tau1sqr1(m_s) #for m_tau1(correction not needed for negative values)
    nmask3 = x3 <= 0
    print("x3 values that produce negative result", sum(nmask3))

    x4 = m_B(mu_SUSY,beta)**2/m_tau1sqr2(m_s) #for m_tau2 
    nmask4 = x4 <= 0 
    print("x4 values with the adjsusted m_s that still produce a negative result",sum(nmask4))
    def F_1N(x):
        return (2/((1-x)**4))*(1 - 6*x + 3*x**2 + 2*x**3 - 6*x**2*np.log(x))
    def F_2N(x):
        return (3/((1-x)**3))*(1 - x**2 + 2*x*np.log(x))
    a_mu1= ((g_csqr*m_mu)/((4*np.pi)**2))*((S_bmul1**2 + P_bmul1**2)*(m_mu/(6*m_mu1sqr1(m_s)))*F_1N(x1) - (S_bmul1**2 - P_bmul1**2)*(m_B(mu_SUSY,beta)/(3*m_mu1sqr1(m_s)))*F_2N(x1))
    a_mu2= ((g_csqr*m_mu)/((4*np.pi)**2))*((S_buml2**2 + P_bmul2**2)*(m_mu/(6*m_mu1sqr2(m_s)))*F_1N(x2) - (S_buml2**2 - P_bmul2**2)*(m_B(mu_SUSY,beta)/(3*m_mu1sqr2(m_s)))*F_2N(x2))
    a_mu3= ((g_csqr*m_mu)/((4*np.pi)**2))*((S_buml3**2 + P_bmul3**2)*(m_mu/(6*m_tau1sqr1(m_s)))*F_1N(x3) - (S_buml3**2 - P_bmul3**2)*(m_B(mu_SUSY,beta)/(3*m_tau1sqr1(m_s)))*F_2N(x3))
    a_mu4= ((g_csqr*m_mu)/((4*np.pi)**2))*((S_buml4**2 + P_bmul4**2)*(m_mu/(6*m_tau1sqr2(m_s)))*F_1N(x4) - (S_buml4**2 - P_bmul4**2)*(m_B(mu_SUSY,beta)/(3*m_tau1sqr2(m_s)))*F_2N(x4))
    a_mutot = a_mu1 + a_mu2 + a_mu3 + a_mu4
    return a_mutot
result = a_mutot(mu_SUSY,A_0,m_s,M1divms,beta)
#print("Total values of a_mu:\n", result)
#print(result.shape)
# Here we apply the condition to keep values bounded to the range that yields a solution
# between 3.15 and 4.15 sigmas of a_muSUSY (i.e., within the [3.15, 4.15]σ band for a_muSUSY) because that's what we're looking for
a_muvalid = np.array([])
a_mususy = 200.445e-11 #allowed limit that can solve the anomaly
tol = 152.145e-11
a_muvalid = np.append(a_muvalid, result[(tol <= result) & (result <= a_mususy)])
# print("The values that satisfy the condition (i.e., yield a solution between 3.15 and 4.15 sigmas of a_muSUSY):\n", a_muvalid)
# First, a_mutot has shape (2, 100) while the masks are (N,) (1-D). Even if we flatten,
# mu_SUSY would be (100,) and a_mutot would become (200,), which don’t match.
# Therefore, we need to split a_mutot and compare row by row, so we end up with two (100,) arrays.

# Work with each row individually
result_row1 = result[0, :]
result_row2 = result[1, :]
# Make the masks for each row 
mask_row1 = (result_row1 >= tol) & (result_row1 <= a_mususy)
mask_row2 = (result_row2 >= tol) & (result_row2 <= a_mususy)

# Apply masks to mu_SUSY
filtered_mususy_row1 = mu_SUSY[mask_row1]
filtered_mususy_row2 = mu_SUSY[mask_row2]
#print(filtered_mususy_row1.shape, "filtered_mususy_row1")
# Concatenate the filtered results of both rows
combined_filtered_mususy = np.concatenate((filtered_mususy_row1, filtered_mususy_row2))

# Apply masks to the other arrays and concatenate
filtered_A0_row1 = A_0[mask_row1]
filtered_A0_row2 = A_0[mask_row2]
combined_filtered_A0 = np.concatenate((filtered_A0_row1, filtered_A0_row2))

filtered_ms_row1 = m_s[mask_row1]
filtered_ms_row2 = m_s[mask_row2]
combined_filtered_ms = np.concatenate((filtered_ms_row1, filtered_ms_row2))

filtered_m1divms_row1 = M1divms[mask_row1]
filtered_m1divms_row2 = M1divms[mask_row2]
combined_filtered_m1divms = np.concatenate((filtered_m1divms_row1, filtered_m1divms_row2))

filtered_beta_row1 = beta[mask_row1]
filtered_beta_row2 = beta[mask_row2]
combined_filtered_beta = np.concatenate((filtered_beta_row1, filtered_beta_row2))
# Print the combined filtered results to recover the full set of parameters,
# rather than a partition of them from a_mutot.
#Joining all parameters in one array but with subsets
filtered_parameters = list(zip(combined_filtered_mususy, combined_filtered_A0, combined_filtered_ms, combined_filtered_m1divms, combined_filtered_beta))
#print("\nfiltered values of mu_SUSY, A0, m_s, M1divms & beta that satisfy the condition but grouped into subsets of their union:" , "\n",filtered_parameters)
end = perf_counter()
print(f'Duración(s): {end-start}')
#We create a file with the generated values: A0, beta, M1/ms, ms^{~}, mu_susy and a_muvalid related to them
#Print the a_mu generated from those parameters also 
long = len(combined_filtered_A0)
with open('Generated values of the 5 parameters', 'w')  as archivo: #'w' es para escribir solamente, si quieres read el archivo y escribir, es 'w+'
    archivo.write("|       A0         |---|        beta       |---|      m1divms     |---|          ms         |---|       mususy     |===>|       a_muvalid      |\n")  
    # la \t es un tabulador organizan el texto en columnas cuando se abre en un editor
    for i in range(long):
        archivo.write(f"{combined_filtered_A0[i]}\t"  #la f"" es una cadena formateada (f-string) permiten insertar expresiones o valores de variables directo dentro de la cadena
                      f"{combined_filtered_beta[i]}\t"
                      f"{combined_filtered_m1divms[i]}\t"
                      f"{combined_filtered_ms[i]}\t"
                      f"{combined_filtered_mususy[i]}\t"
                      f"{a_muvalid[i]}\n")
#aquí también creamos un archivo con los a_muvalid asociados a los parámetros del otro archivo individualmente

long_amu = len(a_muvalid) 
with open('a_mu values related to the 5 parameters', 'w') as archivo2:
    for i in range(long_amu):
        archivo2.write(f"{a_muvalid[i]}\n")
        
filtered_params_arr = np.array(filtered_parameters)
#print(f' Array shape of the filtered parameters that give $[3.15, 4.15]\sigma$: {filtered_params_arr.shape}')
#print(f'Array shape of the anomaly $a_muvalid$: {a_muvalid.shape}')  #(lots of row, 5 colmuns or values)
#print(filtered_params_arr)
#print(a_muvalid)
X = filtered_params_arr
y = a_muvalid  #this definitions are for the neural network 
#print(f'Filtered parameters Values: {X}')
#print(f'$a_muvalid$ related to this parameters{y}')
