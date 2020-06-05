'''
This module consists of code for the global Pad\'e approximant of type (7,2) 
of the two-parametric Mittag-Leffler function E_{\alpha, \alpha}(-z), z >= 0.
The coefficients of the rational approximants have been obtained symbolically 
and are provided in this file.

Reference: Sarumi, Ibrahim O. and Furati, Khaled M. and Khaliq, Abdul Q. M.
           Highly Accurate Global {P}adé Approximations of Generalized 
           Mittag–Leffler Function and Its Inverse
           
           J Sci Comput, 2020
'''
import numpy as np
from scipy.special import  gamma
from scipy.signal import residue
    
def gpa_72aa(z,alf):
    '''
    Signature: gpa_72aa(z,alf)
    
    Returns: y (array_like), 
    an approximation to the value of two-parameteric Mittag-Leffler 
    function evaluated at z, E_{\alpha, \alpha}(-z), z >= 0, based on the 
    global Pad\'e approximant of type (7,2).
    
    Parameters
    -------------------
    z: array_like
        positive number(s)
    alf: scalar
        positive number in (0, 1)
    '''
    z = np.abs(z) # This line ensure that z is positive. Keep in mind, we seek E_{\alpha, \beta}(-z)
    a = gamma(-alf)/gamma(alf)
    b = gamma(-alf)/gamma(2*alf)
    c = gamma(-alf)/gamma(3*alf)
    d = gamma(-alf)/gamma(4*alf)
    e = gamma(-alf)/gamma(5*alf)
    f = gamma(-alf)/gamma(-2*alf)
    
    p2 = a*(a**4 - 2*a**2*c - a**2*d*f + 2*a*b**2 + 2*a*b*c*f - b**3*f - b*d + 
    c**2)/(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - b**4 
    + b**2*e - 2*b*c*d + c**3)
    
    p3 = (-a*b*e + a*c*d - a*(a**2*d - 2*a*b*c + b**3) + b**2*d - b*c**2 + 
    f*(-a**3*e + 2*a**2*b*d + a**2*c**2 - 3*a*b**2*c + b**4))/(a**3*e - 
    2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - b**4 + b**2*e - 
    2*b*c*d + c**3)
    
    q0 = (a**2*c - a*b**2 - a*(a**3 - a*c + b**2) + b*d - c**2 + f*(a**2*d - 
    2*a*b*c + b**3))/(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + 
    a*d**2 - b**4 + b**2*e - 2*b*c*d + c**3)
    
    q1 = (-a**3*b + a**2*d + a**2*e*f - a*b*d*f - a*c**2*f - b**3 + b**2*c*f + 
    b*e - c*d)/(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - 
    b**4 + b**2*e - 2*b*c*d + c**3)
    
    q2 = (-a**2*e + 2*a*b*d + a*(a**2*c - a*b**2 + b*d - c**2) - b**2*c + c*e - 
    d**2 + f*(a*b*e - a*c*d - b**2*d + b*c**2))/(a**3*e - 2*a**2*b*d - a**2*c**2
    + 3*a*b**2*c - a*c*e + a*d**2 - b**4 + b**2*e - 2*b*c*d + c**3)
    
    q3 = (-a*b*e + a*c*d - a*(a**2*d - 2*a*b*c + b**3) + b**2*d - b*c**2 + 
    f*(-a*c*e + a*d**2 + b**2*e - 2*b*c*d + c**3))/(a**3*e - 2*a**2*b*d - 
    a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - b**4 + b**2*e - 2*b*c*d + c**3)
    
    num = p2 + p3*z + z**2
    
    denum = -gamma(-alf)*(q0+q1*z+q2*(z**2)+q3*(z**3)+(z**4))
    return num/denum
    
def gpa_72(z, alf, bet):
    '''
    Signature: gpa_72(z,alf, bet)
    
    Returns: y (array_like),
    an approximation to the value of two-parameteric Mittag-Leffler 
    function evaluated at z, E_{\alpha, \beta}(-z), z >= 0, based on the 
    global Pad\'e approximant of type (7,2).
    
    Parameters
    -------------------
    z: array_like
        positive number(s)
    alf: scalar
        positive number in (0, 1]
    bet: scalar
        positive number such that bet > alf
    '''
    z = np.abs(z) # This line ensure that z is positive. Keep in mind, we seek E_{\alpha, \beta}(-z)
    if alf == 1 and bet == 1:
        return np.exp(-z)
    if alf < 1 and alf == bet:
        return gpa_72aa(z,alf)
    
    a = gamma(bet-alf)/gamma(bet)
    b = gamma(bet-alf)/gamma(bet+alf)
    c = gamma(bet-alf)/gamma(bet+2*alf)
    d = gamma(bet-alf)/gamma(bet+3*alf)
    e = gamma(bet-alf)/gamma(bet+4*alf)
    f = gamma(bet-alf)/gamma(bet+5*alf)
    g = gamma(bet-alf)/gamma(bet-2*alf)
    
    p1 = a*(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - 2*a*b*e + 2*a*c*d - 
    a*c*e*g + a*d**2*g - b**4 + 2*b**2*d + b**2*e*g - 2*b*c**2 - 2*b*c*d*g + 
    c**3*g + c*e - d**2)/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 
    2*a*c**2*e - 2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - 
    b*d*f + b*e**2 - c**4 + c**2*f - 2*c*d*e + d**3)
    
    p2 = (a**4*f - 2*a**3*b*e - 2*a**3*c*d + 3*a**2*b**2*d + 3*a**2*b*c**2 - 
    2*a**2*b*f + a**2*c*e - a**2*c*f*g + a**2*d**2 + a**2*d*e*g - 4*a*b**3*c + 
    3*a*b**2*e + a*b**2*f*g - 2*a*b*c*d - 2*a*b*d**2*g - a*c**3 + a*c**2*d*g + 
    a*c*f - a*d*e + b**5 - 2*b**3*d - b**3*e*g + 2*b**2*c**2 + 2*b**2*c*d*g - 
    b*c**3*g - b*c*e + b*d**2)/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 
    2*a*c**2*e - 2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - 
    b*d*f + b*e**2 - c**4 + c**2*f - 2*c*d*e + d**3)
    
    p3 = (a*d*f - a*e**2 - a*(a*c*f - a*d*e - b**2*f + b*c*e + b*d**2 - c**2*d) 
    - b*c*f + b*d*e - b*(-a*c*e + a*d**2 + b**2*e - 2*b*c*d + c**3) + 
    c**2*e - c*d**2 + g*(-a**2*d*f + a**2*e**2 + 2*a*b*c*f - 2*a*b*d*e - 
    2*a*c**2*e + 2*a*c*d**2 - b**3*f + 2*b**2*c*e + b**2*d**2 - 3*b*c**2*d + 
    c**4))/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 
    2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - 
    c**4 + c**2*f - 2*c*d*e + d**3)
    
    q0 = (a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - 2*a*b*e + 2*a*c*d - 
    a*c*e*g + a*d**2*g - b**4 + 2*b**2*d + b**2*e*g - 2*b*c**2 - 2*b*c*d*g + 
    c**3*g + c*e - d**2)/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 
    2*a*c**2*e - 2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - 
    b*d*f + b*e**2 - c**4 + c**2*f - 2*c*d*e + d**3)
    
    q1 = (-a*b*f + a*c*e + a*(a**2*f - 2*a*c*d + b*c**2 - b*f + d**2) + b*c*d -
    b*(a**2*e - a*b*d - a*c**2 + b**2*c - b*e + c*d) - c**3 + c*f - d*e - 
    g*(a*c*f - a*d*e - b**2*f + b*c*e + b*d**2 - c**2*d))/(a**2*d*f - a**2*e**2 
    - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 2*a*c*d**2 + b**3*f - 
    2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - c**4 + c**2*f - 
    2*c*d*e + d**3)
    
    q2 = (a*(a*b*f - a*c*e - b*c*d + c**3 - c*f + d*e) - b**2*f + 2*b*c*e - 
    b*(a*b*e - a*c*d - b**2*d + b*c**2 - c*e + d**2) - c**2*d + d*f - 
    e**2 - g*(a*d*f - a*e**2 - b*c*f + b*d*e + c**2*e - c*d**2))/(a**2*d*f - 
    a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 2*a*c*d**2 + b**3*f - 
    2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - c**4 + c**2*f - 
    2*c*d*e + d**3)
    
    q3 = (a*d*f - a*e**2 - a*(a*c*f - a*d*e - b**2*f + b*c*e + b*d**2 - c**2*d) 
    - b*c*f + b*d*e - b*(-a*c*e + a*d**2 + b**2*e - 2*b*c*d + c**3) + 
    c**2*e - c*d**2 + g*(-b*d*f + b*e**2 + c**2*f - 2*c*d*e + d**3))/(a**2*d*f 
    - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 2*a*c*d**2 + 
    b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - c**4 + 
    c**2*f - 2*c*d*e + d**3)
    
    num = p1 + p2*z + p3*(z**2) + z**3
    
    denum = gamma(bet-alf)*(q0+q1*z+q2*(z**2)+q3*(z**3)+(z**4))
    
    return num/denum    

def gpa_72aa_pf(alf):
    '''
    This code computes the residues (weights) and poles of the partial fraction 
    decomposition of the global Pad\'e approximant, type (7,2) for \alpha = \beta.
    
    Signature: gpa_72aa_pf(alf)
               
    Parameters: alf 
                positive number in (0, 1)
    
    Returns: r : ndarray
                 Residues (or Weights).
             p : ndarray
                 Poles
         The partial fraction decomposition can be assumed to take the form
         r[0]       r[1]        r[2]         r[3]
       -------- + -------- +  ---------  + ---------
       (s-p[0])   (s-p[1])    (s-p[2])     (s-p[3])  
    '''
    a = gamma(-alf)/gamma(alf)
    b = gamma(-alf)/gamma(2*alf)
    c = gamma(-alf)/gamma(3*alf)
    d = gamma(-alf)/gamma(4*alf)
    e = gamma(-alf)/gamma(5*alf)
    f = gamma(-alf)/gamma(-2*alf)
    
    p2 = a*(a**4 - 2*a**2*c - a**2*d*f + 2*a*b**2 + 2*a*b*c*f - b**3*f - b*d + 
    c**2)/(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - b**4 
    + b**2*e - 2*b*c*d + c**3)
    
    p3 = (-a*b*e + a*c*d - a*(a**2*d - 2*a*b*c + b**3) + b**2*d - b*c**2 + 
    f*(-a**3*e + 2*a**2*b*d + a**2*c**2 - 3*a*b**2*c + b**4))/(a**3*e - 
    2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - b**4 + b**2*e - 
    2*b*c*d + c**3)
    
    q0 = (a**2*c - a*b**2 - a*(a**3 - a*c + b**2) + b*d - c**2 + f*(a**2*d - 
    2*a*b*c + b**3))/(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + 
    a*d**2 - b**4 + b**2*e - 2*b*c*d + c**3)
    
    q1 = (-a**3*b + a**2*d + a**2*e*f - a*b*d*f - a*c**2*f - b**3 + b**2*c*f + 
    b*e - c*d)/(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - 
    b**4 + b**2*e - 2*b*c*d + c**3)
    
    q2 = (-a**2*e + 2*a*b*d + a*(a**2*c - a*b**2 + b*d - c**2) - b**2*c + c*e - 
    d**2 + f*(a*b*e - a*c*d - b**2*d + b*c**2))/(a**3*e - 2*a**2*b*d - a**2*c**2
    + 3*a*b**2*c - a*c*e + a*d**2 - b**4 + b**2*e - 2*b*c*d + c**3)
    
    q3 = (-a*b*e + a*c*d - a*(a**2*d - 2*a*b*c + b**3) + b**2*d - b*c**2 + 
    f*(-a*c*e + a*d**2 + b**2*e - 2*b*c*d + c**3))/(a**3*e - 2*a**2*b*d - 
    a**2*c**2 + 3*a*b**2*c - a*c*e + a*d**2 - b**4 + b**2*e - 2*b*c*d + c**3)
    
    return residue([0,0,1,p3,p2],[-gamma(-alf), -gamma(-alf)*q3, 
    -gamma(-alf)*q2, -gamma(-alf)*q1, -gamma(-alf)*q0])

    
def gpa_72_pf(alf, bet):
    '''
    This code computes the residues (weights) and poles of the partial fraction 
    decomposition of the global Pad\'e approximant, type (7,2), for \alpha < \beta.
    
    Signature: gpa_72_pf(alf, bet)
               
    Parameters: alf 
                positive number in (0, 1]
    
    Returns: r : ndarray
                 Residues (or Weights).
             p : ndarray
                 Poles
         The partial fraction decomposition can be assumed to take the form
         r[0]       r[1]        r[2]         r[3]
       -------- + -------- +  ---------  + ---------
       (s-p[0])   (s-p[1])    (s-p[2])     (s-p[3])  
    '''
    if alf < 1 and alf == bet:
        return gpa_72aa(alf)
    
    a = gamma(bet-alf)/gamma(bet)
    b = gamma(bet-alf)/gamma(bet+alf)
    c = gamma(bet-alf)/gamma(bet+2*alf)
    d = gamma(bet-alf)/gamma(bet+3*alf)
    e = gamma(bet-alf)/gamma(bet+4*alf)
    f = gamma(bet-alf)/gamma(bet+5*alf)
    g = gamma(bet-alf)/gamma(bet-2*alf)
    
    p1 = a*(a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - 2*a*b*e + 2*a*c*d - 
    a*c*e*g + a*d**2*g - b**4 + 2*b**2*d + b**2*e*g - 2*b*c**2 - 2*b*c*d*g + 
    c**3*g + c*e - d**2)/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 
    2*a*c**2*e - 2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - 
    b*d*f + b*e**2 - c**4 + c**2*f - 2*c*d*e + d**3)
    
    p2 = (a**4*f - 2*a**3*b*e - 2*a**3*c*d + 3*a**2*b**2*d + 3*a**2*b*c**2 - 
    2*a**2*b*f + a**2*c*e - a**2*c*f*g + a**2*d**2 + a**2*d*e*g - 4*a*b**3*c + 
    3*a*b**2*e + a*b**2*f*g - 2*a*b*c*d - 2*a*b*d**2*g - a*c**3 + a*c**2*d*g + 
    a*c*f - a*d*e + b**5 - 2*b**3*d - b**3*e*g + 2*b**2*c**2 + 2*b**2*c*d*g - 
    b*c**3*g - b*c*e + b*d**2)/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 
    2*a*c**2*e - 2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - 
    b*d*f + b*e**2 - c**4 + c**2*f - 2*c*d*e + d**3)
    
    p3 = (a*d*f - a*e**2 - a*(a*c*f - a*d*e - b**2*f + b*c*e + b*d**2 - c**2*d) 
    - b*c*f + b*d*e - b*(-a*c*e + a*d**2 + b**2*e - 2*b*c*d + c**3) + 
    c**2*e - c*d**2 + g*(-a**2*d*f + a**2*e**2 + 2*a*b*c*f - 2*a*b*d*e - 
    2*a*c**2*e + 2*a*c*d**2 - b**3*f + 2*b**2*c*e + b**2*d**2 - 3*b*c**2*d + 
    c**4))/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 
    2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - 
    c**4 + c**2*f - 2*c*d*e + d**3)
    
    q0 = (a**3*e - 2*a**2*b*d - a**2*c**2 + 3*a*b**2*c - 2*a*b*e + 2*a*c*d - 
    a*c*e*g + a*d**2*g - b**4 + 2*b**2*d + b**2*e*g - 2*b*c**2 - 2*b*c*d*g + 
    c**3*g + c*e - d**2)/(a**2*d*f - a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 
    2*a*c**2*e - 2*a*c*d**2 + b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - 
    b*d*f + b*e**2 - c**4 + c**2*f - 2*c*d*e + d**3)
    
    q1 = (-a*b*f + a*c*e + a*(a**2*f - 2*a*c*d + b*c**2 - b*f + d**2) + b*c*d - 
    b*(a**2*e - a*b*d - a*c**2 + b**2*c - b*e + c*d) - c**3 + c*f - d*e - 
    g*(a*c*f - a*d*e - b**2*f + b*c*e + b*d**2 - c**2*d))/(a**2*d*f - a**2*e**2 
    - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 2*a*c*d**2 + b**3*f - 
    2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - c**4 + c**2*f - 
    2*c*d*e + d**3)
    
    q2 = (a*(a*b*f - a*c*e - b*c*d + c**3 - c*f + d*e) - b**2*f + 2*b*c*e - 
    b*(a*b*e - a*c*d - b**2*d + b*c**2 - c*e + d**2) - c**2*d + d*f - 
    e**2 - g*(a*d*f - a*e**2 - b*c*f + b*d*e + c**2*e - c*d**2))/(a**2*d*f - 
    a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 2*a*c*d**2 + b**3*f - 
    2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - c**4 + c**2*f - 
    2*c*d*e + d**3)
    
    q3 = (a*d*f - a*e**2 - a*(a*c*f - a*d*e - b**2*f + b*c*e + b*d**2 - c**2*d) 
    - b*c*f + b*d*e - b*(-a*c*e + a*d**2 + b**2*e - 2*b*c*d + c**3) + 
    c**2*e - c*d**2 + g*(-b*d*f + b*e**2 + c**2*f - 2*c*d*e + d**3))/(a**2*d*f -
    a**2*e**2 - 2*a*b*c*f + 2*a*b*d*e + 2*a*c**2*e - 2*a*c*d**2 + 
    b**3*f - 2*b**2*c*e - b**2*d**2 + 3*b*c**2*d - b*d*f + b*e**2 - c**4 + 
    c**2*f - 2*c*d*e + d**3)
    
    return residue([0,1,p3,p2,p1],[gamma(bet-alf), gamma(bet-alf)*q3, 
    gamma(bet-alf)*q2, gamma(bet-alf)*q1, gamma(bet-alf)*q0])
