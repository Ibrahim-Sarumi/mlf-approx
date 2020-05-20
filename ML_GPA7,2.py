import numpy as np
import scipy.special as ss
    
def gpa_72aa(z,alf): #7,2
    z = np.abs(z)
    a = ss.gamma(-alf)/(ss.gamma(alf))
    b = ss.gamma(-alf)/(ss.gamma(2*alf))
    c = ss.gamma(-alf)/(ss.gamma(3*alf))
    d = ss.gamma(-alf)/(ss.gamma(4*alf))
    e = ss.gamma(-alf)/ss.gamma(5*alf)
    f = ss.gamma(-alf)/ss.gamma(-2*alf)
    
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
    
    denum = -ss.gamma(-alf)*(q0+q1*z+q2*(z**2)+q3*(z**3)+(z**4))
    return num/denum
    
def gpa_72(z, alf, bet): #7,2
    z = np.abs(z)
    if alf == 1 and bet == 1:
        return np.exp(-z)
    if alf < 1 and alf == bet:
        return gpa_72aa(z,alf)
    
    a = ss.gamma(bet-alf)/ss.gamma(bet)
    b = ss.gamma(bet-alf)/ss.gamma(bet+alf)
    c = ss.gamma(bet-alf)/ss.gamma(bet+2*alf)
    d = ss.gamma(bet-alf)/ss.gamma(bet+3*alf)
    e = ss.gamma(bet-alf)/ss.gamma(bet+4*alf)
    f = ss.gamma(bet-alf)/ss.gamma(bet+5*alf)
    g = ss.gamma(bet-alf)/ss.gamma(bet-2*alf)
    
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
    
    denum = ss.gamma(bet-alf)*(q0+q1*z+q2*(z**2)+q3*(z**3)+(z**4))
    
    return num/denum    