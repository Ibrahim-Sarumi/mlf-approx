import numpy as np
from scipy.special import gamma
from scipy.signal import residue
from numpy.linalg import solve

def gpa10_15_6_ml_coff(alf, bet):
    a = -gamma(bet-alf)/gamma(bet)
    b =  gamma(bet-alf)/gamma(bet+alf)
    c = -gamma(bet-alf)/gamma(bet+2*alf)
    d =  gamma(bet-alf)/gamma(bet+3*alf)
    e = -gamma(bet-alf)/gamma(bet+4*alf)
    f =  gamma(bet-alf)/gamma(bet+5*alf)
    g = -gamma(bet-alf)/gamma(bet+6*alf)
    h =  gamma(bet-alf)/gamma(bet+7*alf)
    i = -gamma(bet-alf)/gamma(bet+8*alf)
    j =  gamma(bet-alf)/gamma(bet+9*alf)
    k = -gamma(bet-alf)/gamma(bet+10*alf)
    l =  gamma(bet-alf)/gamma(bet+11*alf)
    s = -gamma(bet-alf)/gamma(bet+12*alf)
    r =  -gamma(bet-alf)/gamma(bet+13*alf)
    #m = -gamma(bet-alf)/gamma(bet+12*alf)
    
     
    m =  gamma(bet-alf)/gamma(bet-2*alf)
    n =  -gamma(bet-alf)/gamma(bet-3*alf) # original
    o =  gamma(bet-alf)/gamma(bet-4*alf)
    p =  -gamma(bet-alf)/gamma(bet-5*alf)
    q =  gamma(bet-alf)/gamma(bet-6*alf)
    
    
    # n = -gamma(bet-alf)/gamma(bet-2*alf)
    # p = gamma(bet-alf)/gamma(bet-3*alf)
    
    row1  = [1, 0, 0, 0, 0, 0, 0, 0, 0, a,  0,  0,  0,  0, 0, 0,  0, 0, 0]
    row2  = [0, 1, 0, 0, 0, 0, 0, 0, 0, b,  a,  0,  0,  0, 0, 0,  0, 0, 0]
    row3  = [0, 0, 1, 0, 0, 0, 0, 0, 0, c,  b,  a,  0,  0, 0, 0,  0, 0, 0]
    row4  = [0, 0, 0, 1, 0, 0, 0, 0, 0, d,  c,  b,  a,  0, 0, 0,  0, 0, 0]
    row5  = [0, 0, 0, 0, 1, 0, 0, 0, 0, e,  d,  c,  b,  a, 0, 0,  0, 0, 0]
    row6  = [0, 0, 0, 0, 0, 1, 0, 0, 0, f,  e,  d,  c,  b, a, 0,  0, 0, 0]
    row7  = [0, 0, 0, 0, 0, 0, 1, 0, 0, g,  f,  e,  d,  c, b, a,  0, 0, 0]
    row8  = [0, 0, 0, 0, 0, 0, 0, 1, 0, h,  g,  f,  e,  d, c, b,  a, 0, 0]
    row9  = [0, 0, 0, 0, 0, 0, 0, 0, 1, i,  h,  g,  f,  e, d, c,  b, a, 0]
    row10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, j,  i,  h,  g,  f, e, d,  c, b, a]
    row11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, k,  j,  i,  h,  g, f, e,  d, c, b]
    row12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, l,  k,  j,  i,  h, g, f,  e, d, c]
    row13 = [0, 0, 0, 0, 0, 0, 0, 0, 0, s,  l,  k,  j,  i, h, g,  f, e, d]
    row14 = [0, 0, 0, 0, 0, 0, 0, 0, 0, r,  s,  l,  k,  j, i, h, g,  f, e]
    #row14 = [0, 0, 0, 1, 0, 0, 0, 0, 0, r,  0,  0,  0, -1, m, n,  o, p, q]
    
    
    row15 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,-1, m, n, o, p]
    row16 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0, 0,-1, m, n, o]
    row17 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0, 0, 0,-1, m, n]
    row18 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0, 0, 0, 0,-1, m]
    row19 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0, 0, 0, 0, 0,-1]
    
    mat = np.matrix((row1, row2, row3, row4, row5, row6, row7, row8, 
                    row9, row10, row11, row12, row13, row14, row15, row16, 
                    row17, row18, row19))
                    
    vect = np.matrix(([0], [0], [0], [0], [0], [0], [0], [0], [0], 
                      [-1], [-a], [-b], [-c], [-d], [-q], [-p], [-o], [-n], [-m]))
                      
    return solve(mat, vect)

def gpa10_15_6_ml(x, alf, bet):
    x = np.abs(x)
    coeff = gpa10_15_6_ml_coff(alf, bet)
    
    num = (coeff[0,0] + coeff[1,0]*x + coeff[2,0]*(x**2) + coeff[3,0]*(x**3) 
            + coeff[4,0]*(x**4) + coeff[5,0]*(x**5) + coeff[6,0]*(x**6) 
            + coeff[7,0]*(x**7)+ coeff[8,0]*(x**8) + (x**9))
    
    denum = (coeff[9,0] + coeff[10,0]*x + coeff[11,0]*(x**2) 
            + coeff[12,0]*(x**3) + coeff[13,0]*(x**4) + coeff[14,0]*(x**5) 
            + coeff[15,0]*(x**6) + coeff[16,0]*(x**7) + coeff[17,0]*(x**8) 
            + coeff[18,0]*(x**9) + (x**10))
    
    return num/(gamma(bet-alf)*denum)
    
    
def gpa10_15_6_ml_pf(alf, bet):
    coeff = gpa10_15_6_ml_coff(alf, bet)
    
    num =   [1, coeff[6,0], coeff[5,0], coeff[4,0], coeff[3,0], coeff[2,0], coeff[1,0], coeff[0,0]]
    denum = gamma(bet-alf)*np.array([1, coeff[14,0], coeff[13,0], coeff[12,0], coeff[11,0], coeff[10,0], coeff[9,0], coeff[8,0], coeff[7,0]])

    return residue(num, denum)
    

def gpa10_14_7_ml_coff(alf, bet):
    a = -gamma(bet-alf)/gamma(bet)
    b =  gamma(bet-alf)/gamma(bet+alf)
    c = -gamma(bet-alf)/gamma(bet+2*alf)
    d =  gamma(bet-alf)/gamma(bet+3*alf)
    e = -gamma(bet-alf)/gamma(bet+4*alf)
    f =  gamma(bet-alf)/gamma(bet+5*alf)
    g = -gamma(bet-alf)/gamma(bet+6*alf)
    h =  gamma(bet-alf)/gamma(bet+7*alf)
    i = -gamma(bet-alf)/gamma(bet+8*alf)
    j =  gamma(bet-alf)/gamma(bet+9*alf)
    k = -gamma(bet-alf)/gamma(bet+10*alf)
    l =  gamma(bet-alf)/gamma(bet+11*alf)
    s = -gamma(bet-alf)/gamma(bet+12*alf)
    #m = -gamma(bet-alf)/gamma(bet+12*alf)
    
     
    m =  gamma(bet-alf)/gamma(bet-2*alf)
    n =  -gamma(bet-alf)/gamma(bet-3*alf) # original
    o =  gamma(bet-alf)/gamma(bet-4*alf)
    p =  -gamma(bet-alf)/gamma(bet-5*alf)
    q =  gamma(bet-alf)/gamma(bet-6*alf)
    r =  -gamma(bet-alf)/gamma(bet-7*alf)
    
    # n = -gamma(bet-alf)/gamma(bet-2*alf)
    # p = gamma(bet-alf)/gamma(bet-3*alf)
    
    row1  = [1, 0, 0, 0, 0, 0, 0, 0, 0, a,  0,  0,  0,  0, 0, 0,  0, 0, 0]
    row2  = [0, 1, 0, 0, 0, 0, 0, 0, 0, b,  a,  0,  0,  0, 0, 0,  0, 0, 0]
    row3  = [0, 0, 1, 0, 0, 0, 0, 0, 0, c,  b,  a,  0,  0, 0, 0,  0, 0, 0]
    row4  = [0, 0, 0, 1, 0, 0, 0, 0, 0, d,  c,  b,  a,  0, 0, 0,  0, 0, 0]
    row5  = [0, 0, 0, 0, 1, 0, 0, 0, 0, e,  d,  c,  b,  a, 0, 0,  0, 0, 0]
    row6  = [0, 0, 0, 0, 0, 1, 0, 0, 0, f,  e,  d,  c,  b, a, 0,  0, 0, 0]
    row7  = [0, 0, 0, 0, 0, 0, 1, 0, 0, g,  f,  e,  d,  c, b, a,  0, 0, 0]
    row8  = [0, 0, 0, 0, 0, 0, 0, 1, 0, h,  g,  f,  e,  d, c, b,  a, 0, 0]
    row9  = [0, 0, 0, 0, 0, 0, 0, 0, 1, i,  h,  g,  f,  e, d, c,  b, a, 0]
    row10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, j,  i,  h,  g,  f, e, d,  c, b, a]
    row11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, k,  j,  i,  h,  g, f, e,  d, c, b]
    row12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, l,  k,  j,  i,  h, g, f,  e, d, c]
    row13 = [0, 0, 0, 0, 0, 0, 0, 0, 0, s,  l,  k,  j,  i, h, g,  f, e, d]
    
    
    row14 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0, -1, m, n, o, p, q]
    row15 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,-1, m, n, o, p]
    row16 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0, 0,-1, m, n, o]
    row17 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0, 0, 0,-1, m, n]
    row18 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0, 0, 0, 0,-1, m]
    row19 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0, 0, 0, 0, 0,-1]
    
    mat = np.matrix((row1, row2, row3, row4, row5, row6, row7, row8, 
                    row9, row10, row11, row12, row13, row14, row15, row16, 
                    row17, row18, row19))
                    
    vect = np.matrix(([0], [0], [0], [0], [0], [0], [0], [0], [0], 
                      [-1], [-a], [-b], [-c], [-r], [-q], [-p], [-o], [-n], [-m]))
                      
    return solve(mat, vect)

def gpa10_14_7_ml(x, alf, bet):
    x = np.abs(x)
    coeff = gpa10_14_7_ml_coff(alf, bet)
    
    num = (coeff[0,0] + coeff[1,0]*x + coeff[2,0]*(x**2) + coeff[3,0]*(x**3) 
            + coeff[4,0]*(x**4) + coeff[5,0]*(x**5) + coeff[6,0]*(x**6) 
            + coeff[7,0]*(x**7)+ coeff[8,0]*(x**8) + (x**9))
    
    denum = (coeff[9,0] + coeff[10,0]*x + coeff[11,0]*(x**2) 
            + coeff[12,0]*(x**3) + coeff[13,0]*(x**4) + coeff[14,0]*(x**5) 
            + coeff[15,0]*(x**6) + coeff[16,0]*(x**7) + coeff[17,0]*(x**8) 
            + coeff[18,0]*(x**9) + (x**10))
    
    return num/(gamma(bet-alf)*denum)
    
    
def gpa10_14_7_ml_pf(alf, bet):
    coeff = gpa10_13_8_ml_coff(alf, bet)
    
    num =   [1, coeff[6,0], coeff[5,0], coeff[4,0], coeff[3,0], coeff[2,0], coeff[1,0], coeff[0,0]]
    denum = gamma(bet-alf)*np.array([1, coeff[14,0], coeff[13,0], coeff[12,0], coeff[11,0], coeff[10,0], coeff[9,0], coeff[8,0], coeff[7,0]])

    return residue(num, denum)
    

def gpa10_13_8_ml_coff(alf, bet):
    a = -gamma(bet-alf)/gamma(bet)
    b =  gamma(bet-alf)/gamma(bet+alf)
    c = -gamma(bet-alf)/gamma(bet+2*alf)
    d =  gamma(bet-alf)/gamma(bet+3*alf)
    e = -gamma(bet-alf)/gamma(bet+4*alf)
    f =  gamma(bet-alf)/gamma(bet+5*alf)
    g = -gamma(bet-alf)/gamma(bet+6*alf)
    h =  gamma(bet-alf)/gamma(bet+7*alf)
    i = -gamma(bet-alf)/gamma(bet+8*alf)
    j =  gamma(bet-alf)/gamma(bet+9*alf)
    k = -gamma(bet-alf)/gamma(bet+10*alf)
    l =  gamma(bet-alf)/gamma(bet+11*alf) 
     
    m =  gamma(bet-alf)/gamma(bet-2*alf)
    n =  -gamma(bet-alf)/gamma(bet-3*alf)
    o =  gamma(bet-alf)/gamma(bet-4*alf)
    p =  -gamma(bet-alf)/gamma(bet-5*alf)
    q =  gamma(bet-alf)/gamma(bet-6*alf)
    r =  -gamma(bet-alf)/gamma(bet-7*alf)
    s =  gamma(bet-alf)/gamma(bet-8*alf)
    
    row1  = [1, 0, 0, 0, 0, 0, 0, 0, 0, a,  0,  0,  0,  0, 0, 0,  0, 0, 0]
    row2  = [0, 1, 0, 0, 0, 0, 0, 0, 0, b,  a,  0,  0,  0, 0, 0,  0, 0, 0]
    row3  = [0, 0, 1, 0, 0, 0, 0, 0, 0, c,  b,  a,  0,  0, 0, 0,  0, 0, 0]
    row4  = [0, 0, 0, 1, 0, 0, 0, 0, 0, d,  c,  b,  a,  0, 0, 0,  0, 0, 0]
    row5  = [0, 0, 0, 0, 1, 0, 0, 0, 0, e,  d,  c,  b,  a, 0, 0,  0, 0, 0]
    row6  = [0, 0, 0, 0, 0, 1, 0, 0, 0, f,  e,  d,  c,  b, a, 0,  0, 0, 0]
    row7  = [0, 0, 0, 0, 0, 0, 1, 0, 0, g,  f,  e,  d,  c, b, a,  0, 0, 0]
    row8  = [0, 0, 0, 0, 0, 0, 0, 1, 0, h,  g,  f,  e,  d, c, b,  a, 0, 0]
    row9  = [0, 0, 0, 0, 0, 0, 0, 0, 1, i,  h,  g,  f,  e, d, c,  b, a, 0]
    row10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, j,  i,  h,  g,  f, e, d,  c, b, a]
    row11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, k,  j,  i,  h,  g, f, e,  d, c, b]
    row12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, l,  k,  j,  i,  h, g, f,  e, d, c]
    
    row13 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0, -1,  m, n, o, p, q, r]
    row14 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0, -1, m, n, o, p, q]
    row15 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,-1, m, n, o, p]
    row16 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0, 0,-1, m, n, o]
    row17 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0, 0, 0,-1, m, n]
    row18 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0, 0, 0, 0,-1, m]
    row19 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0, 0, 0, 0, 0,-1]
    
    mat = np.matrix((row1, row2, row3, row4, row5, row6, row7, row8, 
                    row9, row10, row11, row12, row13, row14, row15, row16, 
                    row17, row18, row19))
                    
    vect = np.matrix(([0], [0], [0], [0], [0], [0], [0], [0], [0], 
                      [-1], [-a], [-b], [-s], [-r], [-q], [-p], [-o], [-n], [-m]))
                      
    return solve(mat, vect)

def gpa10_13_8_ml(x, alf, bet):
    x = np.abs(x)
    coeff = gpa10_13_8_ml_coff(alf, bet)
    
    num = (coeff[0,0] + coeff[1,0]*x + coeff[2,0]*(x**2) + coeff[3,0]*(x**3) 
            + coeff[4,0]*(x**4) + coeff[5,0]*(x**5) + coeff[6,0]*(x**6) 
            + coeff[7,0]*(x**7)+ coeff[8,0]*(x**8) + (x**9))
    
    denum = (coeff[9,0] + coeff[10,0]*x + coeff[11,0]*(x**2) 
            + coeff[12,0]*(x**3) + coeff[13,0]*(x**4) + coeff[14,0]*(x**5) 
            + coeff[15,0]*(x**6) + coeff[16,0]*(x**7) + coeff[17,0]*(x**8) 
            + coeff[18,0]*(x**9) + (x**10))
    
    return num/(gamma(bet-alf)*denum)
    
    
def gpa10_13_8_ml_pf(alf, bet):
    coeff = gpa10_13_8_ml_coff(alf, bet)
    
    num =   [1, coeff[6,0], coeff[5,0], coeff[4,0], coeff[3,0], coeff[2,0], coeff[1,0], coeff[0,0]]
    denum = gamma(bet-alf)*np.array([1, coeff[14,0], coeff[13,0], coeff[12,0], coeff[11,0], coeff[10,0], coeff[9,0], coeff[8,0], coeff[7,0]])

    return residue(num, denum)
    
# print(gpa10_14_7_ml(0, 0.5, 0.6))