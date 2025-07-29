import re
import numpy as np
from sympy import sympify, diff, symbols

ssss = """{{{0,0,0,0,0},{-(4/3) u[x,y,z],4/3,0,0,0},{-v[x,y,z],0,1,0,0},{-w[x,y,z],0,0,1,0},{-Const e[x,y,z]+(-(4/3)+Const) u[x,y,z]^2+(-1+Const) (v[x,y,z]^2+w[x,y,z]^2),-(1/3) (-4+3 Const) u[x,y,z],-(-1+Const) v[x,y,z],-(-1+Const) w[x,y,z],Const}},{{0,0,0,0,0},{-v[x,y,z],0,1,0,0},{2/3 u[x,y,z],-(2/3),0,0,0},{0,0,0,0,0},{-(1/3) u[x,y,z] v[x,y,z],-(2/3) v[x,y,z],u[x,y,z],0,0}},{{0,0,0,0,0},{-w[x,y,z],0,0,1,0},{0,0,0,0,0},{2/3 u[x,y,z],-(2/3),0,0,0},{-(1/3) u[x,y,z] w[x,y,z],-(2/3) w[x,y,z],0,u[x,y,z],0}},{{0,0,0,0,0},{2/3 v[x,y,z],0,-(2/3),0,0},{-u[x,y,z],1,0,0,0},{0,0,0,0,0},{-(1/3) u[x,y,z] v[x,y,z],v[x,y,z],-(2/3) u[x,y,z],0,0}},{{0,0,0,0,0},{-u[x,y,z],1,0,0,0},{-(4/3) v[x,y,z],0,4/3,0,0},{-w[x,y,z],0,0,1,0},{-Const e[x,y,z]+(-1+Const) u[x,y,z]^2-4/3 v[x,y,z]^2+Const v[x,y,z]^2-w[x,y,z]^2+Const w[x,y,z]^2,-(-1+Const) u[x,y,z],-(1/3) (-4+3 Const) v[x,y,z],-(-1+Const) w[x,y,z],Const}},{{0,0,0,0,0},{0,0,0,0,0},{-w[x,y,z],0,0,1,0},{2/3 v[x,y,z],0,-(2/3),0,0},{-(1/3) v[x,y,z] w[x,y,z],0,-(2/3) w[x,y,z],v[x,y,z],0}},{{0,0,0,0,0},{2/3 w[x,y,z],0,0,-(2/3),0},{0,0,0,0,0},{-u[x,y,z],1,0,0,0},{-(1/3) u[x,y,z] w[x,y,z],w[x,y,z],0,-(2/3) u[x,y,z],0}},{{0,0,0,0,0},{0,0,0,0,0},{2/3 w[x,y,z],0,0,-(2/3),0},{-v[x,y,z],0,1,0,0},{-(1/3) v[x,y,z] w[x,y,z],0,w[x,y,z],-(2/3) v[x,y,z],0}},{{0,0,0,0,0},{-u[x,y,z],1,0,0,0},{-v[x,y,z],0,1,0,0},{-(4/3) w[x,y,z],0,0,4/3,0},{-Const e[x,y,z]+(-1+Const) u[x,y,z]^2-v[x,y,z]^2+Const v[x,y,z]^2-4/3 w[x,y,z]^2+Const w[x,y,z]^2,-(-1+Const) u[x,y,z],-(-1+Const) v[x,y,z],-(1/3) (-4+3 Const) w[x,y,z],Const}}}"""

ssss = ssss.replace('[x,y,z]','').replace('{','[').replace('}',']')

ssss,re.sub(r'([tuvwe)\d])\s(?=[Cuvwe(])', r'\1*', ssss)
sym_code = sympify(re.sub(r'([tuvwe)\d])\s(?=[Cuvwe(])', r'\1*', ssss))

def cpp_code(expr):
    from sympy.printing.cxx import cxxcode
    return cxxcode(expr, standard='C++17', 
                  user_functions={'Pow': 'std::pow'})


len(sym_code)
rr = 'return {'
for i in range(3):
    rr += '{'
    for j in range(3):
        rr += 'G_f%c_u%c'%('xyz'[i],'xyz'[j])
        print('const auto& ' + rr[-7:] + ' = mu/rho * DenseMatrix<5,5>(' + cpp_code(sum(sym_code[3*j+i],start=[])) + ');')
        rr += ', '
    rr = rr[:-2] + '},  '
print(rr[:-3]+'};')
