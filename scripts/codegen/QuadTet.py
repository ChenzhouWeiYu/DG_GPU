import os
import re
import numpy as np


sss = ''
for filename in os.listdir('../../data/QuadData'):
    if(re.search(r'NME_6313_cubature_tetra_p\d+_n\d+',filename)):
        # print(filename)
        result = re.findall(r'\d+',filename)
        polynomial_order = int(result[1])
        points_num = int(result[2])
        with open('../../data/QuadData/'+filename,) as fp:
            lines = fp.readlines()
        s1 = """    struct Degree%dPoints%d {
        public:
            static constexpr int num_points = %d;
            __host__ __device__
            static constexpr std::array<std::array<Scalar,3>, num_points> get_points() {
                return {{
"""%(polynomial_order,points_num,points_num)
        for mm,line in enumerate(lines):
            x,y,z,w = line.split(' ')
            s1 += '                {\n                    %s, \n                    %s, \n                    %s\n                }, \n'%(x,y,z)
        s1 += """                    }};
            }

            __host__ __device__
            static constexpr std::array<Scalar, num_points> get_weights() {
                return {
"""
        for mm,line in enumerate(lines):
            x,y,z,w = line.split(' ')
            # if((mm)%3==0): s1 += '                '
            s1 += '                %s/6.0,\n'%(w[:-1])
            # if((mm)%3==2): s1 += '\n'
        s1 += """            };
            }
    };
"""
        # print(s1)
        sss += s1

with open('./QuadTetHighOrder.h','w') as f:
    print(sss,file=f)