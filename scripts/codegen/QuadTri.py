import re


sss = ''
with open('../../data/QuadData/rotationalsymmetry.txt') as fp:
    lines = fp.readlines()
    k = 0
    while(k<len(lines)):
        line = lines[k]
        k = k+1
        if(line[0]=='#' or line[0]=='\n'): continue
        # print(line.find('degree:'))
        l1 = re.findall(r'degree: +\d+',line)
        l2 = re.findall(r'points: +\d+',line)
        l3 = re.findall(r'orbits: +\[\d+, *\d+\]',line)
        l4 = re.findall(r'quality: PI',line)
        if(len(l1+l2+l3+l4)==4):
            polynomial_order = int(re.findall(r'\d+',l1[0])[0])
            points_num = int(re.findall(r'\d+',l2[0])[0])
            orbits = [int(s) for s in re.findall(r'\d+',l3[0])]
            # print(polynomial_order,points_num,orbits)
            points = []
            weight = []
            if(orbits[0] == 1):
                # print(lines[k])
                line = lines[k].split('  ')
                k = k+1
                weight.append(float(line[0]))
                points.append((float(line[1]),float(line[2]),float(line[3])))
                # print(points,weight)
            for _ in range(orbits[1]):
                line = lines[k].split('  ')
                k = k+1
                weight.append(float(line[0]))
                points.append((float(line[1]),float(line[2]),float(line[3])))
                weight.append(float(line[0]))
                points.append((float(line[2]),float(line[3]),float(line[1])))
                weight.append(float(line[0]))
                points.append((float(line[3]),float(line[1]),float(line[2])))
            # print(len(points),len(weight))
            # print(points)
            # print(weight)
            s1 = """    struct Degree%dPoints%d {
        public:
            static constexpr int num_points = %d;
            __host__ __device__
            static constexpr std::array<std::array<Scalar, 2>, num_points> get_points() {
                return {{
"""%(polynomial_order,points_num,points_num)
            for mm,(x,y,z) in enumerate(points):
                if((mm-orbits[0])%3==0 or (orbits[0] and not mm)): s1 += '                '
                s1 += '{%.18f, %.18f},  '%(y,z)
                if((mm-orbits[0])%3==2): s1 += '\n'
            s1 += """                }};
            }

            __host__ __device__
            static constexpr std::array<Scalar, num_points> get_weights() {
                return { 
"""
            for mm,w in enumerate(weight):
                if((mm-orbits[0])%3==0 or (orbits[0] and not mm)): s1 += '                '
                s1 += '0.5*%.18f,  '%(w)
                if((mm-orbits[0])%3==2): s1 += '\n'
            s1 += """            };
            }
    };
"""
            sss += s1
            
            


with open('../../data/QuadData/fullsymmetry.txt') as fp:
    lines = fp.readlines()
    k = 0
    while(k<len(lines)):
        line = lines[k]
        k = k+1
        if(line[0]=='#' or line[0]=='\n'): continue
        # print(line.find('degree:'))
        l1 = re.findall(r'degree: +\d+',line)
        l2 = re.findall(r'points: +\d+',line)
        l3 = re.findall(r'orbits: +\[\d+, *\d+, *\d+\]',line)
        l4 = re.findall(r'quality: PI',line)
        if(len(l1+l2+l3+l4)==4):
            polynomial_order = int(re.findall(r'\d+',l1[0])[0])
            points_num = int(re.findall(r'\d+',l2[0])[0])
            orbits = [int(s) for s in re.findall(r'\d+',l3[0])]
            # print(polynomial_order,points_num,orbits)
            points = []
            weight = []
            if(orbits[0] == 1):
                # print(lines[k])
                line = lines[k].split('  ')
                k = k+1
                weight.append(float(line[0]))
                points.append((float(line[1]),float(line[2]),float(line[3])))
                # print(points,weight)
            for _ in range(orbits[1]):
                line = lines[k].split('  ')
                k = k+1
                weight.append(float(line[0]))
                points.append((float(line[1]),float(line[2]),float(line[3])))
                weight.append(float(line[0]))
                points.append((float(line[2]),float(line[3]),float(line[1])))
                weight.append(float(line[0]))
                points.append((float(line[3]),float(line[1]),float(line[2])))
            for _ in range(orbits[2]):
                line = lines[k].split('  ')
                k = k+1
                weight.append(float(line[0]))
                points.append((float(line[1]),float(line[2]),float(line[3])))
                weight.append(float(line[0]))
                points.append((float(line[2]),float(line[3]),float(line[1])))
                weight.append(float(line[0]))
                points.append((float(line[3]),float(line[1]),float(line[2])))
                weight.append(float(line[0]))
                points.append((float(line[1]),float(line[3]),float(line[2])))
                weight.append(float(line[0]))
                points.append((float(line[2]),float(line[1]),float(line[3])))
                weight.append(float(line[0]))
                points.append((float(line[3]),float(line[2]),float(line[1])))
            # print(len(points),len(weight))
            # print(points)
            # print(weight)
            s1 = """    struct Degree%dPoints%d {
        public:
            static constexpr int num_points = %d;
            __host__ __device__
            static constexpr std::array<std::array<Scalar, 2>, num_points> get_points() {
                return {{
"""%(polynomial_order,points_num,points_num)
            for mm,(x,y,z) in enumerate(points):
                if((mm-orbits[0])%3==0 or (orbits[0] and not mm)): s1 += '                '
                s1 += '{%.18f, %.18f},  '%(y,z)
                if((mm-orbits[0])%3==2): s1 += '\n'
            s1 += """                }};
            }

            __host__ __device__
            static constexpr std::array<Scalar, num_points> get_weights() {
                return { 
"""
            for mm,w in enumerate(weight):
                if((mm-orbits[0])%3==0 or (orbits[0] and not mm)): s1 += '                '
                s1 += '0.5*%.18f,  '%(w)
                if((mm-orbits[0])%3==2): s1 += '\n'
            s1 += """            };
            }
    };
"""
            sss += s1



with open('./QuadTriHighOrder.h','w') as f:
    print(sss,file=f)
            
            