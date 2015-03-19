from numpy import *
from scipy import *
from operator import itemgetter
from random import sample
from ast import *
import matplotlib.pyplot as plt
from copy import deepcopy


operator_dict = {-1: 'sqrt', 0: 'noise', 1: '+', 2: '-', 3: '*', 4: '/', 5: 'sin', 6: 'cos', 7: 'exp', 8: 'x'}

evaluations = 0
flipper=[]
for i in range(50):
    if i%2==1:
        flipper.extend([1])
    else:
        flipper.extend([-1])
flipper=array(flipper)

class person(object):
    def __init__(self, root = None, left = None, right = None):
        self.root = root
        self.left = left
        self.right = right
    
    def evaluate(self,dic):
        try:
            if self.root == operator_dict[-1]:
                return self.left.evaluate(dic)*(abs(sqrt(self.right.evaluate(dic)))*flipper)

            elif self.root == operator_dict[0]:
                r = []
                for i in range(len(array(self.right.evaluate(dic)))):
                    y = list(random.normal(1,0.05,1))
                    r.extend(y)
                return self.left.evaluate(dic) * (r*array(self.right.evaluate(dic)))
            elif self.root == operator_dict[1]:
                return self.left.evaluate(dic) + self.right.evaluate(dic)
            elif self.root == operator_dict[2]:
                return self.left.evaluate(dic) - self.right.evaluate(dic)
            elif self.root == operator_dict[3]:
                return self.left.evaluate(dic) * self.right.evaluate(dic)
            elif self.root == operator_dict[4]:
                return self.left.evaluate(dic) / self.right.evaluate(dic)
            elif self.root == operator_dict[5]:
                return self.left.evaluate(dic) * sin(self.right.evaluate(dic))
            elif self.root == operator_dict[6]:
                return self.left.evaluate(dic) * cos(self.right.evaluate(dic))
            elif self.root == operator_dict[7]:
                return self.left.evaluate(dic) * exp(self.right.evaluate(dic))
            elif self.root == operator_dict[8]:
                return array(map(lambda dic: dic[0], dic))
            else:
                return self.root

        except:
            return nan

    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1

    def find_branch(self,depth):
        level = 0
        if level==depth:
            dice = random.randint(1,3)
            if dice == 1:
                return self.right
            else:
                return self.left
        else:
            level+=1
            dice = random.randint(1,3)
            if dice == 1:
                try:
                    return self.left.find_branch(depth)
                except:
                    pass
                try:
                    return self.right.find_branch(depth)
                except:
                    pass
                return person(self.root)
            if dice == 2:
                try:
                    return self.right.find_branch(depth)
                except:
                    pass
                try:
                    return self.left.find_branch(depth)
                except:
                    pass
                return person(self.root)

    def mother_chromosome(self,depth):
        chromosome = ['child']
        level = 0
        if level==depth:
            dice = random.randint(1,3)
            if dice == 1:
                chromosome.extend(['.right'])
                return ''.join(chromosome)
            else:
                chromosome.extend(['.left'])
                return ''.join(chromosome)
        else:
            level+=1
            dice = random.randint(1,3)
            if dice == 1:
                try:
                    chromosome.extend(['.left'])
                    return self.left.mother_chromosome(depth)
                except:
                    pass
                try:
                    chromosome.extend(['.right'])
                    return self.right.mother_chromosome(depth)
                except:
                    pass
                return ''.join(chromosome)
            if dice == 2:
                try:
                    chromosome.extend(['.right'])
                    return self.right.mother_chromosome(depth)
                except:
                    pass
                try:
                    chromosome.extend(['.left'])
                    return self.left.mother_chromosome(depth)
                except:
                    pass
                return ''.join(chromosome)

    def fitness(self,dic):
        evals = evaluations
        evals = evals + 1
        global evaluations
        evaluations = evals
        try:
            genotype = self.evaluate(dic)
            if type(genotype)==float64:
                return 10000000.
            elif type(genotype)==ndarray and math.isnan(genotype[0]):
                return 10000000.
            elif type(genotype)==ndarray and len(genotype)<5:
                return 10000000.
            elif type(genotype)==float:
                return 10000000.
            else:
                value = sum((self.evaluate(dic)-\
                    array(map(lambda dic: dic[1],dic)))**2)
                return value
        except:
            print "Couldn't evaluate fitness"
            return 1000000000.

    def crossover(self,spouse):
        max_depth = min(self.depth(),spouse.depth())
        depth = random.randint(1,max_depth)
        child = deepcopy(self)
        father = '=spouse.find_branch(depth)'
        mother = child.mother_chromosome(depth)
        DNA = mother+father
        exec(DNA)
        return child

    def mutate(self,operator_mutation_rate,constant_mutation_rate,variable_intro_rate,variable_outro_rate,extension_rate,max_constant,maximum_depth):
        if type(self.root)==str and self.root!=operator_dict[8]:
            dice = random.random()
            if dice < operator_mutation_rate:
                op = random.randint(-1,8)
                self.root = operator_dict[op]
                self.left.mutate(operator_mutation_rate,\
                    constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
                    extension_rate,max_constant,maximum_depth)
                self.right.mutate(operator_mutation_rate,\
                    constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
                    extension_rate,max_constant,maximum_depth)
            else:
                self.left.mutate(operator_mutation_rate,\
                    constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
                    extension_rate,max_constant,maximum_depth)
                self.right.mutate(operator_mutation_rate,\
                    constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
                    extension_rate,max_constant,maximum_depth)
        elif type(self.root)==float:
            dice = random.random()
            dice1 = random.random()
            dice2 = random.random()
            if dice < constant_mutation_rate:
                const = random.uniform(self.root-0.5*self.root,self.root+0.5*self.root)
                self.root=const
            if dice1 < variable_intro_rate:
                self.root=operator_dict[8]
            if dice2 < extension_rate and self.depth()<maximum_depth:
                op = random.randint(-1,8)
                self.root = operator_dict[op]
                self.right = person(random.uniform(0,max_constant))
                self.left = person(random.uniform(0,max_constant))
        elif self.root==operator_dict[8]:
            dice = random.random()
            if dice < variable_outro_rate:
                self.root = random.uniform(0,max_constant)
            else:
                pass


class population(object):
    def __init__(self, people=None):
        self.people=[people]

    def add_person(self,person):
        self.people.extend([person])

    def rank(self,dic):
        caliber = []
        for i in self.people:
            caliber.extend([i.fitness(dic)])
        rating = sorted(zip(caliber,self.people),key=itemgetter(0))
        self.people = map(lambda rating: rating[1], rating)

    def select(self,elite_pressure,total_pressure):
        parent_population=[]
        elite = int(elite_pressure*len(self.people))
        parent_population.extend(self.people[0:elite])
        remaining = int(total_pressure*len(self.people))-elite
        for i in range(elite,elite+remaining):
            parent = random.randint(elite,len(self.people))
            parent_population.extend([self.people[parent]])
        self.people = parent_population

    def breed(self,population_size,operator_mutation_rate,constant_mutation_rate,variable_intro_rate,variable_outro_rate,extension_rate,max_constant,maximum_depth):
        parents = len(self.people)
        num_children = population_size - parents
        for i in range(num_children):
            mother = self.people[random.randint(0,parents)]
            father = self.people[random.randint(0,parents)]
            child = mother.crossover(father)
            child.mutate(operator_mutation_rate,constant_mutation_rate,\
                    variable_intro_rate,variable_outro_rate,extension_rate,max_constant,\
                    maximum_depth)
            self.add_person(child)

    def plotter(self,dic):
        x1=[]
        y1=[]
        x2=[]
        y2=[]
        for i in range(len(dic)):
            x1.extend([dic[i][0]])
            x2.extend([dic[i][0]])
            y1.extend([dic[i][1]])
        y2=list(self.people[0].evaluate(dic))
        plt.plot(x2,y2,'r*',label="Genetic Algorithm")
        plt.plot(x1,y1,'b*',label="Data")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Genetic Algorithm')
        plt.legend(loc='upper left',prop={'size':8})
        plt.show()

    def evolve(self,dic,generations,elite_pressure,total_pressure,population_size,operator_mutation_rate,constant_mutation_rate,variable_intro_rate,variable_outro_rate,extension_rate,max_constant,maximum_depth):
        for i in range(generations):
            print 'Generation Number: '+str(i)
            self.rank(dic)
            # if i%50==0:
            #     self.plotter(dic)
            print self.people[0].fitness(dic)
            print evaluations
            print '\n'
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,operator_mutation_rate,\
                constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

    def synth(self,dic,generations,elite_pressure,total_pressure,population_size,operator_mutation_rate,constant_mutation_rate,variable_intro_rate,variable_outro_rate,extension_rate,max_constant,maximum_depth):
        fit1=[]
        fit2=[]
        fit3=[]

        fit4=[]
        fit5=[]
        fit6=[]

        fit7=[]
        fit8=[]
        fit9=[]

        evals1=[]
        evals2=[]
        evals3=[]

        evals4=[]
        evals5=[]
        evals6=[]

        evals7=[]
        evals8=[]
        evals9=[]

        global evaluations
        evaluations = 0
        print 'first loop'
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit1.extend([self.people[0].fitness(dic)])
                evals1.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.5,\
                constant_mutation_rate,0.5,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'third loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit2.extend([self.people[0].fitness(dic)])
                evals2.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.5,\
                constant_mutation_rate,0.5,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'fourth loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit3.extend([self.people[0].fitness(dic)])
                evals3.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.8,\
                constant_mutation_rate,0.5,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'fifth loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit4.extend([self.people[0].fitness(dic)])
                evals4.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.8,\
                constant_mutation_rate,0.5,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'sixth loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit5.extend([self.people[0].fitness(dic)])
                evals5.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.8,\
                constant_mutation_rate,0.5,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'seventh loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit6.extend([self.people[0].fitness(dic)])
                evals6.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,operator_mutation_rate,\
                constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'eighth loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit7.extend([self.people[0].fitness(dic)])
                evals7.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.5,\
                constant_mutation_rate,0.2,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'ninth loop'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit8.extend([self.people[0].fitness(dic)])
                evals8.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.5,\
                constant_mutation_rate,0.2,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)

        self.people=[]
        for i in range(population_size+1):
            citizen = person()
            citizen.root = operator_dict[random.randint(0,8)]
            decider = random.random()
            if decider<0.5:
                citizen.left = person(random.uniform(-max_constant,max_constant))
                citizen.right = person(operator_dict[8])
            if decider>0.5:
                citizen.right = person(random.uniform(-max_constant,max_constant))
                citizen.left = person(operator_dict[8])
            self.add_person(citizen)
        self.people=self.people[1:]

        print 'tenth loop (actually ninth)'
        global evaluations
        evaluations = 0
        for i in range(generations):
            self.rank(dic)
            if i%5==0:
                fit9.extend([self.people[0].fitness(dic)])
                evals9.extend([evaluations])
            self.select(elite_pressure,total_pressure)
            self.breed(population_size,0.5,\
                constant_mutation_rate,0.2,variable_outro_rate,\
                extension_rate,max_constant,maximum_depth)


        g=sorted(zip(fit1,evals1)+zip(fit2,evals2)+zip(fit3,evals3),key=itemgetter(1))
        h=sorted(zip(fit4,evals4)+zip(fit5,evals5)+zip(fit6,evals6),key=itemgetter(1))
        r=sorted(zip(fit7,evals7)+zip(fit8,evals8)+zip(fit9,evals9),key=itemgetter(1))

        g_mean=[]
        g_mean_eval=[]
        g_stdv=[]
        h_mean=[]
        h_mean_eval=[]
        h_stdv=[]
        r_mean=[]
        r_mean_eval=[]
        r_stdv=[]


        for i in arange(0,len(g)-8,8):
            a=(g[i][0]+g[i+1][0]+g[i+2][0]+g[i+3][0]+g[i+4][0]+g[i+5][0]+g[i+6][0]+g[i+7][0])/8.
            b=(g[i][1]+g[i+1][1]+g[i+2][1]+g[i+3][1]+g[i+4][1]+g[i+5][1]+g[i+6][1]+g[i+7][1])/8.
            g_stdv.extend([std([g[i][0],g[i+1][0],g[i+2][0],g[i+3][0],g[i+4][0],g[i+5][0],g[i+6][0],g[i+7][0]])])
            g_error_bar = array(g_stdv)/sqrt(3)
            g_mean.extend([a])
            g_mean_eval.extend([b])

        for i in arange(0,len(h)-8,8):
            a=(h[i][0]+h[i+1][0]+h[i+2][0]+h[i+3][0]+h[i+4][0]+h[i+5][0]+h[i+6][0]+h[i+7][0])/8.
            b=(h[i][1]+h[i+1][1]+h[i+2][1]+h[i+3][1]+h[i+4][1]+h[i+5][1]+h[i+6][1]+h[i+7][1])/8.
            h_stdv.extend([std([h[i][0],h[i+1][0],h[i+2][0],h[i+3][0],h[i+4][0],h[i+5][0],h[i+6][0],h[i+7][0]])])
            h_error_bar = array(h_stdv)/sqrt(3)
            h_mean.extend([a])
            h_mean_eval.extend([b])

        for i in arange(0,len(r)-8,8):
            a=(r[i][0]+r[i+1][0]+r[i+2][0]+r[i+3][0]+r[i+4][0]+r[i+5][0]+r[i+6][0]+r[i+7][0])/8.
            b=(r[i][1]+r[i+1][1]+r[i+2][1]+r[i+3][1]+r[i+4][1]+r[i+5][1]+r[i+6][1]+r[i+7][1])/8.
            r_stdv.extend([std([r[i][0],r[i+1][0],r[i+2][0],r[i+3][0],r[i+4][0],r[i+5][0],r[i+6][0],r[i+7][0]])])
            r_error_bar = array(r_stdv)/sqrt(3)
            r_mean.extend([a])
            r_mean_eval.extend([b])

        plt.errorbar(g_mean_eval,g_mean,g_error_bar/2.,fmt='',label="Genetic Algorithm (max depth = 100, operator mutation rate = 0.5, elitism = 0.3, variable intro rate = 0.5, variable outro rate = 0.4)")
        plt.errorbar(h_mean_eval,h_mean,h_error_bar/2.,fmt='',label="Genetic Algorithm (max depth = 100, operator mutation rate = 0.8, elitism = 0.3, variable intro rate = 0.5, variable outro rate = 0.4)")
        plt.errorbar(r_mean_eval,r_mean,r_error_bar/2.,fmt='',label="Genetic Algorithm (max depth = 100, operator mutation rate = 0.5, elitism = 0.3, variable intro rate = 0.2, variable outro rate = 0.4)")

        plt.legend(loc='upper left',prop={'size':8})
        plt.ylim([0,6])
        plt.xlabel('evaluations')
        plt.ylabel('fitness (mean square error)')
        plt.title('Genetic Fitting Algorithms, Line Without Noise')
        plt.show()
