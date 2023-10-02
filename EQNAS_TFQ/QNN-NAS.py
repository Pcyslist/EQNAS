#图片读取处理所需库
from PIL import ImageGrab
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt

#量子线路搭建以及训练所需库
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
import math
import random



fitness_best = []


#种群各类参数

N=15                

Genome=64             

generation_max=20

list_empty=[]

best_arch_para = [list(list_empty) for i in range(16)] #保存每代中每个个体训练后的参数

                     




av_fitness = []


popSize=N+1

genomeLength=Genome+1

top_bottom=3

best_acc = np.empty([generation_max])
best_arch = np.empty([generation_max, genomeLength])

QuBitZero = np.array([[1],[0]])

QuBitOne = np.array([[0],[1]])

AlphaBeta = np.empty([top_bottom])

fitness = np.empty([popSize])

probability = np.empty([popSize])

# qpv: quantum chromosome (or population vector, QPV)

qpv = np.empty([popSize, genomeLength, top_bottom])         

nqpv = np.empty([popSize, genomeLength, top_bottom])

# chromosome: classical chromosome

chromosome = np.empty([popSize, genomeLength],dtype=np.int) 

child1 = np.empty([popSize, genomeLength, top_bottom])

child2 = np.empty([popSize, genomeLength, top_bottom])

best_chrom = np.empty([generation_max])



#初始化全局变量

theta=0

iteration=0

the_best_chrom=0

generation=0


# visualization tools
#%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
#读取文件夹中所有图像并转灰度
images = []
for f in glob.iglob("/home/lyy/327/zpx/copy1/Burke/*"):
    images.append(np.asarray(Image.open(f).convert('L')))
for f in glob.iglob("/home/lyy/327/zpx/copy1/Nimitz/*"):
    images.append(np.asarray(Image.open(f).convert('L')))
#将图像进行归一化
images = np.array(images)
images = images/255
images = images[..., np.newaxis]#多加一维，与tfq保持一致
#print(images)
#print(len(images))
#p1 = plt.imshow(images[0,:,:,0],cmap='gray')

images_test = []
for f in glob.iglob("/home/lyy/327/zpx/copy1/test_burke/*"):
    images_test.append(np.asarray(Image.open(f).convert('L')))
for f in glob.iglob("/home/lyy/327/zpx/copy1/test_nimitz/*"):
    images_test.append(np.asarray(Image.open(f).convert('L')))
images_test = np.array(images_test)
images_test = images_test/255
images_test = images_test[..., np.newaxis]#多加一维，与tfq保持一致
#print(images_test)
print(len(images_test))
#plt.imshow(images_test[0, :, :,0],cmap='gray')
#plt.colorbar()
#p2 = plt.imshow(images_test[0],cmap='gray')

#生成标签，并打乱训练集,1代表burke，0代表Nimitz
train_label = np.array([])
test_label = np.array([])
for i in range(202):
    train_label=np.append(train_label,1)
for i in range(209):
    train_label=np.append(train_label,0)
for i in range(40):
    test_label=np.append(test_label,1)
for i in range(40):
    test_label=np.append(test_label,0)

#进行打乱处理   
state1=np.random.get_state()
np.random.shuffle(images)

np.random.set_state(state1)
np.random.shuffle(train_label)

state2=np.random.get_state()
np.random.shuffle(images_test)

np.random.set_state(state2)
np.random.shuffle(test_label)

def filter_10(x, y):
    keep = (y == 1) | (y == 0)
    x, y = x[keep], y[keep]
    y = y == 1
    return x,y
images, train_label = filter_10(images, train_label)
images_test, test_label = filter_10(images_test, test_label)

#print(train_label)
print(len(train_label))
#print(test_label)
print(len(test_label))

#缩小至4x4
x_train_small = tf.image.resize(images, (4,4)).numpy()
x_test_small = tf.image.resize(images_test, (4,4)).numpy()
print(len(x_train_small))

#plt.imshow(x_train_small[4,:,:,0],cmap='gray')
#plt.colorbar()


#删除矛盾例子
def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for x,y in zip(xs, ys):
        labels = mapping[tuple(x.flatten())]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(list(labels)[0])
        else:
          #去除掉同一个图像属于多个类别的情况.
            pass

    num_burke = sum(1 for value in mapping.values() if True in value)
    num_nimitz = sum(1 for value in mapping.values() if False in value)
    num_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of burke: ", num_burke)
    print("Number of nimitz: ", num_nimitz)
    print("Number of contradictory images: ", num_both)
    print()
    print("Initial number of examples: ", len(xs))
    print("Remaining non-contradictory examples: ", len(new_x))

    return np.array(new_x), np.array(new_y)


y_train = train_label
y_test = test_label
x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
#print(x_train_nocon[0])
print("----------------------")
#编码为量子线路
#二值化阈值设置
THRESHOLD = 0.5

x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)
print(len(x_train_bin))
print("=================================")
#_ = remove_contradicting(x_train_bin, y_train_nocon)
print("++++++++++++++++++++++++++")
#将矛盾例子删除
x_train_bin, y_train_nocon = remove_contradicting(x_train_bin, y_train_nocon)
x_test_bin, y_test = remove_contradicting(x_test_bin,y_test)
print("train len:",len(x_train_bin))
#print(len(y_train_nocon))
print("test len:",len(x_test_bin))

#编码经典图像在量子线路中
def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]
#print(x_train_bin[35])
#print("label:",y_train_nocon[35],"len:",len(y_train_nocon))
#x_train_circ = convert_to_circuit(x_train_bin[0])
#print(len(x_train_circ))
#print(x_train_circ[0])
#SVGCircuit(circuit)


#bin_img = x_train_bin[35,:,:,0]
#print(bin_img)
#indices = np.array(np.where(bin_img)).T
#print(indices)

#将这些Cirq电路转换为张量tfq：
x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


#量子神经网络线路
#声明线路层的class
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)
    
    def add_layer_single(self, circuit, gate, prefix, i):
        symbol = sympy.Symbol(prefix + '-' + str(i))
        if i <=16:
            if gate==cirq.I:
                circuit.append(gate(self.data_qubits[i-1])**symbol)
            else:
                circuit.append(gate(self.data_qubits[i-1],self.readout)**symbol)
        elif i>17:
            if gate==cirq.I:
                circuit.append(gate(self.data_qubits[i-17])**symbol)
            else:
                circuit.append(gate(self.data_qubits[i-17],self.readout)**symbol)
            
#调用class 构建对象 !!!
demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(4,1),
                                   readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
#XX是两个X门的张量积
demo_builder.add_layer(circuit, gate = cirq.XX, prefix='xx')
#SVGCircuit(demo_builder)


def create_quantum_model(chromosome_i):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  #4x4的量子线路.
    readout = cirq.GridQubit(-1, -1)         #单独的量子比特作为读出
    circuit = cirq.Circuit()

    # 准备readout读出量子比特
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    #向量子线路添加逻辑门.
    #builder.add_layer(circuit, cirq.XX, "xx1")
    #builder.add_layer(circuit, cirq.YY, "yy1")
    #builder.add_layer(circuit, cirq.ZZ, "zz1")
    for j in range(1,genomeLength,2):
        if chromosome_i[j]==0:
            if chromosome_i[j+1]==0:
                builder.add_layer_single(circuit, cirq.XX, "xx",j//2)
            else:
                builder.add_layer_single(circuit, cirq.ZZ, "zz",j//2)
        elif chromosome_i[j]==1:
            if chromosome_i[j+1]==0:
                builder.add_layer_single(circuit, cirq.YY, "yy",j//2)
            else:
                builder.add_layer_single(circuit, cirq.I, "i",j//2)
            

    #准备一个读出比特
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)



def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


#qnn训练

def train_qnn(chromosome_i,i,EPOCHS):
    model_circuit, model_readout = create_quantum_model(chromosome_i)
    #model_circuit.ParamResolver({xx1-0:0.1})
    #print(model_readout)
    SVGCircuit(model_circuit)


    #将线路加载至keras模型中
    #创建keras模型
    #init = tf.keras.initializers.RandomNormal(mean=5, stddev=3, seed=42)
    PQC1 = tfq.layers.PQC(model_circuit, model_readout)#,initializer=init)
    print("---------------sym value ",i,"---------------")
    print(PQC1.symbol_values())
    print("---------------------------------------")
    model = tf.keras.Sequential([
        #将Input输入作为tf.string类型加入到keras模型中
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        PQC1,
    ])

    #调整hinge loss
    y_train_hinge = 2.0*y_train_nocon-1.0
    y_test_hinge = 2.0*y_test-1.0



    #编译keras模型
    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(lr=0.005,epsilon=1e-08),
        #optimizer=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=0.0004,nesterov=False),
        metrics=[hinge_accuracy])
    print(model.summary())
    #print(model.get_weights())#打印模型初始化的参数
    #print(model.get_layer())


    #train
    #EPOCHS = 15
    BATCH_SIZE = 32


    NUM_EXAMPLES = len(x_train_tfcirc)

    x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
    y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

    qnn_history = model.fit(
          x_train_tfcirc_sub, y_train_hinge_sub,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test_tfcirc, y_test_hinge))

    qnn_results = model.evaluate(x_test_tfcirc, y_test)
    best_arch_para[i].append(PQC1.symbol_values())
    return qnn_results


def Init_population():

    # Hadamard gate

    r2=math.sqrt(2.0)

    h=np.array([[1/r2,1/r2],[1/r2,-1/r2]])

    # Rotation Q-gate

    theta=0;

    rot =np.empty([2,2])

    # Initial population array (individual x chromosome)

    i=1; j=1;

    for i in range(1,popSize):

        for j in range(1,genomeLength):

            theta=np.random.uniform(0,1)*90

            theta=math.radians(theta)

            rot[0,0]=math.cos(theta); rot[0,1]=-math.sin(theta);

            rot[1,0]=math.sin(theta); rot[1,1]=math.cos(theta);

            AlphaBeta[0]=rot[0,0]*(h[0][0]*QuBitZero[0])+rot[0,1]*(h[0][1]*QuBitZero[1])

            AlphaBeta[1]=rot[1,0]*(h[1][0]*QuBitZero[0])+rot[1,1]*(h[1][1]*QuBitZero[1])

        # alpha squared          

            qpv[i,j,0]=np.around(2*pow(AlphaBeta[0],2),2) 

        # beta squared

            qpv[i,j,1]=np.around(2*pow(AlphaBeta[1],2),2) 





def Show_population():

    i=1; j=1;

    for i in range(1,popSize):

        print()

        print()

        print("qpv = ",i," : ")

        print()

        for j in range(1,genomeLength):

            print(qpv[i, j, 0],end="")

            print(" ",end="")

        print()

        for j in range(1,genomeLength):

            print(qpv[i, j, 1],end="")

            print(" ",end="")

    print()

    

#Obverse观测函数，坍塌阈值p_alpha

def Measure(p_alpha):#Obverse

    for i in range(1,popSize):

        #print()

        for j in range(1,genomeLength):

            if p_alpha<=qpv[i, j, 0]:

                chromosome[i,j]=0

            else:

                chromosome[i,j]=1

            #print(chromosome[i,j]," ",end="")

        #print()

    #print()



#适应度计算函数

def Fitness_evaluation(generation):

    i=1; j=1; fitness_total=0; sum_sqr=0;

    fitness_average=0; variance=0;
  


    for i in range(1,popSize):

        fitness[i]=0
        
#根据染色体构建量子神经网络
        qnn_results = train_qnn(chromosome[i],i,EPOCHS=10)
    
    #------------------------------------QNN训练结束------------------------------
    
        fitness[i] = qnn_results[1]


      

        #print("fitness = ",f," ",fitness[i])

        fitness_total=fitness_total+fitness[i]

    fitness_average=fitness_total/N
    av_fitness.append(fitness_average)

    i=1;

    while i<=N:

        sum_sqr=sum_sqr+pow(fitness[i]-fitness_average,2)

        i=i+1

    variance=sum_sqr/N

    if variance<=1.0e-4:

        variance=0.0

    # Best chromosome selection

    the_best_chrom=0;

    fitness_max=fitness[1];

    for i in range(1,popSize):

        if fitness[i]>=fitness_max:

            fitness_max=fitness[i]

            the_best_chrom=i

    best_chrom[generation]=the_best_chrom
    fitness_best.append(fitness[the_best_chrom])
    best_acc[generation] = fitness[the_best_chrom]
    best_arch[generation] = chromosome[the_best_chrom]
    

    # Statistical output

    #f = open("output.dat","a")

    #f.write(str(generation)+" "+str(fitness_average)+"\n")

    #f.write(" \n")

    #f.close()

    print("Population size = ",popSize - 1)

    print("mean fitness = ",fitness_average)

    print("variance = ",variance," Std. deviation = ",math.sqrt(variance))
    
    print("fitness max = ",fitness[the_best_chrom])

    #print("fitness sum = ",fitness_total)



#量子旋转门更新

def rotation():

    rot=np.empty([2,2])

    # Lookup table of the rotation angle

    for i in range(1,popSize):

        for j in range(1,genomeLength):
            #print("---------here-----------")
            #print(fitness[i])
            #print(best_chrom[generation])
            #b_g = int(best_chrom[generation])
            #best_chrom[generation] = b_g
            #print(best_chrom[generation])
            #print(fitness[best_chrom[generation]])
            #print("---------here-----------")

            if fitness[i]<fitness[int(best_chrom[generation])]:

# if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:

                if chromosome[i,j]==0 and chromosome[int(best_chrom[generation]),j]==1:

# 旋转角0.03pi

                    delta_theta=0.0942477796


                    rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);

                    rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])

                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])

                    qpv[i,j,0]=round(nqpv[i,j,0],2)

                    qpv[i,j,1]=round(1-nqpv[i,j,0],2)

                if chromosome[i,j]==1 and chromosome[int(best_chrom[generation]),j]==0:



                    delta_theta=-0.0942477796

                    rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);

                    rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])

                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])

                    qpv[i,j,0]=round(nqpv[i,j,0],2)

                    qpv[i,j,1]=round(1-nqpv[i,j,0],2)

             # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:




#量子染色体突变

def mutation(pop_mutation_rate, mutation_rate):

    

    for i in range(1,popSize):

        up=np.random.random_integers(100)

        up=up/100

        if up<=pop_mutation_rate:

            for j in range(1,genomeLength):

                um=np.random.random_integers(100)

                um=um/100

                if um<=mutation_rate:

                    nqpv[i,j,0]=qpv[i,j,1]

                    nqpv[i,j,1]=qpv[i,j,0]

                else:

                    nqpv[i,j,0]=qpv[i,j,0]

                    nqpv[i,j,1]=qpv[i,j,1]

        else:

            for j in range(1,genomeLength):

                nqpv[i,j,0]=qpv[i,j,0]

                nqpv[i,j,1]=qpv[i,j,1]

    for i in range(1,popSize):

        for j in range(1,genomeLength):

            qpv[i,j,0]=nqpv[i,j,0]

            qpv[i,j,1]=nqpv[i,j,1]

            
            
#全干扰交叉
def all_cross():
    qpv_new = qpv
    qpv_a = qpv
    qpv_swap = qpv[1,1]
    print(qpv[2,2])

    for i in range(1,popSize):
        for j in range(1,genomeLength):
            if j >=2:
                k = i-j+1
                if i-j+1 <=0:#当需交换的下标为0或小于等于0时再向前一步
                    k=k-1
                qpv_swap = qpv_a[i,k]
                qpv_new[i,j] = qpv_swap
                qpv[i,j] = qpv_new[i,j]
    print(qpv[2,2])



#画图
def plot_Output():

    #data = np.loadtxt('output.dat')

    # plot the first column as x, and second column as y

    x=range(0,generation_max)

    y=fitness_best

    plt.plot(x,y)

    plt.xlabel('Generation')

    plt.ylabel('Fitness best')

    plt.xlim(0.0, 100.0)

    plt.show()



#QGA主程序

def Q_GA():
    

    generation=0;

    print("============== GENERATION: ",generation," =========================== ")

    print()

    Init_population()

    #Show_population()

    Measure(0.5)

    Fitness_evaluation(generation)

    while (generation<generation_max-2):#此处有逻辑问题，两次覆盖generation=0的情况！

        print("The best of generation [",generation,"] ", best_chrom[generation])

        print()

        print("============== GENERATION: ",generation+1," =========================== ")

        print()
        
        generation=generation+1

        rotation()

        mutation(0.01,0.002)
        
        #if(generation>20 and fitness_best[generation]==fitness_best[generation-10]):
            #a=1
        #all_cross()
        #print(chromosome)
        #print("zhixing!!")
            
        Measure(0.5)
        
        Fitness_evaluation(generation)




Q_GA()
print("-------------------------final-------------")
print("best_acc:",best_acc)
print("best_arch:",best_arch)
print("-------------final_training------------------")
best_acc[-1]=0
best_acc_list = best_acc.tolist()
#max_index = np.where(best_acc==np.max(best_acc))
max_index = best_acc_list.index(max(best_acc_list))+1
print(max_index)
final_qnn_results = train_qnn(best_arch[max_index],max_index,EPOCHS=40)

with open('best.txt','w') as f:
    f.write("best_acc: \n")
    f.write(str(best_acc_list))
    f.write("best_arch: \n")
    f.write(str(best_arch))
    f.write("best_arch_para: \n")
    f.write(str(best_arch_para))


#plot_Output()