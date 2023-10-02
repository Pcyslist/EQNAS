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
print(images)
print(len(images))
p1 = plt.imshow(images[0,:,:,0],cmap='gray')

images_test = []
for f in glob.iglob("/home/lyy/327/zpx/copy1/test_burke/*"):
    images_test.append(np.asarray(Image.open(f).convert('L')))
for f in glob.iglob("/home/lyy/327/zpx/copy1/test_nimitz/*"):
    images_test.append(np.asarray(Image.open(f).convert('L')))
images_test = np.array(images_test)
images_test = images_test/255
images_test = images_test[..., np.newaxis]#多加一维，与tfq保持一致
print(images_test)
print(len(images_test))
plt.imshow(images_test[0, :, :,0],cmap='gray')
plt.colorbar()
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

plt.imshow(x_train_small[4,:,:,0],cmap='gray')
plt.colorbar()


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
    """
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass
    """

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
x_test_nocon,y_test = remove_contradicting(x_test_small,y_test)
#print(x_train_nocon[0])
print("----------------------")
#编码为量子线路
#二值化阈值设置
"""THRESHOLD = 0.5

x_train_bin = np.array(x_train_nocon> THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)
#print(len(x_train_bin))
print("=================================")
#_ = remove_contradicting(x_train_bin, y_train_nocon)
print("++++++++++++++++++++++++++")
#将矛盾例子删除
x_train_bin, y_train_nocon = remove_contradicting(x_train_bin, y_train_nocon)
x_test_bin, y_test = remove_contradicting(x_test_bin,y_test)
print("train len:",len(x_train_bin))
#print(len(y_train_nocon))
print("test len:",len(x_test_bin))"""

#编码经典图像在量子线路中
def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.rx(value*np.pi)(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x) for x in x_train_nocon]
x_test_circ = [convert_to_circuit(x) for x in x_test_nocon]
#print(x_train_bin[35])
#print("label:",y_train_nocon[35],"len:",len(y_train_nocon))
#x_train_circ = convert_to_circuit(x_train_bin[0])
print(len(x_train_circ))
print(x_train_circ[0])
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
            
#调用class 构建对象 !!!
#demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(4,1),
                                   #readout=cirq.GridQubit(-1,-1))

#circuit = cirq.Circuit()
#XX是两个X门的张量积
#demo_builder.add_layer(circuit, gate = cirq.XX, prefix='xx')
#SVGCircuit(demo_builder)


def create_quantum_model():
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
    builder.add_layer(circuit, cirq.XX, "xx1")
    #builder.add_layer(circuit, cirq.YY, "yy1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    #准备一个读出比特
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)
model_circuit, model_readout = create_quantum_model()
#model_circuit.ParamResolver({xx1-0:0.1})
#print(model_readout)
SVGCircuit(model_circuit)


#将线路加载至keras模型中
#创建keras模型
init = tf.keras.initializers.RandomNormal(mean=5, stddev=3, seed=42)
PQC1 = tfq.layers.PQC(model_circuit, model_readout,initializer=init)
#PQC1 = tfq.layers.PQC(model_circuit, model_readout)
print("---------------sym value---------------")
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

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

#编译keras模型
model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(lr=0.02,epsilon=1e-08),
    metrics=[hinge_accuracy])
print(model.summary())
print(model.get_weights())#打印模型初始化的参数
#print(model.get_layer())


#train
EPOCHS = 30
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
print(model.weights)
print(tfq.util.get_circuit_symbols(model_circuit))
print("------------------------")
print(PQC1.symbol_values())
#model.save_weights("D:/qnn_weights/1/qnn_w.h5")
print("*********************************")
print(qnn_results)