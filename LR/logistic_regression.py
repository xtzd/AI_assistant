# complete the logistic regression code in "Python's way" as well.
# Tips: It's almost like the linear regression code.
# The only difference is you need to complete a sigmoid function
# and use the result of that as your "new X"
# and also you need to generate your own training data.

import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import random

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def inference(w, b, x):
    p = sigmoid(w * np.asarray(x) + b)
    pred_y = np.asarray([0 if i<0.5 else 1 for i in p])
    return pred_y

def eval_loss(w, b, x, gt_y):
    pred_y = inference(w, b, x)
    loss = -gt_y * np.log(sigmoid(w*x+b)) - (1-gt_y) * np.log(1-sigmoid(w*x+b))
    loss = sum(loss)
    return loss

def gradient(pred_y, gt_y, x, w, b):
    x = np.asarray(x)
    diff = gt_y - sigmoid(w*x+b)
    dw = diff * x
    db = diff
    return dw, db

def cal_step_gradient(batch_x, batch_gt_y, w, b, lr):
    batch_size = len(batch_x)
    pred_y = inference(w, b, batch_x)
    dw, db = gradient(pred_y, batch_gt_y, batch_x, w, b)
    avg_dw = sum(dw)/batch_size
    avg_db = sum(db)/batch_size
    w = w + lr * avg_dw
    b = b + lr * avg_db
    
    return w, b

def train(x, y, batch_size, lr, max_iter):
    w, b = 0, 0
    num = len(x)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num, batch_size)
        batch_x = [x[j] for j in batch_idxs]
        batch_y = [y[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        pred_y = inference(w, b, x)
        print('w:{0},b:{1}'.format(w,b))
        print('loss:', eval_loss(w, b, x, y))
        print('acc:', sum([int(i==j) for (i,j) in zip(pred_y, y)])/len(y))

def gen_sample_data():
    w = -10 + random.random()
    b = 200  + random.random()
    print('gen_w:{0},gen_b:{1}'.format(w,b))
    num_sample = 100
    x = np.random.randint(1, 50, num_sample)
    p = sigmoid(w*x+b)
    y = np.asarray([0 if i<0.5 else 1 for i in p])
    print(x)    
    print(y)
    return x, y


def run():
    x, y = gen_sample_data()
    batch_size = 100
    lr = 0.01
    max_iter = 300
    train(x, y, batch_size, lr, max_iter)

if __name__=='__main__':
    run()