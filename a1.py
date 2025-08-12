import math
import random




data = [
    [[180, 85], 1], [[165, 55], 0], [[192, 92], 1], [[170, 60], 0], [[178, 75], 1],
    [[160, 48], 0], [[185, 80], 1], [[155, 45], 0], [[190, 88], 1], [[163, 52], 0],
    [[182, 83], 1], [[167, 58], 0], [[195, 90], 1], [[172, 62], 0], [[176, 72], 1],
    [[158, 50], 0], [[183, 82], 1], [[157, 47], 0], [[188, 86], 1], [[161, 53], 0],
    [[181, 84], 1], [[166, 56], 0], [[193, 91], 1], [[171, 61], 0], [[177, 74], 1],
    [[159, 49], 0], [[184, 81], 1], [[156, 46], 0], [[189, 87], 1], [[162, 54], 0],
    [[179, 77], 1], [[164, 57], 0], [[191, 89], 1], [[169, 59], 0], [[175, 70], 1],
    [[168, 63], 0], [[186, 79], 1], [[154, 44], 0], [[187, 78], 1], [[152, 42], 0],
    [[200, 95], 1], [[150, 40], 0], [[198, 93], 1], [[148, 38], 0], [[196, 91], 1],
    [[146, 36], 0], [[194, 90], 1], [[144, 34], 0], [[197, 92], 1], [[142, 32], 0],
    [[199, 94], 1], [[140, 30], 0], [[201, 96], 1], [[138, 28], 0], [[203, 98], 1],
    [[136, 26], 0], [[205, 100], 1], [[134, 24], 0], [[207, 102], 1], [[132, 22], 0],
    [[209, 104], 1], [[130, 20], 0], [[211, 106], 1], [[128, 18], 0], [[213, 108], 1],
    [[126, 16], 0], [[215, 110], 1], [[124, 14], 0], [[217, 112], 1], [[122, 12], 0],
    [[219, 114], 1], [[120, 10], 0], [[221, 116], 1], [[118, 8], 0], [[223, 118], 1],
    [[116, 6], 0], [[225, 120], 1], [[114, 4], 0], [[227, 122], 1], [[112, 2], 0],
    [[229, 124], 1], [[110, 1], 0], [[231, 126], 1], [[108, 0], 0], [[233, 128], 1],
    [[106, -2], 0], [[235, 130], 1], [[104, -4], 0], [[237, 132], 1], [[102, -6], 0],
    [[239, 134], 1], [[100, -8], 0], [[241, 136], 1], [[98, -10], 0], [[243, 138], 1],
    [[96, -12], 0], [[245, 140], 1], [[94, -14], 0], [[247, 142], 1], [[92, -16], 0]
]

imput_s =2
count_neuro = 2
result_output=0
learn_rate = 0.1
ex2 =2000

neuro=[]
b=[]
weights = []
neuro = [0]*count_neuro

heights = [d[0][0] for d in data]
weights_val = [d[0][1] for d in data]
min_h, max_h = min(heights), max(heights)
min_w, max_w = min(weights_val), max(weights_val)

normalized_data = []
for d in data:
    norm_h = (d[0][0] - min_h) / (max_h - min_h)
    norm_w = (d[0][1] - min_w) / (max_w - min_w)
    normalized_data.append([[norm_h, norm_w], d[1]])



for _ in range(imput_s+1):
    neuron_weights = random.random()
    b.append(neuron_weights)


for _ in range(imput_s):
    neuron_weights = [random.random() for _ in range(imput_s)]
    weights.append(neuron_weights)

output_weights = [random.random() for _ in range(count_neuro)]

def activ(result):
    return 1/(1+math.e**(-result))

def dervi_active(x):
    fx = activ(x)
    return fx*(1-fx)

def forward_pass(x):
    global result_output
    result_output = 0  
    for  i in range(count_neuro):
        result = 0
        for j in range(imput_s):
            result += (x[j]* weights[i][j])
        neuro[i] = activ(result+(b[i]))

    for k in range(count_neuro):
        result_output += neuro[k] *output_weights[k]

    output = activ(result_output+b[-1])

    return output


def train ():
    for ex in range(ex2):
        loss=0
        for x , y_true in normalized_data:
            
            output = forward_pass(x)
            y_pred = output

            error = y_pred - y_true
            loss += error **2

            del_output  = error * dervi_active(y_pred)

            for i in range(count_neuro):
                output_weights[i] -= learn_rate * del_output * neuro[i]

            b[-1]-= learn_rate * del_output

            delta_hidden = [0.0] * count_neuro

            for  i in range(count_neuro):
                erorr_h = del_output * output_weights[i]
                delta_hidden[i]=erorr_h * dervi_active(neuro[i])

            for i in range(count_neuro):
                for j in range(imput_s):
                    weights[i][j] -= learn_rate * delta_hidden[i] * x[j]

                b[i] -= learn_rate * delta_hidden[i]


        if ex % 100 == 0:
            print("Epox",ex,"loos:",loss)   

train()
norm=0
            


print("\nРезультаты после обучения:")
for i, (x, y_true) in enumerate(normalized_data):
    y_pred = forward_pass(x)
    prediction = 1 if y_pred > 0.5 else 0
    print(f"Пример {i+1}: Прогноз: {y_pred:.4f} ({prediction}), Фактическое: {y_true}")
    if prediction == y_true:
        norm+=1


print(norm)
while True:


    try:
        height, weight = map(int, input().split())
        norm_h = (height - min_h) / (max_h - min_h)
        norm_w = (weight - min_w) / (max_w - min_w)

        otvet = forward_pass([norm_h, norm_w])

        if otvet > 0.5:
            print("Мужчина ", int(otvet*100),"%",sep="")
        else:
            print("Женщина", otvet)
    except ValueError:
        print("Вы ввели хуйню")