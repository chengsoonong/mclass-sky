# For counting the number of rejections made in rejection sampling

reject_counter = 0
milestone = 10

def reset():
    global reject_counter
    global milestone
    reject_counter = 0
    milestone = 10
    print('**** Starting or restarting experiment ****')