import os

print('start dl')

# train_dir = '/output/train'
# if not os.path.isdir(train_dir):
#     os.makedirs(train_dir)
#     print('create train dir')
# else:
#     print('true')
print('current working dir [{0}]'.format(os.getcwd()))
w_d = os.path.dirname(os.path.abspath(__file__))
print('change wording dir to [{0}]'.format(w_d))
os.chdir(w_d)

for l in os.popen('/bin/bash -c "cd {0} && source ./scripts/train_cifarnet_on_cifar10.sh"'.format(w_d)):
    print(l.strip())
