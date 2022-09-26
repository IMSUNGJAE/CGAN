import os
from argsparse_file import args
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
from dataloader import custom_dataset
from network import *
from data_preprocessing import all_Data,origin_data,normalized_data,nan_data,de_normalized_data
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# print(args)

# USE_CUDA = torch.cuda.is_available()
# print(USE_CUDA)
#
# device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('학습을 진행하는 기기:',device)
# print('cuda index:', torch.cuda.current_device())
# print('gpu 개수:', torch.cuda.device_count())
# print('graphic name:', torch.cuda.get_device_name())

cuda = True if torch.cuda.is_available() else False
cuda = torch.device('cuda')

def weights_init_normal(m):

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main():

    # cudnn.benchmark = True  #cudnn의 benchmark를 통해 최적의 backend 연산을 찾는 flag를 true로 하겠다는 의미

    train_data = normalized_data(args.train_data_path)
    validation_data = normalized_data(args.validation_data_path)

    L_vector, L_index = nan_data()


    """=============== train data loader ==============="""
    train_dataset = custom_dataset(train_data, L_vector,L_index)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers= args.num_workers,
                              shuffle=True,
                              drop_last=True)

    """============= validation data loader ============"""
    validation_dataset = custom_dataset(validation_data, L_vector ,L_index)
    validation_loader = DataLoader(dataset=validation_dataset,
                              batch_size=args.val_batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True)


    adversarial_loss = torch.nn.MSELoss()
    validation_loss = torch.nn.MSELoss()

    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    Gen_list=[]
    Dis_list=[]
    Val_list=[]

    for epoch in tqdm(range(args.n_epochs)):

        valid_loss, gen_loss, dis_loss = 0.0, 0.0, 0.0
        MAE = np.array([])

        for i, (ground_truth, nan_value, nan_value_index, label) in enumerate(train_loader):

            valid = Variable(Tensor(ground_truth.shape[0], 1).fill_(1.0), requires_grad=False) # variable = tensor에 대한 data, grad, backward 총 3개의 파라미터로 구성
            fake = Variable(Tensor(ground_truth.shape[0], 1).fill_(0.0), requires_grad=False)

            ground_truth = ground_truth.float().cuda()
            nan_value = nan_value.float().cuda()

            real_data = Variable(Tensor(ground_truth.reshape(args.batch_size,16)))
            label = Variable(label.type(LongTensor))

            optimizer_G.zero_grad()

            z = Variable(Tensor(nan_value.reshape(args.batch_size,16)))
            gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, args.batch_size)))

            gen_data = generator(z, gen_labels)

            # cgan  은 여기서 discriminator  validity 검사함 왬?
            validity = discriminator(gen_data, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()


            """==============================================================="""

            validity_real = discriminator(real_data, label)
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_data.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            """==============================================================="""

            d_loss = (d_real_loss + d_fake_loss)/2

            d_loss.backward()
            optimizer_D.step()

            gen_loss += g_loss.item()
            dis_loss += d_loss.item()

        print("[Epoch %d/%d] [Iteration %d] [G loss: %f] [D loss: %f]"
            % (epoch, args.n_epochs, len(train_loader), gen_loss/len(train_loader), dis_loss/len(train_loader))
            )


        generator.eval()
        # 학습할때만 필요한 Dropout,Bathnorm 기능을 비활성화 시킴
        with torch.no_grad():

            # #gradient 계산 context를 비활성화 해주는 역할 -> 비활성화되면 더이상 gradient를 트래킹하지 않음
            # for k, (val_ground_truth, val_nan_value, val_nan_index,val_label) in enumerate(validation_loader):
            #
            #     val_ground_truth = val_ground_truth.float().cuda()
            #     val_nan_value = val_nan_value.float().cuda()
            #     val_nan_index = val_nan_index.float().cuda()
            #
            #     val_real_data = Variable(Tensor(val_ground_truth.reshape(args.val_batch_size,16)))
            #
            #
            #     val_z = Variable(Tensor(val_nan_value.reshape(args.val_batch_size,16)))
            #     val_label = Variable(val_label.type(LongTensor))
            #
            #     val_gen_data = generator(val_z,val_label)
            #
            #
            #     val_loss = validation_loss(val_real_data, val_gen_data)
            #     valid_loss += val_loss.item()
            #     """val real data 값이 0인경우 생각을 못함. """
            #
            #     real_data_denorm = de_normalized_data(val_real_data)
            #     gen_data_denorm = de_normalized_data(val_gen_data)
            #
            #     ae = abs(real_data_denorm - gen_data_denorm)
            #
            #     for i in range(args.val_batch_size):
            #         for j in val_nan_index[i]:
            #             MAE = np.append(MAE, ae.cpu()[i][int(j)])

            real = np.array([[-8.227700921,	30.39291462,	178.1083663,	166.2624113,	11.79088097,	18.81073629,	173.0616592,	153.3170075,	17.56060371,	11.8224508,	166.4792563,	151.783595,	3.149534351,	-4.497263869,	140.7145961,	134.1124605 ]])

            nan = np.array([[-8.227700921,	30.39291462,	178.1083663,	166.2624113,	11.79088097,	0,	173.0616592,	153.3170075,	17.56060371,	11.8224508,	166.4792563,	151.783595,	3.149534351,	0,	140.7145961,	134.1124605 ]])

            nan = (nan-np.min(all_Data()))/(np.max(all_Data())-np.min(all_Data()))

            real_norm = Variable(torch.FloatTensor(real)).cuda()

            val_z = Variable(Tensor(np.random.normal(0, 1, (1,16))))
            # val_z = Variable(torch.FloatTensor(nan)).cuda() # 결측 위치 : 1,4,15
            val_label = Variable(torch.LongTensor(np.arange(12,13))).cuda()

            val_gen_data = generator(val_z,val_label)

            generated_Data = de_normalized_data(val_gen_data)

            ae = abs(real_norm - generated_Data).sum()
            print("[MAE %f]" % (ae/16))
            print('val_gen_Data = ', generated_Data)

            # T_MAE = torch.FloatTensor(MAE)
            # print("[V loss %f] [MAE %f ]" % (valid_loss / len(validation_loader), T_MAE.sum() / (len(validation_loader) * args.batch_size * args.num)))


        generator.train()

        Gen_list.append(gen_loss/len(train_loader))
        Dis_list.append(dis_loss/len(train_loader))
        Val_list.append(valid_loss/len(validation_loader))

        """생성된 이미지 확인"""

        # save the model parameters for each epoch
        g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
        torch.save(generator.state_dict(), g_path)

    """플롯하는 부분분"""

    plt.rc('font', family='Times New Roman')
    plt.figure()

    plt.title("Train_loss")
    plt.plot(Gen_list,
             color='red',
             linestyle='--',
             linewidth=1, label='generator_loss')
    plt.plot(Dis_list,
             color='blue',
             linestyle='--',
             linewidth=1, label='discriminator_loss')

    plt.legend()
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    plt.title("Validation_loss")
    plt.plot(Val_list,
             color='green',
             linestyle='--',
             linewidth=1, label='validation_loss')

    plt.legend()
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    main()

