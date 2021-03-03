"""
This part is done in the Colab, here it is just a copy.
I have tried the optimizer as Adam, AdamP, adabound, Ranger and RAdam
activate function as ReLu and LeakyRelu
ReLu has jitter issue, adabound too slow, Ranger has overfitting, AdamP is like Adam

Final version is RAdam, and leakyRelu
"""

#%%
from utils import *
import torch_optimizer as optim
opt = {}
opt['n_epochs'] = 25
opt['lr'] = 0.001

model = UNet(n_channels=1, n_classes=1).cuda()
optimizer = optim.RAdam(
    model.parameters(),
    lr= opt['lr'],
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)
criterion = nn.MSELoss()
tr = torch.load('../data/data_ss/fhss1_tr_200.pt')  # x, y, l
va = torch.load('../data/data_ss/fhss1_va_200.pt')


#%%
loss_train = []
loss_cv = []

for epoch in range(opt['n_epochs']):
    
    model.train()
    for i, (x, y, l) in enumerate(tr): 
        out = model(x.unsqueeze(1).cuda())
        optimizer.zero_grad()  

        loss = criterion(out.squeeze(), y.cuda())              
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.item())
        torch.cuda.empty_cache()
        if i%50 == 0: print(f'Current iter is {i} in epoch {epoch}')
 
    model.eval()
    with torch.no_grad():
        cv_loss = 0
        for xval, yval, lval in va: 
            cv_cuda = xval.unsqueeze(1).cuda()
            cv_yh = model(cv_cuda).cpu().squeeze()
            cv_loss = cv_loss + Func.mse_loss(cv_yh, yval)
            torch.cuda.empty_cache()
        loss_cv.append(cv_loss/106)  # averaged over all the iterations
    
    if epoch%1 ==0:
        plt.figure()
        plt.plot(loss_train[-1400::50], '-x')
        plt.title('train loss per 50 iter in last 1400 iterations')

        plt.figure()
        plt.plot(loss_cv, '--xr')
        plt.title('val loss per epoch')
        plt.show()
    
    torch.save(model.state_dict(), './f1_unet'+str(epoch)+'.pt')
    print('current epoch is ', epoch)


# %% test part
va = torch.load('../data/data_ss/fhss1_va_200.pt')
a = next(iter(va))
a[0].shape
model = UNet(n_channels=1, n_classes=1).cuda()
model.load_state_dict(torch.load('./models/f1_unet4.pt'))  # l1+l2
model.eval()

with torch.no_grad():
    xte, yte, l = a

    te_cuda = xte.unsqueeze(1).cuda()
    te_yh = model(te_cuda).cpu().squeeze()
    torch.cuda.empty_cache()

    plt.figure()
    plt.imshow(xte[1], interpolation='None')
    plt.colorbar()
    plt.title('Input')

    plt.figure()
    plt.imshow(te_yh[1], interpolation='None')
    plt.colorbar()
    plt.title('Output')

    plt.figure()
    plt.imshow(yte[1], interpolation='None')
    plt.colorbar()
    plt.title('Ground Truth')