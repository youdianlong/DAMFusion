import torch

from torch.utils.data import DataLoader

from dataloader import FusionData
from losses import FusionLoss
from network import DAMFusion

device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")


def train():
    task = 'IVF'  # or 'MIF
    ir_path = '...'
    vi_path = '...'
    model_path = "..."

    batch_size = 8
    epochs = 3000
    lr = 8e-5 if task == 'IVF' else 4e-5
    optim_step = 600 if task == 'IVF' else 800
    optim_gamma = 0.85 if task == 'IVF' else 0.9
    weight_decay = 0

    dataset = FusionData(ir_path, vi_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DAMFusion().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

    Loss = FusionLoss(task=task).to(device)

    for epoch in range(epochs):
        model.train()

        for i, (vi, ir) in enumerate(dataloader):
            vi = vi.to(device)
            ir = ir.to(device)

            optimizer.zero_grad()

            fuse = model(ir, vi)

            total_loss, grad_loss, int_loss, ssim_loss = Loss(ir, vi, fuse)

            total_loss.backward()
            optimizer.step()

            mesg = (f"Epoch:[{epoch + 1}/{epochs}]    Batch:[{i}/{len(dataloader)}]    Loss_total: {total_loss:.4f}    "
                    f"Loss_int: {int_loss:.4f}     Loss_grad: {grad_loss:.4f}     Loss_ssim: {ssim_loss:.4f}")

            print(mesg)

        scheduler.step()

        if (epoch + 1) == epochs:
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    train()
