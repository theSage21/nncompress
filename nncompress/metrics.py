from tensorboardX import SummaryWriter

writer = SummaryWriter("logs/")


def report(step, loss, epoch):
    writer.add_scalar("loss", loss, step)
    writer.add_scalar("epoch", epoch, step)
