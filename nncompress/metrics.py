from tensorboardX import SummaryWriter
import torch

writer = SummaryWriter("logs/", flush_secs=5)


def report(step, loss, epoch, inp, out, pred):
    writer.add_scalar("loss", loss, step)
    writer.add_scalar("epoch", epoch, step)
    p = torch.argmax(pred, dim=1)  # BDL -> BL
    # n_wrong_bits
    total_bits = out.shape[0] * out.shape[1]
    n_wrong = (p != out).sum().float()
    writer.add_scalar("percent wrong", n_wrong / total_bits, step)
    writer.add_scalar("avg_target", out.float().mean(), step)
