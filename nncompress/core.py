from nncompress.model import Compressor
from nncompress.metrics import report
from tqdm import tqdm
import json
import torch
import os


def compress(args):
    with open(args.config, "r") as fl:
        cfg = json.loads(fl.read())
    net = Compressor(**cfg["nn_kwargs"])
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    step = cfg["current_step"]
    # Helpers
    def save():
        if cfg["keep_max"] < 1:
            return
        wk = "savepoints"
        known = sorted([i for i in os.listdir(wk)], key=lambda x: int(x.split(".")[1]))
        path = f"{wk}/model.{step}"
        torch.save(net.state_dict(), path)
        if len(known) + 1 > cfg["keep_max"]:
            os.remove(known[0])

    def data_generator():
        """
        Yield input/output batches
        """
        fname = args.fname
        chunksize = cfg.get("chunksize", 512)
        batchsize = cfg.get("batchsize", 32)
        repetitions = [None] * cfg.get("repetitions", 5)
        symbol = {"0": 0, "1": 1, "START": 2, "END": 3}
        I, O = [], []
        seqlen = chunksize * 8
        last_bits = torch.Tensor([symbol["START"]] * seqlen)
        with open(fname, "rb") as fl:
            while True:
                chunk = fl.read(chunksize)
                bits = (
                    torch.Tensor([symbol["END"]] * seqlen)
                    if len(chunk) < chunksize
                    else torch.Tensor(
                        [symbol[i] for bt in chunk for i in format(bt, "b").zfill(8)]
                    )
                )
                I.append(last_bits.long())
                O.append(bits.long())
                last_bits = bits
                if len(chunk) < chunksize:
                    break
                if len(I) >= batchsize:
                    for _ in repetitions:
                        yield torch.stack(I), torch.stack(O)
                    I, O = [], []
        if i:
            for _ in repetitions:
                yield torch.stack(I), torch.stack(O)

    # Train
    epoch = 0
    while True:
        # TODO: refresh config each epoch to pick up changes
        with tqdm(desc="Starting") as pbar:
            for inp, out in data_generator():
                opt.zero_grad()
                p = net(inp)
                B, L, D = p.shape
                p = p.reshape(B, D, L)
                loss = loss_fn(p, out)  # BDL, BL
                loss.backward()
                opt.step()
                if step % cfg["log_step"] == 0:
                    report(step, loss, epoch)
                if step % cfg["save_step"] == 0:
                    save()
                step += 1
                pbar.set_description(f"Loss: {loss}")
                pbar.update(1)
        epoch += 1
