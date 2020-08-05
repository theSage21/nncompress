from nncompress.model import Compressor
import random
from nncompress.metrics import report
from tqdm import tqdm
import json
import torch
import os


def compress(args):
    with open(args.config, "r") as fl:
        cfg = json.loads(fl.read())
    for d in "savepoints logs workspace".split():
        try:
            os.mkdir(d)
        except OSError:
            pass
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
            os.remove(f"{wk}/{known[0]}")
        cfg["current_step"] = step
        with open(args.config, "w") as fl:
            fl.write(json.dumps(cfg, indent=2))

    def data_generator(with_cache):
        """
        Yield input/output batches
        """
        if with_cache:
            idxes = list(with_cache.keys())
            random.shuffle(idxes)
            I, O = [], []
            for i in idxes:
                inp, out = with_cache[i]
                I.append(inp)
                O.append(out)
                if len(I) >= batchsize:
                    I, O = torch.stack(I), torch.stack(O)
                    for _ in repetitions:
                        yield I, O
                    I, O = [], []
            if I:
                I, O = torch.stack(I), torch.stack(O)
                for _ in repetitions:
                    yield I, O
            return
        fname = args.fname
        chunksize = cfg.get("chunksize", 512)
        batchsize = cfg.get("batchsize", 32)
        repetitions = [None] * cfg.get("repetitions", 5)
        symbol = {"0": 0, "1": 1, "START": 2, "END": 3}
        I, O = [], []
        seqlen = chunksize * 8
        last_bits = torch.Tensor([symbol["START"]] * seqlen).long()
        rowid = 0
        with open(fname, "rb") as fl:
            while True:
                chunk = fl.read(chunksize)
                bits = (
                    torch.Tensor([symbol["END"]] * seqlen)
                    if len(chunk) < chunksize
                    else torch.Tensor(
                        [symbol[i] for bt in chunk for i in format(bt, "b").zfill(8)]
                    )
                ).long()
                with_cache[rowid] = (last_bits, bits)
                rowid += 1
                I.append(last_bits)
                O.append(bits)
                last_bits = bits
                if len(chunk) < chunksize:
                    break
                if len(I) >= batchsize:
                    I, O = torch.stack(I), torch.stack(O)
                    for _ in repetitions:
                        yield I, O
                    I, O = [], []
        if I:
            I, O = torch.stack(I), torch.stack(O)
            for _ in repetitions:
                yield I, O

    # Train
    epoch = 0
    with_cache = {}
    while True:
        # TODO: refresh config each epoch to pick up changes
        with tqdm(desc="Starting") as pbar:
            for inp, out in data_generator(with_cache):
                opt.zero_grad()
                p = net(inp)
                B, L, D = p.shape
                p = p.reshape(B, D, L)
                loss = loss_fn(p, out)  # BDL, BL
                loss.backward()
                opt.step()
                if step % cfg["log_step"] == 0:
                    report(step, loss, epoch, inp, out, p)
                if step % cfg["save_step"] == 0:
                    save()
                step += 1
                pbar.set_description(f"Loss: {loss}")
                pbar.update(1)
        epoch += 1
        print("\n")
