"""CLIP-L (dfn2b) + SigLIP image/text encoders, fp16, flip-TTA; same crops as audit/mat_features2.crop."""
import numpy as np
import open_clip
import torch
import torch.nn.functional as Fn

from common import MODELS, PROMPT_SETS, TEMPL

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Encoders:
    @torch.no_grad()
    def __init__(self, tags=("clipL", "siglip")):
        self.m, self.text = {}, {}
        for tag in tags:
            name, pre_name = MODELS[tag]
            model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pre_name, device=DEV)
            model.eval()
            tok = open_clip.get_tokenizer(name)
            for sname, (pb, ps) in PROMPT_SETS.items():
                T = []
                for plist in (pb, ps):
                    e = Fn.normalize(model.encode_text(tok([t.format(p) for p in plist for t in TEMPL]).to(DEV)).float(), dim=-1).mean(0)
                    T.append(Fn.normalize(e, dim=0))
                self.text[(tag, sname)] = torch.stack(T).cpu().numpy()
            model.half()
            size = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
            mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
            std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
            self.m[tag] = (model, size, mean, std)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def embed(self, tag, imgs):
        model, size, mean, std = self.m[tag]
        out = []
        for i in range(0, len(imgs), 32):
            x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + 32])).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = ((Fn.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) - mean) / std).half()
            f = model.encode_image(x).float() + model.encode_image(torch.flip(x, dims=[3])).float()
            out.append(Fn.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)
