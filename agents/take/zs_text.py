"""Encode a bank of heaviness/fullness prompts with the same CLIP-L and SigLIP text towers whose
image embeddings are cached in shared_cache; dot products are computed later on CPU."""
import json
from pathlib import Path
import numpy as np
import open_clip
import torch
import torch.nn.functional as Fn

OUT = Path("/home/cleaner/Cleaner/agents/take/cache")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CONCEPTS = {
    "full_bottle": ["a full bottle of liquid", "a bottle filled with liquid", "an unopened bottle of drink",
                    "a sealed full plastic bottle", "a bottle full of water", "a full glass bottle of beer"],
    "empty_bottle": ["an empty bottle", "an empty transparent plastic bottle", "an empty glass bottle",
                     "a crushed empty bottle", "an empty drink container"],
    "empty_can": ["an empty aluminium can", "a crushed soda can", "an empty tin can", "a light aluminium can"],
    "metal_block": ["a solid metal block", "a heavy steel block", "a solid aluminium bar", "a heavy piece of machined metal",
                    "a cast iron part", "a heavy metal bracket"],
    "rock": ["a heavy rock", "a large stone", "a brick", "a piece of concrete", "a heavy boulder"],
    "bag": ["a plastic bag", "a crumpled plastic bag", "a light plastic film", "a garbage bag", "a shopping bag"],
    "paper_light": ["crumpled paper", "a tissue", "a paper towel", "a cardboard box", "an empty cardboard box",
                    "a toilet paper roll", "a light paper package"],
    "foam_light": ["a styrofoam piece", "a foam block", "a sponge", "a hollow plastic object", "a light plastic toy"],
    "solid_plastic": ["a solid heavy plastic block", "a thick rubber part", "a heavy rubber object", "a dense plastic machine part",
                      "a solid rubber block", "a heavy black plastic part"],
    "heavy": ["a heavy object", "a dense heavy object", "something very heavy to lift", "a massive solid object",
              "a heavy weight"],
    "light": ["a lightweight object", "a very light object", "something easy to lift with one hand", "a hollow light object",
              "a flimsy object"],
    "hollow": ["a hollow object", "an empty container", "an empty box", "an empty package"],
    "solid": ["a solid object", "a solid chunk of material", "a dense solid object", "a filled container"],
    "liquid": ["a container with liquid inside", "visible liquid inside a bottle", "a bottle with a liquid level"],
    "transparent": ["a transparent object", "a clear see-through container"],
    "crushed": ["a crushed object", "a dented crumpled container", "a squashed object"],
    "intact": ["an intact undamaged object", "a new unopened product"],
    "tool_part": ["a heavy tool", "a machine part", "an industrial metal component", "a car part"],
    "powder": ["a pile of sand", "a pile of sawdust", "wood shavings", "a pile of small chips"],
    "screws": ["a pile of metal screws and bolts", "a pile of nuts and bolts", "metal hardware"],
}
TEMPLATES = ["a photo of {} on the floor.", "a close-up photo of {}.", "{} lying on a tiled floor.", "a photo of {}."]


@torch.no_grad()
def encode(name, pretrained):
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer(name)
    T, P = [], []
    for c, prompts in CONCEPTS.items():
        txt = [t.format(p) for p in prompts for t in TEMPLATES]
        e = Fn.normalize(model.encode_text(tok(txt).to(DEV)).float(), dim=-1)
        T.append(Fn.normalize(e.mean(0), dim=0).cpu().numpy())
        # per-prompt (template-averaged) too
        e = e.view(len(prompts), len(TEMPLATES), -1).mean(1)
        P.append(Fn.normalize(e, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(T), np.concatenate(P)


def main():
    out = {}
    for tag, (n, p) in {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}.items():
        T, P = encode(n, p)
        out[f"{tag}_concepts"] = T
        out[f"{tag}_prompts"] = P
        print(tag, T.shape, P.shape, flush=True)
    names = list(CONCEPTS)
    prompts = [p for c in CONCEPTS for p in CONCEPTS[c]]
    prompt_concept = [c for c in CONCEPTS for _ in CONCEPTS[c]]
    np.savez_compressed(OUT / "zs_text.npz", **out, concepts=np.array(names), prompts=np.array(prompts),
                        prompt_concept=np.array(prompt_concept))
    json.dump(CONCEPTS, open(OUT / "zs_concepts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
